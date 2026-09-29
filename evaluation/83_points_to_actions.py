#!/usr/bin/env python3
"""Can the action be recovered from where the gripper's points are?

If perfect points cannot give the action back, a policy that predicts points is pointless.
Each test removes one idealisation. All recover the POSE change only (translation, rotation;
not the gripper opening), from the 68 palm points that are rigid with the wrist flange.

    1   exact 3D FK points          rigid fit (Kabsch)   is our geometry right? (expect ~0)
    2   exact 2D pixels (fk_uv)     PnP                  can it be solved at all? (coverage)
    2v  exact 2D pixels, only where Track-On sees the point   what visibility alone costs
    3   exact pixels + noise        PnP                  how accurate must a tracker be?
    4   Track-On's tracked pixels   PnP                  the realistic ceiling
    4r  Track-On's tracked pixels   robust PnP (RANSAC)  the same, ignoring jumped points

at horizons k = 1, 4, 8, 16 steps (7.2 Hz, so up to 2.2 s), on the frozen validation windows.
Every error is compared, on the same frame pairs, with MOMENTUM: repeating the previous
one-step action k times (at k = 1 this is the copy-last baseline, 4.55 mm).

Decision rule, fixed before running: points -> actions is viable at a horizon if test 4
beats momentum there. At k = 1 the trained policy's 6.49 mm is the other reference.

    $PY_POINTWORLD -B evaluation/83_points_to_actions.py --out $PTP_OUT/points_to_actions
Run with $PY_POINTWORLD (kinematics, cv2, h5py). CPU only; parallel over clips.
"""
from __future__ import annotations

import os

# Before numpy loads: BLAS/OpenMP read these once, at import. Spawned workers inherit them.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402

HORIZONS = (1, 4, 8, 16)
SIGMAS = (1, 2, 4, 8, 16)
VARIANTS = ("independent", "correlated", "dropout", "half_occluded")
WIN = 48
# per-pair row: k, translation error (mm), rotation error (deg), depth part, sideways part
# (mm, error vector in camera axes), momentum's translation and rotation error on the same pair
COLS = ("k", "trans_mm", "rot_deg", "depth_mm", "side_mm", "mom_trans_mm", "mom_rot_deg", "clip", "t")

_W = {}


def _init():
    # One thread per worker for every numerical library. Forked workers deadlocked in futex
    # (job 1718513): a child inherits locks from thread pools it cannot see. Workers are
    # spawned, not forked (main); BLAS threads are pinned at the top of the file.
    import cv2
    import torch
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    from droid_qp.pose_fit import gripper_model
    from droid_qp.robot_points import load_sampler
    s = load_sampler(128, gripper_only=True, seed=0)
    model, rigid = gripper_model(s)
    _W.update(sampler=s, model=model, rigid=rigid,
              shards=json.loads((DATA / "shard_index.json").read_text()))


def load_clip(vid):
    import h5py
    from droid_qp.actions import sixd_to_rot
    from droid_qp.robot_points import PW_FLOWS, window_calibration
    base = DATA / "gen" / _W["shards"][vid]
    with np.load(base / "actions/labels" / f"{vid}.npz") as a:
        lab = {k: a[k] for k in ("position", "rot_6d", "joint_positions", "gripper_positions",
                                 "delta_pos_body", "delta_rot6d_body", "valid")}
    lab["R"] = sixd_to_rot(lab["rot_6d"].astype(np.float64))
    lab["dR"] = sixd_to_rot(lab["delta_rot6d_body"].astype(np.float64))
    (rs,) = sorted((base / "robot_sam/query_points").glob(f"{vid}_f*.npz"))
    with np.load(rs) as r:
        fk_uv, fk_ok, rel = r["fk_uv"].astype(np.float64), r["fk_ok"], r["fk_release_frame"]
    if not np.array_equal(rel, np.arange(len(rel))):
        raise ValueError("FK steps are not the video frames (expected identity at 7.2 Hz)")
    (mg,) = sorted((base / "merged/query_points").glob(f"{vid}_f*.npz"))
    with np.load(mg) as m:
        ids = m["point_index"]
    with np.load(base / "trackon/tracks" / f"{vid}.npz") as t:
        tracks, vis, role = t["tracks"].astype(np.float64), t["visibility"], t["role"]
    cols = np.flatnonzero(role == 1)
    if len(cols) != len(ids):
        raise ValueError("robot columns and FK ids disagree")
    ep, cam = vid.rsplit("__", 1)
    with h5py.File(PW_FLOWS / f"{ep}_flows.h5", "r") as f:
        w0 = sorted((k for k in f.keys() if ":" in k), key=lambda k: int(k.split(":")[0]))[0]
        E, K = window_calibration(f, w0, cam)
    return lab, fk_uv, fk_ok, ids, tracks[:, cols], vis[:, cols], E, K


def run_clip(args):
    """One validation window. Any error is reported for this clip, never raised."""
    ci = args[0]
    try:
        return _run_clip(*args)
    except Exception as e:
        import traceback
        return ci, None, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=3)[-300:]}"


def _run_clip(ci, rec, seed):
    from droid_qp.geom import project
    from droid_qp.pose_fit import (kabsch, pnp, pnp_ransac, relative, repeat,
                                   rotation_error_deg, scaled_intrinsics)
    from droid_qp.robot_points import robot_points_3d
    model, rigid = _W["model"], _W["rigid"]
    vid, start = rec["video_id"], rec["start"]
    try:
        lab, fk_uv, fk_ok, ids, trk, tvis, E, Kc = load_clip(vid)
    except Exception as e:                                   # report, never silently drop
        return ci, None, f"{type(e).__name__}: {e}"
    K = scaled_intrinsics(Kc)
    n = min(len(lab["position"]), len(fk_uv), len(trk))
    lo, hi = max(0, start - 1), min(n, start + WIN + max(HORIZONS) + 1)
    F = range(lo, hi)
    info = {}

    # ---- test 1: exact 3D points, and the camera sanity check -----------------------------
    P = robot_points_3d(_W["sampler"], lab["joint_positions"][lo:hi], lab["gripper_positions"][lo:hi])
    poses = {"1": {}}
    fk_vs_rec = []
    for i, t in enumerate(F):
        R, p = kabsch(model[rigid], P[i][rigid])
        poses["1"][t] = (R, p)
        fk_vs_rec.append((1000 * np.linalg.norm(p - lab["position"][t]), rotation_error_deg(R, lab["R"][t])))
    uv_chk, _ = project(P.reshape(-1, 3), E, np.asarray(Kc), out_size=(854, 480))
    uv_chk = uv_chk.reshape(P.shape[0], -1, 2)
    ok = fk_ok[lo:hi]
    info["reprojection_max_px"] = float(np.abs(uv_chk[ok] - fk_uv[lo:hi][ok]).max()) if ok.any() else None
    info["fk_vs_recorded_mm"] = float(np.median([a for a, _ in fk_vs_rec]))
    info["fk_vs_recorded_deg"] = float(np.median([b for _, b in fk_vs_rec]))

    # ---- tests 2, 2v, 3, 4, 4r: PnP on palm pixels ---------------------------------------
    rng = np.random.default_rng([seed, ci])
    pal_trk = rigid[ids]                                     # tracked robot columns on the palm
    for key in ("2", "2v", "4", "4r") + tuple(f"3/{v}/{s}" for v in VARIANTS for s in SIGMAS):
        poses[key] = {}
    trk_err, collapsed = [], 0
    for t in F:
        m = rigid & fk_ok[t]
        poses["2"][t] = pnp(model[m], fk_uv[t][m], K)
        seen = pal_trk & tvis[t]                             # palm columns Track-On calls visible
        fid = ids[seen]
        exact = fk_ok[t][fid]
        poses["2v"][t] = pnp(model[fid[exact]], fk_uv[t][fid[exact]], K)
        poses["4"][t] = pnp(model[fid], trk[t][seen], K)
        if seen.sum() >= 6 and trk[t][seen].std(0).max() < 2.0:   # tracker collapsed the palm
            collapsed += 1
        poses["4r"][t] = pnp_ransac(model[fid], trk[t][seen], K)
        if exact.any():
            trk_err.extend(np.linalg.norm(trk[t][seen][exact] - fk_uv[t][fid[exact]], axis=1).tolist())
        mdl, uv = model[m], fk_uv[t][m]
        for v in VARIANTS:
            for s in SIGMAS:
                noise = rng.normal(0, s, uv.shape)
                if v == "correlated":                        # the whole cluster drifts together
                    noise = rng.normal(0, s, (1, 2)) + rng.normal(0, s / 4, uv.shape)
                keep = np.ones(len(uv), bool)
                if v == "dropout":
                    keep = rng.random(len(uv)) < 0.5
                elif v == "half_occluded" and len(uv):       # one contiguous side hidden
                    proj = mdl @ rng.normal(size=3)
                    keep = proj <= np.median(proj)
                poses[f"3/{v}/{s}"][t] = pnp(mdl[keep], (uv + noise)[keep], K)
    info["track_err_px_median"] = float(np.median(trk_err)) if trk_err else None
    info["collapsed_frames"], info["frames"] = collapsed, len(F)

    # ---- frame pairs -----------------------------------------------------------------------
    valid, Rrec, prec = lab["valid"], lab["R"], lab["position"].astype(np.float64)
    Rcam = E[:3, :3]
    rows = {key: [] for key in poses}
    rows["momentum"], rows["zero"] = [], []
    eligible = {k: 0 for k in HORIZONS}
    for t in range(start, min(start + WIN, n)):
        for k in HORIZONS:
            if t + k >= n or not valid[t:t + k].all():
                continue
            eligible[k] += 1
            dRt, dpt = relative(Rrec[t], prec[t], Rrec[t + k], prec[t + k])
            mom = (np.nan, np.nan)
            if t >= 1 and valid[t - 1]:
                mR, mp = repeat(lab["dR"][t - 1], lab["delta_pos_body"][t - 1].astype(np.float64), k)
                mom = (1000 * np.linalg.norm(mp - dpt), rotation_error_deg(mR, dRt))
            to_cam = Rcam @ Rrec[t]                           # body axes -> camera axes

            def row(dR, dp):
                e = to_cam @ (dp - dpt)
                return (k, 1000 * np.linalg.norm(dp - dpt), rotation_error_deg(dR, dRt),
                        1000 * abs(e[2]), 1000 * np.linalg.norm(e[:2]), mom[0], mom[1], ci, t)
            rows["zero"].append(row(np.eye(3), np.zeros(3)))
            if not np.isnan(mom[0]):
                rows["momentum"].append((k, mom[0], mom[1], np.nan, np.nan, mom[0], mom[1], ci, t))
            for key, ps in poses.items():
                a, b = ps.get(t), ps.get(t + k)
                if a is not None and b is not None:
                    rows[key].append(row(*relative(a[0], a[1], b[0], b[1])))
    out = {key: np.array(r, dtype=np.float64).reshape(-1, len(COLS)) for key, r in rows.items()}
    return ci, (out, eligible, info), None


def summarise(all_rows, eligible):
    res = {}
    for key, A in all_rows.items():
        res[key] = {}
        for k in HORIZONS:
            a = A[A[:, 0] == k]
            if not len(a):
                res[key][k] = None
                continue
            has_mom = ~np.isnan(a[:, 5])
            res[key][k] = {
                "n": int(len(a)), "coverage": len(a) / max(eligible[k], 1),
                "trans_mean_mm": float(a[:, 1].mean()), "trans_median_mm": float(np.median(a[:, 1])),
                "trans_p90_mm": float(np.percentile(a[:, 1], 90)),
                "rot_mean_deg": float(a[:, 2].mean()), "rot_median_deg": float(np.median(a[:, 2])),
                # momentum rows have no camera-axis split (NaN): report None, not NaN
                "depth_mean_mm": float(np.nanmean(a[:, 3])) if np.isfinite(a[:, 3]).any() else None,
                "side_mean_mm": float(np.nanmean(a[:, 4])) if np.isfinite(a[:, 4]).any() else None,
                # momentum on exactly the same pairs, for a paired comparison
                "paired_momentum_trans_mean_mm": float(a[has_mom, 5].mean()) if has_mom.any() else None,
                "paired_trans_mean_mm": float(a[has_mom, 1].mean()) if has_mom.any() else None,
                "beats_momentum_fraction": float((a[has_mom, 1] < a[has_mom, 5]).mean()) if has_mom.any() else None,
            }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=DATA / "validation_plan_v1.json")
    ap.add_argument("--limit", type=int, default=0, help="first N windows only (0 = all)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    recs = json.loads(args.plan.read_text())["records"]
    recs = recs[:args.limit] if args.limit else recs
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_rows, eligible, infos, failed = {}, {k: 0 for k in HORIZONS}, [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for i, (ci, res, err) in enumerate(pool.imap_unordered(run_clip, [(ci, r, args.seed) for ci, r in enumerate(recs)])):
            if err:
                failed.append({"clip": ci, "video_id": recs[ci]["video_id"], "error": err})
                continue
            rows, elig, info = res
            for key, A in rows.items():
                all_rows.setdefault(key, []).append(A)
            for k in HORIZONS:
                eligible[k] += elig[k]
            infos.append({"clip": ci, **info})
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(recs)} windows  {time.time() - t0:.0f}s", flush=True)
    all_rows = {key: np.concatenate(v) for key, v in all_rows.items()}
    summary = {"windows": len(recs), "failed": failed, "eligible_pairs": eligible,
               "columns": COLS, "results": summarise(all_rows, eligible),
               "camera_check_max_reprojection_px": max((x["reprojection_max_px"] or 0) for x in infos),
               "fk_vs_recorded_median_mm": float(np.median([x["fk_vs_recorded_mm"] for x in infos])),
               "fk_vs_recorded_median_deg": float(np.median([x["fk_vs_recorded_deg"] for x in infos])),
               "collapsed_frames": int(sum(x["collapsed_frames"] for x in infos)),
               "frames_examined": int(sum(x["frames"] for x in infos)),
               "track_err_px_median_of_clip_medians": float(np.median([x["track_err_px_median"] for x in infos if x["track_err_px_median"] is not None])),
               "seconds": time.time() - t0}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(args.out / "pairs.npz", **{k.replace("/", "_"): v for k, v in all_rows.items()})
    json.dump(infos, open(args.out / "per_clip.json", "w"))
    print(f"done: {len(recs) - len(failed)} windows, {len(failed)} failed, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
