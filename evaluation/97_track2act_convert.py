#!/usr/bin/env python3
"""Track2Act for our setting: predicted point tracks -> a geometric pose plan -> actions.

Track2Act predicts 2D point tracks, infers a rigid transform from them, and executes the resulting
pose plan open-loop (a residual policy corrects it afterwards; not included here). This script runs
stages 1 and 2 on our data, using a point predictor's saved predictions:

    predicted position of gripper point p at t+h  =  FK position at t  +  predicted displacement
    pose at t, pose at t+h                        =  PnP of the palm's known 3D shape
    action over h steps                           =  the body-frame motion between those poses

Only the 68 palm points are used: the finger points move when the gripper opens, so they are not
rigid with the wrist. Every condition below is scored on exactly the same (frame, horizon) pairs.

    predicted   the point predictor's tracks, converted
    oracle      the TRUE future FK positions, converted -- the conversion's own error, expect ~0
    anchor_trk  as `predicted`, but the pose at t comes from Track-On's tracked pixels instead of
                FK: vision-only, no joint encoders
    zero        the gripper does not move
    repeat_last the previous recorded action repeated h times

    $PY_POINTWORLD -B evaluation/97_track2act_convert.py --run out/policy_point_D_1721709 \
        --out $PTP_OUT/track2act_D
"""
import os

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
CONDITIONS = ("predicted", "oracle", "anchor_trk", "zero", "repeat_last")
WIN = 48
_W = {}


def _init(model, palm):
    _W.update(model=model, palm=palm, shards=json.loads((DATA / "shard_index.json").read_text()))


def run_window(args):
    """-> (rows, error) with rows (h, condition index, translation mm, rotation deg)."""
    pred_file, rec = args
    try:
        import h5py
        from droid_qp.actions import sixd_to_rot
        from droid_qp.pose_fit import pnp, relative, repeat, rotation_error_deg, scaled_intrinsics
        from droid_qp.robot_points import PW_FLOWS, window_calibration
        vid = rec["video_id"]
        base = DATA / "gen" / _W["shards"][vid]
        with np.load(pred_file) as p:
            start = int(p["start"])
            disp = p["future_disp_px"].astype(np.float64)        # (WIN, n_rob, H, 2), pixels
        (mg,) = sorted((base / "merged/query_points").glob(f"{vid}_f*.npz"))
        with np.load(mg) as a:
            ids = a["point_index"]
            fk, ok = a["fk_uv"][:, ids].astype(np.float64), a["fk_ok"][:, ids]
        with np.load(base / "trackon/tracks" / f"{vid}.npz") as t:
            cols = np.flatnonzero(t["role"] == 1)
            trk, vis = t["tracks"][:, cols].astype(np.float64), t["visibility"][:, cols]
        with np.load(base / "actions/labels" / f"{vid}.npz") as a:
            pos = a["position"].astype(np.float64)
            R = sixd_to_rot(a["rot_6d"].astype(np.float64))
            dp_lab, dR_lab = a["delta_pos_body"].astype(np.float64), sixd_to_rot(a["delta_rot6d_body"].astype(np.float64))
            valid = a["valid"]
        ep, cam = vid.rsplit("__", 1)
        with h5py.File(PW_FLOWS / f"{ep}_flows.h5", "r") as f:
            w0 = sorted((k for k in f.keys() if ":" in k), key=lambda k: int(k.split(":")[0]))[0]
            _E, Kc = window_calibration(f, w0, cam)
        K = scaled_intrinsics(Kc)
        model, palm_all = _W["model"], _W["palm"][ids]
        n = min(len(fk), len(trk), len(pos))
        rows = []
        for k in range(min(WIN, n - start)):
            t0 = start + k
            for h in HORIZONS:
                t1 = t0 + h
                if t1 >= n or not valid[t0:t1].all():
                    continue
                m = palm_all & ok[t0] & ok[t1]           # the same points for every condition
                if m.sum() < 6:
                    continue
                dRt, dpt = relative(R[t0], pos[t0], R[t1], pos[t1])     # recorded motion over h steps
                a0 = pnp(model[ids[m]], fk[t0][m], K)                   # anchor: where we are now
                if a0 is None:
                    continue
                ends = {"predicted": fk[t0][m] + disp[k, m, h - 1],
                        "oracle": fk[t1][m]}
                out = {}
                for name, uv in ends.items():
                    b = pnp(model[ids[m]], uv, K)
                    out[name] = relative(*a0, *b) if b is not None else None
                mt = m & vis[t0]                                        # vision-only anchor
                if mt.sum() >= 6:
                    at = pnp(model[ids[mt]], trk[t0][mt], K)
                    b = pnp(model[ids[m]], ends["predicted"], K)
                    out["anchor_trk"] = relative(*at, *b) if (at is not None and b is not None) else None
                out["zero"] = (np.eye(3), np.zeros(3))
                if t0 >= 1 and valid[t0 - 1]:
                    out["repeat_last"] = repeat(dR_lab[t0 - 1], dp_lab[t0 - 1], h)
                for ci, name in enumerate(CONDITIONS):
                    v = out.get(name)
                    if v is not None:
                        rows.append((h, ci, 1000 * np.linalg.norm(v[1] - dpt), rotation_error_deg(v[0], dRt)))
        return np.array(rows, dtype=np.float64).reshape(-1, 4), None
    except Exception as e:
        import traceback
        return np.zeros((0, 4)), f"{rec['video_id']}: {type(e).__name__}: {e} | {traceback.format_exc(limit=2)[-200:]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="run directory holding evaluation/predictions")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=DATA / "validation_plan_v1.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    from droid_qp.pose_fit import gripper_model
    from droid_qp.robot_points import load_sampler
    model, palm = gripper_model(load_sampler(128, gripper_only=True, seed=0))
    recs = json.loads(args.plan.read_text())["records"]
    recs = recs[:args.limit] if args.limit else recs
    jobs = []
    for r in recs:
        f = args.run / "evaluation/predictions" / f"{r['video_id']}.npz"
        if f.exists():
            jobs.append((f, r))
    if not jobs:
        raise SystemExit(f"no saved predictions under {args.run}/evaluation/predictions")
    t0 = time.time()
    chunks, failed = [], []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(model, palm)) as pool:
        for i, (rows, err) in enumerate(pool.imap_unordered(run_window, jobs, chunksize=4)):
            if err:
                failed.append(err)
            if len(rows):
                chunks.append(rows)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} windows  {time.time() - t0:.0f}s", flush=True)
    A = np.concatenate(chunks)
    res = {"run": str(args.run), "windows": len(jobs), "failed": failed[:10], "n_failed": len(failed),
           "conditions": CONDITIONS, "horizons": {}}
    for h in HORIZONS:
        row = {}
        for ci, name in enumerate(CONDITIONS):
            a = A[(A[:, 0] == h) & (A[:, 1] == ci)]
            if len(a):
                row[name] = {"n": int(len(a)), "trans_mean_mm": float(a[:, 2].mean()),
                             "trans_median_mm": float(np.median(a[:, 2])),
                             "rot_mean_deg": float(a[:, 3].mean())}
        res["horizons"][str(h)] = row
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez_compressed(args.out / "pairs.npz", rows=A, conditions=np.array(CONDITIONS))
    print(f"\n{'horizon':>8s} " + " ".join(f"{c:>14s}" for c in CONDITIONS))
    for h in HORIZONS:
        r = res["horizons"][str(h)]
        print(f"{h:8d} " + " ".join(f"{r[c]['trans_mean_mm']:11.1f} mm" if c in r else f"{'-':>14s}" for c in CONDITIONS))
    print(f"\n{len(failed)} windows failed; {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
