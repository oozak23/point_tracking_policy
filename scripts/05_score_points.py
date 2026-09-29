#!/usr/bin/env python3
"""Phase 1 -- score a query-point set objectively: do the points move with the gripper?

Replaces both discarded metrics. Eyeball counts do not scale; agreement with
MolmoMotion's released tracks is invalid as a quality score, since those tracks are
demonstrably on the wrong object in real clips (see out/diag_weird.png).

The one thing objectively true of a manipulated object is that it moves with the
gripper. Nothing else in the scene does -- not the table, not the shelves, not a marker
lying on a tray. So every failure we have observed scores near zero automatically.

Works on ANY 2D query-point set, so approach 1 and approach 2 are judged identically.

    our 100 points (854x480, release frame f)
      -> pw_step = f*2, find the PointWorld window containing it
      -> project that window's ~26,600 dense 3D points to 2D, scaled to 854x480
      -> nearest projected dense point for each of ours
      -> read its residual r and movement s  ->  fraction attached to the gripper

Limits, stated up front: only the 14,083-clip overlap; only where the detection frame
lands in a PointWorld window; and a correct object with too few depth points scores
low. A low score means LOOK, not reject.

CONFOUND worth remembering when reading results: the score is measured in the window
containing the *detection frame*. If the grasp happens later, a correctly identified
object is not yet attached there and scores low. `grip_disp` and `n_attached_window`
are reported per clip so this is visible rather than hidden.

    python scripts/05_score_points.py --points-dir out/mw/query_points
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.geom import (attached_mask, carrying_span, find_window, project,
                           release_to_pw_step, rigid_residuals, window_bounds)

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
OUT = Path(__file__).resolve().parent.parent / "out"

NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def score_one(video_id: str, frame: int, pts_xy: np.ndarray, mask: np.ndarray | None = None,
              out_size=(854, 480), r_max=0.03, s_frac=0.3, min_attached=20) -> dict:
    """Primary metric is `attached_recall`: of the dense points PROVABLY attached to the
    gripper, what fraction lands inside our mask?

    The first version of this asked, per query point, "is your nearest dense point
    attached?" That under-reports badly. Measured over 40 episodes restricted to
    carrying windows, only ~52-64 of ~26,600 dense points are attached (~2% of the
    scene), so a query point correctly placed on the object still usually finds a
    STATIC nearest neighbour. Inverting the direction removes that dilution: locate the
    attached cluster first, then ask whether our mask covers it.

    r_max=0.03 comes from out/threshold_calibration_carrying.json -- best yield
    (median 64 attached points, 64% of windows with >=20). The residual histogram has
    no clean valley, so this is a chosen operating point, not a discovered one; the
    static population peaks at 0.04-0.10 m, i.e. at the gripper displacement, as the
    rigid-motion model predicts.
    """
    episode, cam = video_id.rsplit("__", 1)
    row = {"video_id": video_id, "episode": episode, "cam": cam,
           "detection_frame": int(frame), "n_points": int(len(pts_xy))}

    path = PW_FLOWS / f"{episode}_flows.h5"
    if not path.exists():
        row["status"] = "no_pointworld_episode"
        return row

    pw_step = release_to_pw_step(frame)
    row["pw_step"] = pw_step
    with h5py.File(path, "r") as f:
        windows = sorted((k for k in f.keys() if ":" in k),
                         key=lambda k: window_bounds(k)[0])
        w = find_window(windows, pw_step)
        if w is None:
            row["status"] = "frame_outside_pointworld_windows"
            return row
        row["window"] = w

        cam_key = f"camera_{cam}_ext"
        if cam_key not in f[w]:
            row["status"] = "camera_not_in_window"
            return row
        g = f[w][cam_key]

        sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
        vis = np.asarray(g["scene_visibility"][:])
        dvm = np.asarray(g["scene_depth_valid_mask"][:])
        E = np.asarray(g["extrinsic"][:], dtype=np.float64)
        K = np.asarray(g["intrinsic"][:], dtype=np.float64)
        pose = np.asarray(f[w]["gripper_pose"][:], dtype=np.float64)
        gopen = np.asarray(f[w]["gripper_open"][:]).ravel().astype(bool)

    a, _ = window_bounds(w)
    local = int(np.clip(pw_step - a, 0, sf.shape[0] - 1))
    row["local_index"] = local

    # Measure only over the carrying span. A window straddling the grasp spends its
    # first half with the object unattached; since r is a median over frames, that half
    # dominates and a correctly-held object reads as static. This is what produced the
    # spurious "geometric coverage is 43%".
    span = carrying_span(gopen, ref=local)
    if span is None:
        row["status"] = "gripper_never_closed_in_window"
        return row
    row["carrying_span"] = list(span)
    row["carrying_len"] = int(span[1] - span[0] + 1)
    if row["carrying_len"] < 3:
        row["status"] = "carrying_span_too_short"
        return row

    r, s, grip_disp = rigid_residuals(sf, pose, ref=local, span=span)
    row["grip_disp_m"] = grip_disp
    local = int(np.clip(local, span[0], span[1]))

    att = attached_mask(r, s, grip_disp, r_max=r_max, s_frac=s_frac)
    good = vis[local] & dvm[local]
    row["n_dense"] = int(sf.shape[1])
    row["n_dense_valid"] = int(good.sum())
    row["n_attached_window"] = int((att & good).sum())

    if grip_disp < 0.03:
        # With a still gripper every point has a small residual; r carries no signal.
        row["status"] = "gripper_static_undefined"
        return row
    if good.sum() < 100:
        row["status"] = "too_few_valid_dense_points"
        return row

    uv, z = project(sf[local], E, K, out_size=out_size)
    ok = good & (z > 0)
    if ok.sum() < 100:
        row["status"] = "too_few_projected_points"
        return row

    W, H = out_size
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    att_ok = att & ok & in_frame
    row["n_attached_visible"] = int(att_ok.sum())
    if att_ok.sum() < min_attached:
        # Nothing provably attached -- the gripper may have closed on nothing, or the
        # object is too thin/reflective to reconstruct. Cannot adjudicate either way.
        row["status"] = "too_few_attached_points"
        return row

    uv_att = uv[att_ok]
    xi = np.clip(uv_att[:, 0].round().astype(int), 0, W - 1)
    yi = np.clip(uv_att[:, 1].round().astype(int), 0, H - 1)

    # PRIMARY: how much of the attached cluster does our mask cover?
    if mask is not None and mask.shape == (H, W):
        recall = float(mask[yi, xi].mean())
    else:
        # No mask stored: fall back to "within 12 px of one of our query points",
        # which approximates the mask our k-means points were drawn from.
        d2p = ((uv_att[:, None, :] - pts_xy[None, :, :]) ** 2).sum(-1)
        recall = float((np.sqrt(d2p.min(axis=1)) < 12).mean())
    row["attached_recall"] = recall
    row["score"] = recall

    # SECONDARY: how far is our point cloud from the attached cluster?
    c_att = uv_att.mean(axis=0)
    c_ours = pts_xy.mean(axis=0)
    row["centroid_dist_px"] = float(np.linalg.norm(c_att - c_ours))

    # Kept for comparison with the first (diluted) formulation.
    uvv, attv, rv = uv[ok], att[ok], r[ok]
    d2 = ((pts_xy[:, None, :] - uvv[None, :, :]) ** 2).sum(-1)
    j = np.argmin(d2, axis=1)
    row.update({
        "status": "ok",
        "nn_attached_frac": float(attv[j].mean()),
        "median_residual_m": float(np.median(rv[j])),
        "median_nn_dist_px": float(np.median(np.sqrt(d2[np.arange(len(j)), j]))),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-dir", type=Path, required=True)
    ap.add_argument("--label", default=None, help="name for this point set in the report")
    ap.add_argument("--out-size", type=int, nargs=2, default=[854, 480])
    ap.add_argument("--r-max", type=float, default=0.02)
    ap.add_argument("--s-frac", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    label = args.label or args.points_dir.parent.name
    files = sorted(args.points_dir.glob("*.npz"))
    print(f"[score] {label}: {len(files)} query-point files in {args.points_dir}")
    if not files:
        sys.exit("nothing to score")

    rows = []
    for i, p in enumerate(files, 1):
        m = NAME_RE.match(p.name)
        if not m:
            print(f"  [skip] unparseable name: {p.name}")
            continue
        d = np.load(p, allow_pickle=False)
        qp = d["query_points"]
        msk = d["mask"] if "mask" in d.files else None
        row = score_one(m["vid"], int(m["frame"]), qp[:, 1:].astype(np.float64), mask=msk,
                        out_size=tuple(args.out_size),
                        r_max=args.r_max, s_frac=args.s_frac)
        rows.append(row)
        st = row.get("status")
        if st == "ok":
            print(f"  [{i:2d}] {row['video_id'][:44]:44s} f{row['detection_frame']:<4d} "
                  f"recall={row['attached_recall']*100:5.1f}%  "
                  f"cdist={row['centroid_dist_px']:6.1f}px  "
                  f"n_att={row['n_attached_visible']:4d}  grip={row['grip_disp_m']:.3f}m")
        else:
            extra = ""
            if st == "too_few_attached_points":
                extra = f" (n_att={row.get('n_attached_visible', 0)})"
            print(f"  [{i:2d}] {row['video_id'][:44]:44s} f{row['detection_frame']:<4d} "
                  f"-- {st}{extra}")

    out = args.out or (OUT / f"score_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"label": label, "points_dir": str(args.points_dir),
               "r_max": args.r_max, "s_frac": args.s_frac, "rows": rows},
              open(out, "w"), indent=1)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n===== {label} =====")
    print(f"clips              : {len(rows)}")
    print(f"scorable           : {len(ok)}")
    for st in ("gripper_static_undefined", "too_few_attached_points",
               "frame_outside_pointworld_windows", "no_pointworld_episode",
               "camera_not_in_window", "too_few_valid_dense_points",
               "too_few_projected_points"):
        n = sum(r.get("status") == st for r in rows)
        if n:
            print(f"  {st:34s}: {n}")
    if ok:
        sc = np.array([r["attached_recall"] for r in ok])
        cd = np.array([r["centroid_dist_px"] for r in ok])
        na = np.array([r["n_attached_visible"] for r in ok])
        nnf = np.array([r["nn_attached_frac"] for r in ok])
        print(f"\nATTACHED RECALL -- of the points provably moving with the gripper,")
        print(f"what fraction lands inside our mask?")
        print(f"  mean   {sc.mean()*100:5.1f}%")
        print(f"  median {np.median(sc)*100:5.1f}%")
        print(f"  >70%   {int((sc > 0.7).sum())}/{len(sc)} clips")
        print(f"  >30%   {int((sc > 0.3).sum())}/{len(sc)} clips")
        print(f"  ~0%    {int((sc < 0.02).sum())}/{len(sc)} clips  <- mask misses the object")
        print(f"\ncentroid distance to attached cluster: median {np.median(cd):.0f} px "
              f"({int((cd < 50).sum())}/{len(cd)} within 50 px)")
        print(f"attached points available: median {np.median(na):.0f} per clip")
        print(f"\n(old diluted metric, for reference: nn_attached_frac mean "
              f"{nnf.mean()*100:.1f}%)")
    print(f"\n[score] wrote {out}")


if __name__ == "__main__":
    main()
