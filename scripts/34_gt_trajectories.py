#!/usr/bin/env python3
"""Ground truth as POINT TRAJECTORIES in the image, not as time-series graphs.

Fatma hoca's ask. The action plots show one aggregate curve per episode; what she wants to
see is where each point actually went. For the ROBOT half we have that exactly:

    fk_uv[t, i]  -- where FK point i projects at frame t, computed from the arm's own
                    joint encoders through the URDF and the fitted camera calibration.

That is not an estimate. FK reproduces PointWorld's `gripper_pose` to 0.0000 m and the
projection puts 100% of visible points inside the frame, so `fk_uv` is the true image
trajectory of a known physical spot on the gripper, every frame, for free.

Two things follow, and both matter:

  1. It is the GROUND TRUTH a policy can be scored against in pixel space. A predicted EE
     delta maps through FK to predicted point positions, so "how far off is the policy"
     becomes a distance in pixels rather than radians.
  2. It measures TRACK-ON's drift, which we have never actually quantified -- the tracker
     is frozen and its output is the scene memory, so how far it wanders from truth is a
     property of our input representation.

Object points have no exact ground truth: nothing observed where they "really" were.
MolmoMotion's released 2D tracks (AllTracker) are the only reference available, and note
it is only their 3D LIFT that is contaminated, not the 2D tracking.

    python scripts/34_gt_trajectories.py --per-lab 1
Run with any env with numpy + cv2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"

GT_BGR = (80, 200, 255)       # amber  -- FK ground truth
TRACKED_BGR = (120, 255, 120) # green  -- what frozen Track-On produced


def load_pair(tk_npz: Path, merged_npz: Path):
    tk, mg = np.load(tk_npz), np.load(merged_npz)
    role = tk["role"] if "role" in tk.files else mg["role"]
    tracks, vis = tk["tracks"], tk["visibility"]
    T = len(tracks)

    # Object-only clips carry no FK fields: 28_merge_queries.py writes them only when a
    # robot npz exists, and the SAM filter rejects every robot point on a small number of
    # clips. There is no ground truth to compare against, so skip rather than fail.
    if not {"fk_uv", "fk_ok", "point_index"}.issubset(set(mg.files)):
        return None
    fk_uv, fk_ok = mg["fk_uv"], mg["fk_ok"]
    pidx = mg["point_index"]                       # FK id of each kept robot row
    rob_cols = np.where(role == 1)[0]
    if len(rob_cols) != len(pidx):
        # merge wrote robot rows in point_index order; a mismatch means the two files are
        # from different runs and pairing them would compare unrelated points.
        return None
    S = min(T, len(fk_uv))
    gt = fk_uv[:S][:, pidx]                        # (S, n_rob, 2)  exact
    ok = fk_ok[:S][:, pidx]                        # (S, n_rob)
    tr = tracks[:S][:, rob_cols]                   # (S, n_rob, 2)  Track-On
    tv = vis[:S][:, rob_cols]
    return dict(gt=gt, ok=ok, tracked=tr, tvis=tv, S=S,
                obj_tracked=tracks[:S][:, role == 0], obj_vis=vis[:S][:, role == 0],
                birth=int(np.nanmin(tk["queries"][rob_cols, 0])) if len(rob_cols) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--per-lab", type=int, default=1)
    ap.add_argument("--trackon-sub", default="trackon",
                    help="shard subdir with the Track-On output")
    ap.add_argument("--merged-sub", default="merged",
                    help="shard subdir with the merged queries carrying fk_uv")
    ap.add_argument("--save-npz", action="store_true",
                    help="also write the ground-truth trajectories per clip")
    ap.add_argument("--out", type=Path, default=OUT / "gt_trajectories")
    args = ap.parse_args()

    picked, by_lab = [], {}
    for npz in sorted(args.gen.glob(f"*/{args.trackon_sub}/tracks/*.npz")):
        lab = npz.stem.split("+")[0]
        if len(by_lab.setdefault(lab, [])) < args.per_lab:
            by_lab[lab].append(npz); picked.append(npz)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlay").mkdir(exist_ok=True)
    rows, curves = [], []
    n_skip = 0
    for tk in picked:
        shard = tk.parents[2]
        mg = shard / args.merged_sub / "query_points"
        cand = list(mg.glob(f"{tk.stem}_f*.npz"))
        if not cand:
            continue
        try:
            d = load_pair(tk, cand[0])
        except Exception as e:
            print(f"  {tk.stem[:44]} -- {type(e).__name__}: {e}"); n_skip += 1; continue
        if d is None:
            n_skip += 1; continue

        m = d["ok"] & d["tvis"]                    # comparable only where both exist
        if m.sum() < 50:
            continue
        err = np.linalg.norm(d["tracked"] - d["gt"], axis=2)      # (S, n) px
        e = err[m]
        # drift as a function of frames since the robot points were born
        since = np.arange(d["S"])[:, None] - d["birth"]
        sb = np.broadcast_to(since, err.shape)[m]
        rows.append(dict(video_id=tk.stem, n_robot=int(d["gt"].shape[1]),
                         median_px=float(np.median(e)), p90_px=float(np.percentile(e, 90)),
                         frac_within_10px=float((e <= 10).mean())))
        curves.append((sb, e))
        print(f"  {tk.stem.split('+')[0]:9s} n={d['gt'].shape[1]:3d}  "
              f"median {np.median(e):6.2f} px   p90 {np.percentile(e,90):6.2f} px   "
              f"within 10px {np.mean(e<=10)*100:5.1f}%")

        if args.save_npz:
            np.savez(args.out / f"{tk.stem}.npz", robot_gt=d["gt"].astype(np.float32),
                     robot_gt_ok=d["ok"], robot_tracked=d["tracked"].astype(np.float32),
                     robot_tracked_vis=d["tvis"],
                     object_tracked=d["obj_tracked"].astype(np.float32),
                     object_vis=d["obj_vis"])

        # overlay: the two trajectories drawn as PATHS on a late frame
        mp4 = shard / "obj" / "release_videos" / f"{tk.stem}.mp4"
        if mp4.exists():
            cap = cv2.VideoCapture(str(mp4))
            t_show = int(d["S"] * 0.75)
            cap.set(cv2.CAP_PROP_POS_FRAMES, t_show)
            okf, img = cap.read(); cap.release()
            if okf:
                lo = max(d["birth"], t_show - 40)
                for j in range(d["gt"].shape[1]):
                    for arr, col in ((d["gt"], GT_BGR), (d["tracked"], TRACKED_BGR)):
                        p = arr[lo:t_show + 1, j]
                        good = np.isfinite(p).all(axis=1)
                        p = p[good]
                        if len(p) > 1:
                            cv2.polylines(img, [p.astype(np.int32)], False, col, 1,
                                          cv2.LINE_AA)
                H, W = img.shape[:2]
                cv2.rectangle(img, (0, H - 24), (W, H), (0, 0, 0), -1)
                cv2.putText(img, "FK ground truth", (8, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, GT_BGR, 1)
                cv2.putText(img, "Track-On", (210, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, TRACKED_BGR, 1)
                cv2.putText(img, f"median err {np.median(e):.1f} px", (330, H - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.imwrite(str(args.out / "overlay" / f"{tk.stem}.png"), img)

    if n_skip:
        print(f"\n  {n_skip} clip(s) skipped (object-only, or no comparable frames)")
    json.dump(rows, open(args.out / "report.json", "w"), indent=1)
    if rows:
        md = np.array([r["median_px"] for r in rows])
        fw = np.array([r["frac_within_10px"] for r in rows])
        print(f"\n  Track-On vs FK ground truth over {len(rows)} clips:")
        print(f"    median error : {np.median(md):.2f} px   (worst clip {md.max():.2f})")
        print(f"    within 10 px : {np.median(fw)*100:.1f}% of point-frames")
        sb = np.concatenate([c[0] for c in curves]); e = np.concatenate([c[1] for c in curves])
        print(f"\n  drift vs frames since birth:")
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 10**6)]:
            m = (sb >= lo) & (sb < hi)
            if m.sum() > 20:
                lbl = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
                print(f"    {lbl:>8s} frames : median {np.median(e[m]):6.2f} px  "
                      f"(n={m.sum():,})")
    print(f"\n[gt] -> {args.out}")


if __name__ == "__main__":
    main()
