#!/usr/bin/env python3
"""FK gripper trajectories alone -- ground truth, nothing learned, nothing predicted.

Deliberately separate from 34_gt_trajectories.py, which draws FK and Track-On together.
Overlaying them invites the reading that one is a prediction of the other; it is not.
Nothing here involves a model:

    joint encoders -> URDF forward kinematics -> 3D points on the gripper
                   -> camera calibration      -> pixels

FK reproduces PointWorld's `gripper_pose` to 0.0000 m and the projection puts 100% of
visible points in frame, so these paths are where those spots on the gripper actually
were. They are smooth because nothing is being estimated -- unlike a tracker, which
re-guesses each point from pixels every frame.

    python analysis/35_viz_fk_only.py --per-lab 1 --video
Run with any env with numpy + cv2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def fk_series(merged_npz: Path):
    m = np.load(merged_npz)
    return m["fk_uv"], m["fk_ok"], m["point_index"]


def draw_paths(img, uv, ok, t, tail, cmap_lo=(70, 120, 255), cmap_hi=(90, 230, 255)):
    """Trailing FK path per point, fading from old (dim) to new (bright)."""
    lo = max(0, t - tail)
    for j in range(uv.shape[1]):
        p, o = uv[lo:t + 1, j], ok[lo:t + 1, j]
        if o.sum() < 2:
            continue
        pts = p[o]
        for i in range(1, len(pts)):
            f = i / max(len(pts) - 1, 1)
            c = tuple(int(a + (b - a) * f) for a, b in zip(cmap_lo, cmap_hi))
            cv2.line(img, tuple(pts[i - 1].astype(int)), tuple(pts[i].astype(int)),
                     c, 1, cv2.LINE_AA)
        if o[-1]:
            cv2.circle(img, tuple(p[-1].astype(int)), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--per-lab", type=int, default=1)
    ap.add_argument("--pattern", default="*/merged/query_points/*.npz",
                    help="glob under --gen for npz carrying fk_uv/fk_ok/point_index. Use "
                         "'*/robot_droid/query_points/*.npz' to draw the DROID-sourced FK "
                         "directly, without waiting on the merge step.")
    ap.add_argument("--tail", type=int, default=45, help="frames of trailing path")
    ap.add_argument("--video", action="store_true", help="also write an mp4 per clip")
    ap.add_argument("--out", type=Path, default=OUT / "fk_only")
    args = ap.parse_args()

    picked, by_lab = [], {}
    for mg in sorted(args.gen.glob(args.pattern)):
        vid = mg.stem.rsplit("_f", 1)[0]
        lab = vid.split("+")[0]
        if len(by_lab.setdefault(lab, [])) < args.per_lab:
            by_lab[lab].append(mg); picked.append((vid, mg))

    args.out.mkdir(parents=True, exist_ok=True)
    for vid, mg in picked:
        shard = mg.parents[2]
        mp4 = shard / "obj" / "release_videos" / f"{vid}.mp4"
        if not mp4.exists():
            continue
        uv, ok, pidx = fk_series(mg)
        uv, ok = uv[:, pidx], ok[:, pidx]

        cap = cv2.VideoCapture(str(mp4))
        frames = []
        while True:
            r, f = cap.read()
            if not r:
                break
            frames.append(f)
        cap.release()
        T = min(len(frames), len(uv))
        if T < 5:
            continue

        t = int(T * 0.75)
        img = draw_paths(frames[t].copy(), uv, ok, t, args.tail)
        H, W = img.shape[:2]
        cv2.rectangle(img, (0, H - 24), (W, H), (0, 0, 0), -1)
        cv2.putText(img, f"FK ground truth only  --  {int(ok[t].sum())}/{uv.shape[1]} "
                         f"gripper points in frame,  last {args.tail} frames",
                    (8, H - 7), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 230, 255), 1)
        cv2.imwrite(str(args.out / f"{vid}.png"), img)

        if args.video:
            vw = cv2.VideoWriter(str(args.out / f"{vid}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
            for tt in range(T):
                im = draw_paths(frames[tt].copy(), uv, ok, tt, args.tail)
                cv2.rectangle(im, (0, H - 24), (W, H), (0, 0, 0), -1)
                cv2.putText(im, f"FK ground truth  t={tt}/{T-1}  "
                                f"{int(ok[tt].sum())}/{uv.shape[1]} in frame",
                            (8, H - 7), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 230, 255), 1)
                vw.write(im)
            vw.release()
        print(f"  {vid.split('+')[0]:9s} {vid[:44]:44s} T={T:4d} pts={uv.shape[1]}")

    print(f"\n[fk] -> {args.out}")


if __name__ == "__main__":
    main()
