#!/usr/bin/env python3
"""A LEGIBLE ground-truth figure. 35_viz_fk_only.py draws all 128 gripper points over the
whole episode, which is correct and unreadable.

The problem is redundancy, not volume. All 128 points sit on one rigid body, so their
paths are near-parallel copies -- 128 curves carrying about as much information as the
gripper's 6-DoF pose. Drawing every one of them over every frame produces a swept blob.

Three changes:
  * draw the gripper CENTROID as the main path, plus a few spread points for extent
  * colour by TIME, so direction of travel is visible rather than implied
  * mark where the gripper OPENS and CLOSES, which is what makes a manipulation
    trajectory readable -- approach, grasp, transport, release

    python scripts/42_viz_gt_readable.py --per-lab 1
Run with any env with numpy + cv2 (+ matplotlib for the colour ramp).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"


def ramp(f: float):
    """viridis-ish: dark blue (early) -> green -> yellow (late). BGR."""
    stops = [(84, 1, 68), (140, 90, 33), (97, 173, 53), (37, 231, 253)]
    x = f * (len(stops) - 1)
    i = min(int(x), len(stops) - 2)
    a, b = stops[i], stops[i + 1]
    w = x - i
    return tuple(int(p + (q - p) * w) for p, q in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--per-lab", type=int, default=1)
    ap.add_argument("--n-extra", type=int, default=6,
                    help="spread gripper points drawn faintly alongside the centroid, to "
                         "show extent. 0 for centroid only.")
    ap.add_argument("--out", type=Path, default=OUT / "gt_readable")
    args = ap.parse_args()

    picked, by_lab = [], {}
    for mg in sorted(args.gen.glob("*/merged/query_points/*.npz")):
        vid = mg.stem.rsplit("_f", 1)[0]
        lab = vid.split("+")[0]
        if len(by_lab.setdefault(lab, [])) < args.per_lab:
            by_lab[lab].append(mg); picked.append((vid, mg))

    args.out.mkdir(parents=True, exist_ok=True)
    for vid, mg in picked:
        shard = mg.parents[2]
        mp4 = shard / "obj" / "release_videos" / f"{vid}.mp4"
        act = shard / "actions" / "labels" / f"{vid}.npz"
        if not mp4.exists():
            continue
        d = np.load(mg)
        uv, ok = d["fk_uv"][:, d["point_index"]], d["fk_ok"][:, d["point_index"]]

        cap = cv2.VideoCapture(str(mp4)); frames = []
        while True:
            r, f = cap.read()
            if not r:
                break
            frames.append(f)
        cap.release()
        T = min(len(frames), len(uv))
        if T < 10:
            continue

        # centroid of the visible gripper points at each frame
        cen = np.full((T, 2), np.nan)
        for t in range(T):
            if ok[t].sum() >= 3:
                cen[t] = uv[t][ok[t]].mean(axis=0)
        good = np.isfinite(cen).all(axis=1)
        if good.sum() < 10:
            continue

        img = frames[int(T * 0.9)].copy()
        H, W = img.shape[:2]

        # faint spread points first, so the centroid reads on top
        if args.n_extra:
            step = max(1, uv.shape[1] // args.n_extra)
            for j in range(0, uv.shape[1], step)[: args.n_extra]:
                p = uv[:T, j]; m = ok[:T, j] & np.isfinite(p).all(axis=1)
                pts = p[m].astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(img, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)

        for t in range(1, T):
            if not (good[t] and good[t - 1]):
                continue
            cv2.line(img, tuple(cen[t - 1].astype(int)), tuple(cen[t].astype(int)),
                     ramp(t / max(T - 1, 1)), 2, cv2.LINE_AA)

        # grasp events -- the structure of a manipulation trajectory
        n_ev = 0
        if act.exists():
            a = np.load(act)
            g = a["gripper_open_next"][:T, 0]
            v = a["valid"][:T]
            for t in np.where((np.diff(g) != 0) & v[:-1])[0]:
                if good[t]:
                    closing = g[t + 1] < 0.5
                    c = (60, 60, 255) if closing else (120, 255, 120)
                    cv2.drawMarker(img, tuple(cen[t].astype(int)), c,
                                   cv2.MARKER_TILTED_CROSS, 14, 2)
                    n_ev += 1

        cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
        cv2.putText(img, f"gripper path, {T} frames", (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (37, 231, 253), 1)
        cv2.putText(img, "dark=start  bright=end", (250, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (200, 200, 200), 1)
        cv2.putText(img, f"x close", (470, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 60, 255), 1)
        cv2.putText(img, f"x open  ({n_ev} events)", (560, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (120, 255, 120), 1)
        cv2.imwrite(str(args.out / f"{vid}.png"), img)
        print(f"  {vid.split('+')[0]:9s} {vid[:44]:44s} T={T:4d} grasp events={n_ev}")
    print(f"\n[gtr] -> {args.out}")


if __name__ == "__main__":
    main()
