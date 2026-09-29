#!/usr/bin/env python3
"""Overlay what ACTUALLY moves with the gripper against what we picked.

The scorer reduces a clip to one number; this shows the same thing as a picture, which
is what makes a low score interpretable. Cyan = our query points. Green = dense points
provably attached to the gripper (the ground truth we trust). Red cross = MolmoPoint's
seed. If the green cluster sits away from the cyan, we grounded on the wrong object.

    python scripts/12_viz_attached.py --video-id TRI+... --points-dir out/a1_mw40
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import video_io
from droid_qp.geom import (attached_mask, carrying_span, find_window, project,
                           release_to_pw_step, rigid_residuals, window_bounds)

PW = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
          "/droid/flows-fs-optimized")
OUT = Path(__file__).resolve().parent.parent / "out"


def render(video_id: str, frame: int, points_dir: Path, out_png: Path,
           out_size=(854, 480)):
    episode, cam = video_id.rsplit("__", 1)
    npz = points_dir / "query_points" / f"{video_id}_f{frame}.npz"
    d = np.load(npz)
    qp = d["query_points"][:, 1:].astype(float)
    mask = d["mask"] if "mask" in d.files else None

    mp4 = points_dir / "release_videos" / f"{video_id}.mp4"
    if not mp4.exists():
        mp4 = OUT / "release_videos" / f"{video_id}.mp4"
    img = cv2.cvtColor(video_io.read_frame(mp4, frame), cv2.COLOR_RGB2BGR)

    step = release_to_pw_step(frame)
    with h5py.File(PW / f"{episode}_flows.h5", "r") as f:
        wins = sorted((k for k in f.keys() if ":" in k), key=lambda k: window_bounds(k)[0])
        w = find_window(wins, step)
        a, _ = window_bounds(w)
        local = int(np.clip(step - a, 0, 10))
        g = f[w][f"camera_{cam}_ext"]
        sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
        vis = np.asarray(g["scene_visibility"][:]); dvm = np.asarray(g["scene_depth_valid_mask"][:])
        E = np.asarray(g["extrinsic"][:], dtype=np.float64)
        K = np.asarray(g["intrinsic"][:], dtype=np.float64)
        pose = np.asarray(f[w]["gripper_pose"][:], dtype=np.float64)
        gopen = np.asarray(f[w]["gripper_open"][:]).ravel().astype(bool)

    span = carrying_span(gopen, ref=local)
    r, s, grip = rigid_residuals(sf, pose, ref=local, span=span)
    local = int(np.clip(local, 0, sf.shape[0] - 1))
    att = attached_mask(r, s, grip) & vis[local] & dvm[local]
    uv, z = project(sf[local], E, K, out_size=out_size)
    W, H = out_size
    show = att & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    if mask is not None and mask.shape == (H, W):
        tint = np.zeros_like(img); tint[:] = (0, 200, 255)
        img[mask.astype(bool)] = cv2.addWeighted(img, 0.6, tint, 0.4, 0)[mask.astype(bool)]
    for x, y in qp:
        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1, cv2.LINE_AA)   # cyan: ours
    for x, y in uv[show]:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1, cv2.LINE_AA)     # green: attached

    n = int(show.sum())
    inside = 0.0
    if mask is not None and n:
        xi = np.clip(uv[show][:, 0].round().astype(int), 0, W - 1)
        yi = np.clip(uv[show][:, 1].round().astype(int), 0, H - 1)
        inside = float(mask.astype(bool)[yi, xi].mean())
    cv2.rectangle(img, (0, H - 34), (W, H), (0, 0, 0), -1)
    cv2.putText(img, f"cyan=ours({len(qp)})  green=MOVES WITH GRIPPER({n})  "
                     f"recall={inside*100:.0f}%",
                (6, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    print(f"  wrote {out_png}   attached={n}  recall={inside*100:.1f}%")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-dir", type=Path, default=OUT / "a1_mw40")
    ap.add_argument("--score", type=Path, default=OUT / "score_a1_mw40_fixed.json")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--worst", type=int, default=0, help="render the N worst-scoring clips")
    ap.add_argument("--best", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT / "attached_viz")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.score))["rows"] if r.get("status") == "ok"]
    targets = []
    if args.video_id:
        targets = [r for r in rows if r["video_id"] == args.video_id]
    if args.worst:
        targets += sorted(rows, key=lambda r: r["attached_recall"])[: args.worst]
    if args.best:
        targets += sorted(rows, key=lambda r: -r["attached_recall"])[: args.best]

    for r in targets:
        try:
            render(r["video_id"], r["detection_frame"], args.points_dir,
                   args.out_dir / f"{r['video_id']}_f{r['detection_frame']}.png")
        except Exception as e:
            print(f"  [skip] {r['video_id']}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
