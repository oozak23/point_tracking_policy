#!/usr/bin/env python3
"""Re-render tracked-point videos with OBJECT and ROBOT points in different colours.

16_run_trackon.py's `draw()` colours by visibility alone, so all 217 points look alike --
which hides the one distinction the architecture is built on: object points become the
scene memory the action decoder reads, robot points become the action memory it writes.
A demo video that cannot show that difference is not showing the method.

Re-renders from the saved npz (tracks, visibility, role) plus the release video, so the
frozen tracker does NOT run again -- this is seconds per clip, not minutes.

    python analysis/32_viz_tracked_roles.py --per-lab 1
Run with any env that has cv2 + numpy.
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
OBJECT_BGR = (60, 220, 255)    # amber  -- scene memory
ROBOT_BGR = (120, 255, 120)    # green  -- action memory
DIM = 0.35                     # occluded points keep their hue, dimmed


def render(npz: Path, mp4: Path, dst: Path, tail: int = 8):
    d = np.load(npz)
    tracks, vis = d["tracks"], d["visibility"]
    role = d["role"] if "role" in d.files else np.zeros(tracks.shape[1], np.int8)

    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    T = min(len(frames), tracks.shape[0])
    if T == 0:
        return False
    H, W = frames[0].shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    cols = [OBJECT_BGR if r == 0 else ROBOT_BGR for r in role]
    for t in range(T):
        img = frames[t].copy()
        for k in range(tracks.shape[1]):
            base = cols[k]
            for dt in range(1, min(tail, t) + 1):
                a, b = tracks[t - dt, k], tracks[t - dt + 1, k]
                if np.isnan(a).any() or np.isnan(b).any():
                    continue
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                         tuple(int(c * 0.45) for c in base), 1, cv2.LINE_AA)
            p = tracks[t, k]
            if np.isnan(p).any():
                continue
            v = bool(vis[t, k])
            c = base if v else tuple(int(x * DIM) for x in base)
            cv2.circle(img, tuple(p.astype(int)), 2, c, -1 if v else 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
        no = int((role == 0).sum()); nr = int((role == 1).sum())
        cv2.putText(img, f"t={t}/{T-1}", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        # Label what these ARE -- points tracked by the frozen tracker -- not what they
        # will later feed. An earlier version said "(scene memory)" / "(action memory)",
        # which read as though a decoder existed and these were its output. Nothing here
        # is predicted: no policy exists yet, and Track-On emits point positions, never
        # actions.
        cv2.putText(img, f"object pts {no}", (110, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJECT_BGR, 1)
        cv2.putText(img, f"robot pts {nr}", (280, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROBOT_BGR, 1)
        cv2.putText(img, "frozen Track-On, tracked positions (not predictions)",
                    (410, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
        vw.write(img)
    vw.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--per-lab", type=int, default=1,
                    help="clips per lab. AUTOLab is only 14.8% of the data, so a "
                         "first-N selection would misrepresent it.")
    ap.add_argument("--out", type=Path, default=OUT / "demo_tracked")
    args = ap.parse_args()

    picked, by_lab = [], {}
    for npz in sorted(args.gen.glob("*/trackon/tracks/*.npz")):
        lab = npz.stem.split("+")[0]
        if len(by_lab.setdefault(lab, [])) < args.per_lab:
            by_lab[lab].append(npz)
            picked.append(npz)

    ok = 0
    for npz in picked:
        shard = npz.parents[2]
        mp4 = shard / "obj" / "release_videos" / f"{npz.stem}.mp4"
        if not mp4.exists():
            print(f"  no video for {npz.stem[:48]}")
            continue
        if render(npz, mp4, args.out / f"{npz.stem}.mp4"):
            print(f"  {npz.stem[:56]}")
            ok += 1
    print(f"\n[demo] {ok} clips across {len(by_lab)} labs -> {args.out}")


if __name__ == "__main__":
    main()
