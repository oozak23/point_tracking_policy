#!/usr/bin/env python3
"""Robot query points via FK, projected to pixels, with overlays to look at.

Two things at once:
  1. the robot half of the query set (the action representation), saved in the same npz
     format as the object points so they concatenate into ONE Track-On query tensor
  2. overlay frames, since the first question about any FK->pixel pipeline is whether
     the points actually land on the robot

Frame spaces: PointWorld indexes windows by its own step (trajectory/2 ~ 7.2 Hz); our
videos are release frames (trajectory/SUBSAMPLE). At SUBSAMPLE=2 the two spaces coincide
and the mapping is the identity; it was `pw_step // 2` while we generated at 3.6 Hz.
`droid_qp.rates` owns the factor -- never write it out by hand here.

    python scripts/19_robot_points.py --n 8 --num-points 128
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import rates, video_io
from droid_qp.index import load_manifest
from droid_qp.robot_points import (PW_FLOWS, droid_joint_series, episode_joint_series,
                                   load_sampler, robot_points_2d, robot_points_3d,
                                   window_calibration)

OUT = Path(__file__).resolve().parent.parent / "out"
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, default=OUT / "mm_sam_f0",
                   help="run dir whose clips to mirror (reuses its release_videos)")
    ap.add_argument("--n", type=int, default=8,
                    help="number of clips; 0 = ALL. At scale each shard holds 74-123 clips,\n"
                         "so a forgotten --n would silently process only the default and\n"
                         "still report success -- hence 0 means all, matching 24_action_labels.py.")
    ap.add_argument("--num-points", type=int, default=128)
    ap.add_argument("--gripper-only", action="store_true",
                    help="sample only the gripper links, not the whole arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-visible", type=int, default=5,
                    help="minimum FK points projecting into frame for a step to be "
                         "eligible as the birth frame. Matches the threshold "
                         "20_robot_points_sam_filter.py enforces downstream.")
    ap.add_argument("--source", choices=["droid", "pointworld"], default="droid",
                    help="where JOINT ANGLES come from (calibration always comes from "
                         "PointWorld). DEFAULT droid: identical numbers, but PointWorld's "
                         "windows cover ~75%% of an episode and the rest is zero-filled, "
                         "which FK reads as the arm standing upright rather than as "
                         "missing. Measured: 1.085%% of uncovered point-steps still pass "
                         "the visibility test and enter the ground truth as fake positions.")
    ap.add_argument("--out", type=Path, default=OUT / "robot_points")
    args = ap.parse_args()
    print(f"[robot] {rates.describe()}")

    vids = []
    for p in sorted((args.like / "query_points").glob("*.npz")):
        m = NAME_RE.match(p.name)
        if m:
            vids.append(m["vid"])
    vids = vids[: args.n or None]
    print(f"[robot] {len(vids)} clips | N={args.num_points} "
          f"| gripper_only={args.gripper_only}")

    by_ep = ({r.video_id.rsplit("__", 1)[0]: r for r in load_manifest()}
             if args.source == "droid" else {})

    sampler = load_sampler(args.num_points, gripper_only=args.gripper_only,
                           seed=args.seed)
    print(f"[robot] sampler ready: {len(sampler.joint_names)} movable joints")

    qdir, vdir = args.out / "query_points", args.out / "overlay"
    qdir.mkdir(parents=True, exist_ok=True)
    vdir.mkdir(parents=True, exist_ok=True)

    report = []
    for i, vid in enumerate(vids, 1):
        ep, cam = vid.rsplit("__", 1)
        try:
            with h5py.File(PW_FLOWS / f"{ep}_flows.h5", "r") as f:
                jp, gp, filled = episode_joint_series(f)
                if jp is None:
                    print(f"  [{i}] {vid[:44]} -- no windows")
                    continue
                w0 = sorted((k for k in f.keys() if ":" in k),
                            key=lambda k: int(k.split(":")[0]))[0]
                if f"camera_{cam}_ext" not in f[w0]:
                    print(f"  [{i}] {vid[:44]} -- camera not in window")
                    continue
                E, K = window_calibration(f, w0, cam)

            # Calibration is per-episode in practice (the first window's, reused), so it
            # can come from PointWorld while the joints come from DROID.
            if args.source == "droid":
                rec = by_ep.get(ep)
                if rec is None or not rec.trajectory_h5.exists():
                    print(f"  [{i}] {vid[:44]} -- no DROID trajectory.h5")
                    continue
                with h5py.File(rec.trajectory_h5, "r") as fh:
                    jp, gp, filled = droid_joint_series(fh, rates.SUBSAMPLE)

            pts3 = robot_points_3d(sampler, jp, gp)              # (S, N, 3)
            uv, ok = robot_points_2d(pts3, E, K)                 # (S, N, 2), (S, N)
            # A step with no joint data must never be reported visible, whatever the
            # projection says -- with the pointworld source those joints are zeros, i.e.
            # a real pose the arm was never in.
            ok &= filled[:, None]

            # PointWorld step -> release frame
            mp4 = args.like / "release_videos" / f"{vid}.mp4"
            if not mp4.exists():
                print(f"  [{i}] {vid[:44]} -- no video")
                continue
            frames = video_io.read_frames(mp4)
            T = len(frames)

            rel = rates.pw_step_to_release(np.arange(len(uv)))

            # Query set: born at the first covered frame where the gripper is ACTUALLY
            # VISIBLE. `ok` is the projection test (in front of the camera, inside the
            # image); `filled` only says a PointWorld window covers the step. Without the
            # `ok` term s0 can land on a step where the arm sits in its retracted home
            # pose, outside an exterior camera's view -- vis0 is then all-False, an empty
            # query set gets written, and the failure surfaces one stage later in
            # 20_robot_points_sam_filter.py as "no FK step at frame N". Measured on 1 of
            # 8 clips (AUTOLab+5d05c5aa+2023-11-17-22h-46m-01s__24400334: coverage from
            # step ~4, first visible point at step 8).
            #
            # MIN_VIS matches the threshold 20_...py already enforces, so the two stages
            # agree on what counts as usable instead of disagreeing.
            keep_step = (rel < T) & filled & (ok.sum(axis=1) >= args.min_visible)
            if keep_step.sum() == 0:
                n_cov = int(((rel < T) & filled).sum())
                print(f"  [{i}] {vid[:44]} -- no step with >= {args.min_visible} visible "
                      f"FK points ({n_cov} covered steps, max visible "
                      f"{int(ok.sum(axis=1).max())})")
                continue

            s0 = int(np.argmax(keep_step))
            f0 = int(rel[s0])
            vis0 = ok[s0]
            qp = np.stack([np.full(vis0.sum(), f0),
                           np.rint(uv[s0][vis0][:, 0]),
                           np.rint(uv[s0][vis0][:, 1])], axis=1).astype(np.int32)
            # keep_step already guarantees this; belt and braces, because an empty npz
            # written here is invisible until the next stage and reads as a bug there.
            if len(qp) == 0:
                print(f"  [{i}] {vid[:44]} -- empty query set at f{f0}, not writing")
                continue
            np.savez(qdir / f"{vid}_f{f0}.npz", query_points=qp,
                     dim=np.array([480, 854], dtype=np.int32),
                     fk_uv=uv.astype(np.float32), fk_ok=ok,
                     fk_release_frame=rel.astype(np.int32),
                     point_index=np.where(vis0)[0].astype(np.int32))

            # Overlay: FK ground truth at four moments across the episode.
            steps = [s for s in np.linspace(s0, len(uv) - 1, 4).astype(int)
                     if keep_step[s]]
            tiles = []
            for s in steps:
                t = int(rel[s])
                img = cv2.cvtColor(frames[min(t, T - 1)], cv2.COLOR_RGB2BGR).copy()
                for (x, y) in uv[s][ok[s]]:
                    cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(img, f"t={t}  {int(ok[s].sum())}/{uv.shape[1]} in frame",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tiles.append(cv2.resize(img, (560, 315)))
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(vdir / f"{vid}.png"),
                        np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))

            frac = float(ok[keep_step].mean())
            print(f"  [{i:2d}] {vid[:44]:44s} S={len(uv):3d} T={T:3d} "
                  f"born@f{f0:<3d} N={len(qp):3d}/{args.num_points} "
                  f"in-frame={frac*100:4.1f}%")
            report.append({"video_id": vid, "n_query": int(len(qp)),
                           "frac_in_frame": frac, "birth_frame": f0,
                           "pw_steps": int(len(uv)), "T": T})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{i}] {vid[:44]} -- {type(e).__name__}: {e}")

    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print(f"\n===== summary =====")
    print(f"clips           : {len(report)}")
    if report:
        fr = np.array([r["frac_in_frame"] for r in report])
        nq = np.array([r["n_query"] for r in report])
        print(f"points in frame : mean {fr.mean()*100:.1f}%  min {fr.min()*100:.1f}%")
        print(f"query points    : median {np.median(nq):.0f} of {args.num_points}")
    print(f"[robot] -> {args.out}")


if __name__ == "__main__":
    main()
