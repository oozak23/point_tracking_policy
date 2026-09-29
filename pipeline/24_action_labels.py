#!/usr/bin/env python3
"""Extract the policy's supervision targets from PointWorld, aligned to our video frames.

This is the answer key. Everything else in the pipeline builds the policy's INPUT --
scene tokens from frozen Track-On, robot tokens from FK. This produces what the model is
compared against: what the demonstrator actually did at each frame.

All six proprioception fields are extracted, not just the chosen action space. They live
in the same h5 windows, so reading them all costs nothing, and it means DECISIONS.md #2
(EE delta vs joint velocities vs absolute pose) can be settled AFTER the data exists
instead of blocking it.

Frame alignment is the identity at 7.2 Hz -- one PointWorld step per video frame -- which
is a large part of why that rate was chosen (DECISIONS.md #6). At 3.6 Hz half the labels
were being discarded.

    python pipeline/24_action_labels.py --like out/smoke7_obj
Run with $PY_POINTWORLD  (needs h5py)
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

from droid_qp import rates, video_io
from droid_qp.actions import droid_action_series, episode_action_series, pose_deltas
from droid_qp.index import load_manifest
from droid_qp.robot_points import PW_FLOWS

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, default=OUT / "smoke7_obj",
                    help="run dir whose clips to mirror (uses its release_videos for T)")
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--source", choices=["droid", "pointworld"], default="droid",
                    help="where proprioception comes from. DEFAULT droid: the fields are "
                         "bit-identical (verified 0.000e+00 on all of them) but DROID has "
                         "every step, while PointWorld's windows cover a median 75%% of an "
                         "episode. Same labels, ~33%% more of them. `pointworld` is kept "
                         "only so the two can be diffed.")
    ap.add_argument("--out", type=Path, default=OUT / "action_labels")
    args = ap.parse_args()

    vids = []
    for p in sorted((args.like / "query_points").glob("*.npz")):
        m = NAME_RE.match(p.name)
        if m:
            vids.append(m["vid"])
    if args.n:
        vids = vids[: args.n]
    print(f"[act] {len(vids)} clips | {rates.describe()}\n")

    adir = args.out / "labels"
    adir.mkdir(parents=True, exist_ok=True)

    # Accumulated for the normalisation statistics. Position deltas are metres and
    # rotation deltas are unitless; without per-dimension normalisation the loss would be
    # dominated by whichever happens to have the larger numeric range, not by whichever
    # matters. Computed over the TRAINING set only in the real run -- here over whatever
    # was passed, and recorded as such.
    acc = {"delta_pos_body": [], "delta_rot6d_body": [], "joint_velocities": []}
    report = []

    by_ep = {}
    if args.source == "droid":
        by_ep = {r.video_id.rsplit("__", 1)[0]: r for r in load_manifest()}

    for i, vid in enumerate(vids, 1):
        ep, _cam = vid.rsplit("__", 1)
        try:
            if args.source == "droid":
                rec = by_ep.get(ep)
                if rec is None or not rec.trajectory_h5.exists():
                    print(f"  [{i}] {vid[:44]} -- no DROID trajectory.h5")
                    continue
                with h5py.File(rec.trajectory_h5, "r") as f:
                    fields, filled = droid_action_series(f, rates.SUBSAMPLE)
            else:
                with h5py.File(PW_FLOWS / f"{ep}_flows.h5", "r") as f:
                    fields, filled = episode_action_series(f)
            if not fields:
                print(f"  [{i}] {vid[:44]} -- no windows")
                continue

            mp4 = args.like / "release_videos" / f"{vid}.mp4"
            if not mp4.exists():
                print(f"  [{i}] {vid[:44]} -- no video")
                continue
            T = len(video_io.read_frames(mp4))

            d = pose_deltas(fields["gripper_pose"], filled)

            # PointWorld step -> release frame. Identity at 7.2 Hz. Steps beyond the video
            # are dropped; frames beyond PointWorld's coverage stay invalid rather than
            # being back-filled, since a fabricated action is worse than a masked one.
            S = len(filled)
            rel = rates.pw_step_to_release(np.arange(S))
            keep = rel < T
            idx = rel[keep]

            def to_frames(a):
                z = np.zeros((T,) + a.shape[1:], dtype=np.float32)
                z[idx] = a[keep]
                return z

            valid = np.zeros(T, dtype=bool)
            valid[idx] = filled[keep] & d["valid_delta"][keep]

            out = {k: to_frames(v) for k, v in d.items() if k != "valid_delta"}
            for k in ("joint_positions", "joint_velocities", "joint_torques",
                      "gripper_positions"):
                out[k] = to_frames(fields[k])
            # Gripper: the STATE now, and the state at t+1 which is what the policy
            # commands. A binary open/close action is the next state, not the current one.
            go = fields["gripper_open"].astype(np.float32)
            out["gripper_open"] = to_frames(go)
            out["gripper_open_next"] = to_frames(
                np.concatenate([go[1:], go[-1:]], axis=0))
            out["valid"] = valid
            out["subsample"] = np.int32(rates.SUBSAMPLE)

            np.savez(adir / f"{vid}.npz", **out)

            for k in acc:
                acc[k].append(out[k][valid])
            span = float(np.linalg.norm(out["delta_pos_body"][valid], axis=1).max()
                         if valid.any() else 0.0)
            print(f"  [{i:3d}] {vid[:44]:44s} T={T:4d} S={S:4d} "
                  f"valid={int(valid.sum()):4d} ({valid.mean()*100:4.1f}%) "
                  f"max|dp|={span*100:5.2f} cm")
            report.append({"video_id": vid, "T": T, "pw_steps": int(S),
                           "n_valid": int(valid.sum()),
                           "frac_valid": float(valid.mean())})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{i}] {vid[:44]} -- {type(e).__name__}: {e}")

    stats = {}
    for k, v in acc.items():
        if not v:
            continue
        a = np.concatenate(v, axis=0)
        stats[k] = {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-8).tolist(),
                    "n": int(len(a))}
    json.dump({"report": report, "norm_stats": stats,
               "norm_stats_computed_over": "these clips only -- recompute on the TRAIN "
                                           "split before training"},
              open(args.out / "report.json", "w"), indent=1)

    print("\n===== summary =====")
    print(f"clips           : {len(report)}")
    if report:
        fv = np.array([r["frac_valid"] for r in report])
        print(f"valid frames    : mean {fv.mean()*100:.1f}%  min {fv.min()*100:.1f}%")
        if "delta_pos_body" in stats:
            s = np.array(stats["delta_pos_body"]["std"])
            print(f"delta pos std   : {s*100} cm/frame over {stats['delta_pos_body']['n']} frames")
    print(f"[act] -> {args.out}")


if __name__ == "__main__":
    main()
