#!/usr/bin/env python3
"""What should --min-visible be? The birth-frame trade, measured.

Track-On births are ONCE-ONLY: `Predictor.forward` initialises a query only when
`query_times == t` (trackon_predictor.py:193-275). So a robot point not visible at the
chosen birth step is not merely absent early -- it is absent for the ENTIRE clip.

That makes the threshold a real trade rather than a tuning detail:

    low  (5)   births early, few blind frames, but few points ever born
               -- measured on AUTOLab+5d05c5aa+...: born@f8 with 32 of 128 points,
               so 96 slots stay dead for all 451 frames
    high (96)  births late, most points born, but the opening frames have no robot
               representation at all

Note this does NOT change the fixed-dimension property: FK point n is the same physical
spot in every clip (`presample(n, seed)` is deterministic), so the policy input is 128
fixed slots plus a validity mask either way. The threshold decides how many of those slots
are ever populated, not whether the layout is stable.

Reads the PointWorld h5 directly so every threshold is evaluated from one FK pass.

    python scripts/23_sweep_min_visible.py --like out/smoke7_obj
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

from droid_qp import rates, video_io
from droid_qp.robot_points import (PW_FLOWS, episode_joint_series, load_sampler,
                                   robot_points_2d, robot_points_3d, window_calibration)

OUT = Path(__file__).resolve().parent.parent / "out"
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, default=OUT / "smoke7_obj")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--num-points", type=int, default=128)
    ap.add_argument("--thresholds", type=int, nargs="+",
                    default=[5, 16, 32, 48, 64, 80, 96])
    ap.add_argument("--out", type=Path, default=OUT / "min_visible_sweep.json")
    args = ap.parse_args()

    vids = []
    for p in sorted((args.like / "query_points").glob("*.npz")):
        m = NAME_RE.match(p.name)
        if m:
            vids.append(m["vid"])
    vids = vids[: args.n]
    print(f"[sweep] {len(vids)} clips | {rates.describe()}\n")

    sampler = load_sampler(args.num_points, gripper_only=True, seed=0)
    rows = []
    for vid in vids:
        ep, cam = vid.rsplit("__", 1)
        with h5py.File(PW_FLOWS / f"{ep}_flows.h5", "r") as f:
            jp, gp, filled = episode_joint_series(f)
            if jp is None:
                continue
            w0 = sorted((k for k in f.keys() if ":" in k),
                        key=lambda k: int(k.split(":")[0]))[0]
            if f"camera_{cam}_ext" not in f[w0]:
                continue
            E, K = window_calibration(f, w0, cam)

        uv, ok = robot_points_2d(robot_points_3d(sampler, jp, gp), E, K)
        T = len(video_io.read_frames(args.like / "release_videos" / f"{vid}.mp4"))
        rel = rates.pw_step_to_release(np.arange(len(uv)))
        covered = (rel < T) & filled
        nvis = ok.sum(axis=1)

        rec = {"video_id": vid, "T": int(T), "by_threshold": {}}
        for th in args.thresholds:
            keep = covered & (nvis >= th)
            if not keep.any():
                rec["by_threshold"][str(th)] = None
                continue
            s0 = int(np.argmax(keep))
            f0 = int(rel[s0])
            rec["by_threshold"][str(th)] = {
                "birth_frame": f0,
                "n_born": int(nvis[s0]),
                "blind_frames": f0,
                # slot-frames the policy actually gets: born points x frames after birth.
                # This is the quantity the trade is really about -- an early birth with
                # few points and a late birth with many can be compared directly.
                "coverage": int(nvis[s0]) * (T - f0),
            }
        rows.append(rec)

    print(f"{'threshold':>9s} {'clips ok':>9s} {'med birth':>10s} {'med born':>9s} "
          f"{'med blind':>10s} {'mean coverage':>14s}")
    summary = {}
    for th in args.thresholds:
        vals = [r["by_threshold"][str(th)] for r in rows
                if r["by_threshold"].get(str(th))]
        if not vals:
            print(f"{th:9d} {'0':>9s}")
            continue
        born = np.array([v["n_born"] for v in vals])
        birth = np.array([v["birth_frame"] for v in vals])
        cov = np.array([v["coverage"] for v in vals], dtype=float)
        summary[str(th)] = {"clips": len(vals), "median_born": float(np.median(born)),
                            "median_birth": float(np.median(birth)),
                            "mean_coverage": float(cov.mean())}
        print(f"{th:9d} {len(vals):9d} {np.median(birth):10.1f} {np.median(born):9.1f} "
              f"{np.median(birth):10.1f} {cov.mean():14.0f}")

    best = max(summary, key=lambda k: summary[k]["mean_coverage"]) if summary else None
    print(f"\nhighest mean coverage (born points x frames alive): --min-visible {best}")
    print("Coverage is the right single number here, but it weights a long clip more "
          "than a short one;\ncheck the per-clip rows in the json before committing.")
    json.dump({"per_clip": rows, "summary": summary, "best_by_coverage": best},
              open(args.out, "w"), indent=1)
    print(f"[sweep] -> {args.out}")


if __name__ == "__main__":
    main()
