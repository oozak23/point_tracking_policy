#!/usr/bin/env python3
"""Print the gripper open/close timeline of a PointWorld episode.

Shows where the grasp is, so the geometric object-finding proposal can be checked by eye.

    python scripts/inspect_gripper.py AUTOLab+0d4edc83+2023-10-21-19h-03m-59s
    python scripts/inspect_gripper.py --random 5

Run with:  /home/oozak23/.conda/envs/pointworld-env/bin/python
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np

PW = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
          "/droid/flows-fs-optimized")


def show(episode: str):
    path = PW / f"{episode}_flows.h5"
    if not path.exists():
        print(f"  !! not in PointWorld: {episode}")
        return
    with h5py.File(path, "r") as f:
        wins = sorted((k for k in f.keys() if ":" in k),
                      key=lambda k: int(k.split(":")[0]))
        task = str(f.attrs.get("current_task", ""))
        print(f"\n=== {episode}")
        print(f"    task: {task[:96]}")
        print(f"    trajectory_length={int(f.attrs['trajectory_length'])}  "
              f"windows={len(wins)}")
        pos_mag = f["ee_pos_motion_magnitudes"].attrs
        grip_mv = f["has_gripper_movement"].attrs

        print(f"    {'window':>10s}  {'gripper_open':<13s} {'grip_pos':>16s} "
              f"{'ee_move_m':>9s} {'grip_mv':>7s}")
        prev_last = None
        for w in wins:
            c = f[w]
            go = np.asarray(c["gripper_open"]).ravel().astype(int)
            gp = np.asarray(c["gripper_positions"]).ravel()
            bar = "".join("-" if v else "#" for v in go)  # '-' open, '#' closed
            mark = ""
            seq = go if prev_last is None else np.concatenate([[prev_last], go])
            if ((seq[:-1] == 1) & (seq[1:] == 0)).any():
                mark += "  <== GRASP"
            if ((seq[:-1] == 0) & (seq[1:] == 1)).any():
                mark += "  <== RELEASE"
            prev_last = go[-1]
            print(f"    {w:>10s}  {bar:<13s} {gp[0]:7.3f}->{gp[-1]:6.3f} "
                  f"{pos_mag.get(w, float('nan')):9.3f} {str(bool(grip_mv.get(w, False))):>7s}"
                  f"{mark}")
        print("    legend: '-' gripper open, '#' gripper closed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*")
    ap.add_argument("--random", type=int, default=0)
    args = ap.parse_args()

    eps = list(args.episodes)
    if args.random:
        allf = [x[: -len("_flows.h5")] for x in os.listdir(PW) if x.endswith("_flows.h5")]
        random.seed(0)
        eps += random.sample(allf, args.random)
    if not eps:
        ap.error("give an episode id or --random N")
    for e in eps:
        show(e)


if __name__ == "__main__":
    main()
