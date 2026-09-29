#!/usr/bin/env python3
"""Calibrate the rigid-motion residual threshold from the data.

The scorer's r_max was guessed at 2 cm, and the first results sit right on it
(r=0.029 m against a 0.020 m cut), so the number is doing too much work to be left
unvalidated.

This looks at the residual distribution over ALL dense points in high-motion windows.
If the geometric idea holds, it should be bimodal: a large static population whose
residual tracks the gripper displacement, and a small attached population near zero.
Where the valley sits is the threshold; if there is no valley, the idea is weaker than
claimed and that needs to be known now rather than after building on it.

Also reports how many points pass at each candidate cut, which says whether a grasped
object is resolvable at all in a 320x180 depth map -- the thin-object worry.

    python analysis/08_calibrate_threshold.py --episodes 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.geom import rigid_residuals, window_bounds

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
CUTS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--min-grip", type=float, default=0.08,
                    help="only windows where the gripper really moved")
    ap.add_argument("--carrying-only", action="store_true",
                    help="only windows where the gripper is CLOSED throughout. Without "
                         "this the sample includes windows where the arm swings holding "
                         "nothing, which have no attached points by construction and "
                         "drag the whole distribution down.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "threshold_calibration.json")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(PW_FLOWS) if f.endswith("_flows.h5"))
    random.seed(args.seed)
    files = random.sample(files, min(args.episodes, len(files)))

    all_r, all_s, all_grip = [], [], []
    counts = {c: [] for c in CUTS}
    frac_of_scene = {c: [] for c in CUTS}
    n_win = 0

    for k, fn in enumerate(files, 1):
        try:
            with h5py.File(PW_FLOWS / fn, "r") as f:
                wins = sorted((w for w in f.keys() if ":" in w),
                              key=lambda w: window_bounds(w)[0])
                mags = f["ee_pos_motion_magnitudes"].attrs
                for w in wins:
                    if float(mags.get(w, 0.0)) < args.min_grip:
                        continue
                    if args.carrying_only:
                        go = np.asarray(f[w]["gripper_open"][:]).ravel().astype(bool)
                        if go.any():          # any open frame -> not purely carrying
                            continue
                    cams = [c for c in f[w].keys() if c.startswith("camera_")]
                    if not cams:
                        continue
                    g = f[w][cams[0]]
                    sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
                    vis = np.asarray(g["scene_visibility"][:])
                    dvm = np.asarray(g["scene_depth_valid_mask"][:])
                    pose = np.asarray(f[w]["gripper_pose"][:], dtype=np.float64)

                    ref = sf.shape[0] // 2
                    r, s, grip = rigid_residuals(sf, pose, ref=ref)
                    good = vis[ref] & dvm[ref]
                    if good.sum() < 500 or grip < args.min_grip:
                        continue

                    rg, sg = r[good], s[good]
                    n_win += 1
                    all_grip.append(grip)
                    take = np.random.choice(len(rg), min(3000, len(rg)), replace=False)
                    all_r.append(rg[take]); all_s.append(sg[take])
                    for c in CUTS:
                        m = (rg < c) & (sg > max(c, 0.3 * grip))
                        counts[c].append(int(m.sum()))
                        frac_of_scene[c].append(float(m.mean()))
        except Exception as e:
            print(f"  [skip] {fn}: {type(e).__name__}: {e}")
        if k % 10 == 0:
            print(f"  {k}/{len(files)} episodes, {n_win} windows")

    if not n_win:
        sys.exit("no usable windows")

    R = np.concatenate(all_r); S = np.concatenate(all_s)
    grip = np.array(all_grip)
    print(f"\n===== {n_win} high-motion windows, {len(R):,} sampled dense points =====")
    print(f"gripper displacement: median {np.median(grip):.3f} m")

    print(f"\nresidual r distribution (m) -- 'how unlike the gripper did it move'")
    for p in [1, 5, 10, 25, 50, 75, 90, 99]:
        print(f"  p{p:<3d} {np.percentile(R, p):7.4f}")

    print(f"\nmovement s distribution (m) -- 'did it move at all'")
    for p in [1, 5, 10, 25, 50, 75, 90, 99]:
        print(f"  p{p:<3d} {np.percentile(S, p):7.4f}")

    print(f"\nhistogram of r (log-spaced), looking for a valley between")
    print(f"an attached population near 0 and a static population near grip_disp:")
    edges = np.concatenate([[0], np.logspace(-3, 0, 16)])
    h, _ = np.histogram(R, bins=edges)
    for i in range(len(h)):
        bar = "#" * int(60 * h[i] / max(h.max(), 1))
        print(f"  [{edges[i]:6.4f},{edges[i+1]:6.4f}) {h[i]:8d} {bar}")

    print(f"\npoints passing each cut (per window, with s > max(cut, 0.3*grip)):")
    print(f"  {'cut (m)':>8s} {'median n':>9s} {'mean n':>8s} {'% of scene':>11s} "
          f"{'windows with >=20':>18s}")
    for c in CUTS:
        n = np.array(counts[c]); fr = np.array(frac_of_scene[c])
        print(f"  {c:8.3f} {np.median(n):9.0f} {n.mean():8.1f} {100*fr.mean():10.2f}% "
              f"{100*(n >= 20).mean():17.0f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_windows": n_win,
        "r_percentiles": {str(p): float(np.percentile(R, p)) for p in [1,5,10,25,50,75,90,99]},
        "s_percentiles": {str(p): float(np.percentile(S, p)) for p in [1,5,10,25,50,75,90,99]},
        "median_grip_disp_m": float(np.median(grip)),
        "cuts": {str(c): {"median_n": float(np.median(counts[c])),
                          "mean_n": float(np.mean(counts[c])),
                          "mean_frac_of_scene": float(np.mean(frac_of_scene[c])),
                          "frac_windows_ge20": float(np.mean(np.array(counts[c]) >= 20))}
                 for c in CUTS},
    }, open(args.out, "w"), indent=1)
    print(f"\n[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
