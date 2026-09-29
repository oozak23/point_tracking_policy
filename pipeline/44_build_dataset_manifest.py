#!/usr/bin/env python3
"""The training-set definition: which clips, how much each counts, and the train/val split.

Three things the decoder's dataloader needs, decided here rather than scattered through it.

1. A HARD CUT at --cut-px. Above it the tracker is not tracking -- the worst clip has a
   median error of 376 px on an 854-wide frame, so its "scene memory" holds tokens sampled
   from wherever the tracker wandered. There is nothing to learn from that, and it is
   different in kind from a merely noisy clip.

2. A per-clip WEIGHT for everything that survives. Quality is a smooth range from 3 px to
   50 px, not a binary, so a switch fits it badly: at a hard threshold a 29.9 px clip
   counts fully and a 30.1 px clip vanishes. A weight is a dial -- a noisy clip still
   teaches, it just cannot drag the model as hard as a clean one.

   It also protects LAB DIVERSITY, which is the concrete risk here. Bad clips cluster by
   lab, so any threshold silently deletes more of some labs than others -- and every
   quality number this project measured on AUTOLab alone failed to survive contact with
   the other eleven labs.

3. A TRAIN/VAL SPLIT BY EPISODE, not by clip. 3,048 episodes appear twice, once per
   camera: the same robot doing the same thing from two angles. Splitting by clip puts one
   angle in train and the other in val, and validation then measures memorisation.

   Note this does NOT eliminate similarity -- clips share only 63 lab/setup IDs, so val
   will contain scenes train has seen. Splitting on those 63 groups would be stronger but
   one of them holds 3,827 clips (28% of the data), making a balanced split impossible.
   Episode-level is the strongest split that is actually achievable; the residual is
   reported below rather than hidden.

    python pipeline/44_build_dataset_manifest.py
Run with any env with numpy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=OUT / "gt_all" / "report.json")
    ap.add_argument("--cut-px", type=float, default=50.0,
                    help="drop clips whose median tracking error exceeds this. 50 removes "
                         "the 1.4%% that are broken rather than noisy.")
    ap.add_argument("--weight-ref-px", type=float, default=7.72,
                    help="error at which a clip gets weight 1.0. Default is the dataset "
                         "median, so a typical clip counts fully and better-than-typical "
                         "clips are not up-weighted beyond it.")
    ap.add_argument("--min-weight", type=float, default=0.15,
                    help="floor, so a surviving clip is never silently deleted by having "
                         "its weight rounded to nothing.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "dataset_manifest.json")
    args = ap.parse_args()

    rows = json.load(open(args.report))
    med = np.array([r["median_px"] for r in rows])

    keep = med <= args.cut_px
    print(f"{len(rows):,} clips with a tracking measurement")
    print(f"  cut at {args.cut_px:.0f} px : keep {int(keep.sum()):,} "
          f"({keep.mean()*100:.1f}%), drop {int((~keep).sum()):,}")

    # weight 1.0 at or below the reference error, decaying as 1/error above it. Inverse
    # rather than linear because error is a distance: a clip twice as far off is roughly
    # half as informative, whereas a linear ramp would treat 50 px as merely "a bit worse".
    w = np.clip(args.weight_ref_px / np.maximum(med, args.weight_ref_px), args.min_weight, 1.0)

    # split by EPISODE so a clip's other camera angle cannot land on the far side
    rng = np.random.default_rng(args.seed)
    eps = sorted({r["video_id"].rsplit("__", 1)[0] for r, k in zip(rows, keep) if k})
    rng.shuffle(eps)
    n_val = int(len(eps) * args.val_frac)
    val_eps = set(eps[:n_val])

    out, per_lab = [], defaultdict(lambda: [0, 0])
    for r, k, wi in zip(rows, keep, w):
        if not k:
            continue
        ep = r["video_id"].rsplit("__", 1)[0]
        split = "val" if ep in val_eps else "train"
        out.append({"video_id": r["video_id"], "episode": ep, "split": split,
                    "weight": round(float(wi), 4),
                    "median_px": round(float(r["median_px"]), 2),
                    "n_robot": int(r["n_robot"])})
        per_lab[r["video_id"].split("+")[0]][split == "val"] += 1

    tr = [o for o in out if o["split"] == "train"]
    va = [o for o in out if o["split"] == "val"]
    wt = np.array([o["weight"] for o in tr])

    print(f"\n  train {len(tr):,} clips / {len({o['episode'] for o in tr}):,} episodes")
    print(f"  val   {len(va):,} clips / {len({o['episode'] for o in va}):,} episodes")
    print(f"  episode overlap between splits: "
          f"{len({o['episode'] for o in tr} & {o['episode'] for o in va})}  (must be 0)")

    print(f"\n  train weights: mean {wt.mean():.3f}  "
          f"at floor {np.mean(wt <= args.min_weight + 1e-9)*100:.1f}%  "
          f"at 1.0 {np.mean(wt >= 1 - 1e-9)*100:.1f}%")
    print(f"  effective training size {wt.sum():,.0f} of {len(wt):,} clips "
          f"({wt.sum()/len(wt)*100:.1f}%)")

    print(f"\n  {'lab':<10s} {'train':>7s} {'val':>6s} {'val %':>7s}")
    for lab in sorted(per_lab):
        a, b = per_lab[lab]
        print(f"  {lab:<10s} {a:7d} {b:6d} {b/max(a+b,1)*100:6.1f}%")

    json.dump({"config": {"cut_px": args.cut_px, "weight_ref_px": args.weight_ref_px,
                          "min_weight": args.min_weight, "val_frac": args.val_frac,
                          "seed": args.seed, "split_unit": "episode"},
               "clips": out}, open(args.out, "w"))
    print(f"\n[manifest] {len(out):,} clips -> {args.out}")


if __name__ == "__main__":
    main()
