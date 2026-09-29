#!/usr/bin/env python3
"""How good is the tracking, clip by clip? -- the histogram DECISIONS.md #10 needs.

Each clip gets one number: the median distance, in pixels, between where Track-On says a
gripper point is and where forward kinematics says it actually is. FK is exact, so this is
a true error, not a disagreement between two guesses.

The point of plotting it rather than quoting the average: the average (7.72 px) describes
almost none of the clips. The distribution is long-tailed -- most clips are fine, a minority
are catastrophic -- and a threshold can only be chosen sensibly by looking at the shape.

    python scripts/43_quality_histogram.py
Run with any env with numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=OUT / "gt_all" / "report.json")
    ap.add_argument("--out", type=Path, default=OUT / "gt_all" / "quality_hist.png")
    args = ap.parse_args()

    r = json.load(open(args.report))
    med = np.array([x["median_px"] for x in r])
    labs = [x["video_id"].split("+")[0] for x in r]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # (1) the distribution, log-x -- a linear axis wastes 90% of the width on the tail
    bins = np.logspace(np.log10(max(med.min(), .5)), np.log10(med.max()), 60)
    ax[0].hist(med, bins=bins, color="#2a9d8f", alpha=.85)
    for v, c, lbl in [(np.median(med), "#264653", "median"),
                      (10, "#e9a13b", "10 px"), (50, "#d1495b", "50 px")]:
        ax[0].axvline(v, color=c, ls="--", lw=1.2)
        ax[0].annotate(f"{lbl} {v:.1f}" if lbl == "median" else lbl,
                       (v, ax[0].get_ylim()[1] * .92), rotation=90, fontsize=8,
                       color=c, ha="right", va="top")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("per-clip median error (px, log scale)")
    ax[0].set_ylabel("clips")
    ax[0].set_title(f"{len(med):,} clips", fontsize=10, loc="left")

    # (2) what a threshold actually costs -- the only view that answers "where to cut"
    ts = np.logspace(np.log10(2), np.log10(200), 120)
    keep = [(med <= t).mean() * 100 for t in ts]
    ax[1].plot(ts, keep, color="#264653", lw=1.8)
    for t in (10, 20, 50, 100):
        k = (med <= t).mean() * 100
        ax[1].plot([t], [k], "o", color="#d1495b")
        ax[1].annotate(f"{t} px -> keep {k:.0f}%", (t, k), textcoords="offset points",
                       xytext=(6, -10), fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("threshold (px)"); ax[1].set_ylabel("clips kept (%)")
    ax[1].set_title("Cost of a hard cut", fontsize=10, loc="left")
    ax[1].grid(alpha=.2, lw=.5)

    # (3) by lab -- a threshold is not lab-neutral, which is the real risk
    by = defaultdict(list)
    for l, m in zip(labs, med):
        by[l].append(m)
    order = sorted(by, key=lambda k: np.median(by[k]))
    pos = np.arange(len(order))
    ax[2].barh(pos, [np.median(by[k]) for k in order], color="#8d99ae", alpha=.9)
    for i, k in enumerate(order):
        frac = np.mean(np.array(by[k]) > 50) * 100
        ax[2].annotate(f"n={len(by[k])}  {frac:.0f}% >50px", (0.4, i), fontsize=7,
                       va="center", color="#264653")
    ax[2].set_yticks(pos); ax[2].set_yticklabels(order, fontsize=8)
    ax[2].axvline(10, color="#e9a13b", ls="--", lw=1)
    ax[2].set_xlabel("median error (px)")
    ax[2].set_title("A cut is not lab-neutral", fontsize=10, loc="left")

    fig.suptitle("Tracking quality per clip -- Track-On vs exact FK ground truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)

    print(f"{len(med):,} clips\n")
    print("  percentile of per-clip median error:")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f"    p{q:<3d} {np.percentile(med, q):8.2f} px")
    print(f"    max  {med.max():8.2f} px")
    print("\n  what a threshold would keep:")
    for t in (10, 20, 30, 50, 100):
        print(f"    <= {t:3d} px : {(med <= t).mean()*100:5.1f}% of clips "
              f"({int((med <= t).sum()):,})")
    print(f"\n[hist] -> {args.out}")


if __name__ == "__main__":
    main()
