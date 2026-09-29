#!/usr/bin/env python3
"""Plot the action labels, so the supervision signal can be looked at rather than trusted.

Numbers hid a real bug here once: deltas computed across PointWorld coverage gaps read as
51-75 cm per frame (>5 m/s at the end effector) and nothing in the summary statistics
looked wrong until the physics was checked. A plot makes both the gaps and the magnitudes
obvious at a glance, which is the point.

Four rows per clip:
  1. translation delta per axis -- the bulk of the action
  2. rotation delta magnitude, in degrees per frame
  3. gripper: continuous width, and the binary command the policy predicts
  4. the validity mask, i.e. where PointWorld covers the episode

    python scripts/25_viz_actions.py --labels out/action_labels/labels
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import rates
from droid_qp.actions import sixd_to_rot

OUT = Path(__file__).resolve().parent.parent / "out"


def plot_clip(d, title, ax_rows):
    v = d["valid"]
    T = len(v)
    t = np.arange(T) / rates.RELEASE_HZ          # seconds, not frames -- comparable across clips
    a0, a1, a2, a3 = ax_rows

    dp = d["delta_pos_body"] * 100.0             # cm
    for k, (lbl, c) in enumerate(zip("xyz", ["#d1495b", "#2a9d8f", "#3d5a80"])):
        a0.plot(t, np.where(v, dp[:, k], np.nan), lw=0.9, color=c, label=f"d{lbl}")
    a0.set_ylabel("translation\ncm / frame")
    a0.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)
    a0.set_title(title, fontsize=9, loc="left")

    R = sixd_to_rot(d["delta_rot6d_body"].astype(np.float64))
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    a1.plot(t, np.where(v, np.degrees(np.arccos(cos)), np.nan), lw=0.9, color="#e07a5f")
    a1.set_ylabel("rotation\ndeg / frame")

    a2.plot(t, d["gripper_positions"][:, 0], lw=0.9, color="#8d99ae", label="width")
    a2.plot(t, d["gripper_open_next"][:, 0], lw=1.2, color="#264653",
            label="open (command)")
    a2.set_ylabel("gripper")
    a2.legend(loc="center right", fontsize=7, frameon=False)

    a3.fill_between(t, 0, v.astype(float), step="mid", color="#2a9d8f", alpha=.55)
    a3.set_ylabel("valid", fontsize=8)
    a3.set_ylim(-.1, 1.1)
    a3.set_yticks([])
    a3.set_xlabel("seconds", fontsize=8)
    # Every panel of a clip shares its time axis, but clips have different durations, so
    # sharex is per-clip rather than global. Tick labels only on the bottom panel -- with
    # them on all four they collide with the next clip's title.
    for a in (a0, a1, a2, a3):
        a.grid(alpha=.15, lw=.5)
        a.margins(x=0)
        a.tick_params(labelsize=7)
        a.set_xlim(0, t[-1] if len(t) else 1)
    for a in (a0, a1, a2):
        a.set_xticklabels([])
    for a, lab in ((a0, "translation\ncm/frame"), (a1, "rotation\ndeg/frame"),
                   (a2, "gripper")):
        a.set_ylabel(lab, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=OUT / "action_labels" / "labels")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--diverse", action="store_true",
                    help="pick one clip per lab rather than the first N alphabetically")
    ap.add_argument("--out", type=Path, default=OUT / "action_labels" / "actions.png")
    args = ap.parse_args()

    files = sorted(args.labels.glob("*.npz"))
    if args.diverse:
        # One clip per lab. Sorting alphabetically puts AUTOLab first, and AUTOLab is only
        # 14.8% of the overlap set -- a figure built from the first N files would show a
        # single lab's tabletop and misrepresent the data.
        by_lab = {}
        for f in files:
            by_lab.setdefault(f.stem.split("+")[0], f)
        files = [by_lab[k] for k in sorted(by_lab)]
    files = files[: args.n]
    if not files:
        sys.exit(f"no label npz under {args.labels}")

    # One subfigure per clip: clips have independent time axes and their own title, and
    # a single flat gridspec makes the titles collide with the panel above.
    # A leading spacer subfigure holds the suptitle; without it the title lands on top of
    # the first clip's own title, since subfigures fill the whole figure height.
    fig = plt.figure(figsize=(11, 3.4 * len(files) + 0.5))
    rows = fig.subfigures(len(files) + 1, 1, hspace=0.06,
                          height_ratios=[0.16] + [1.0] * len(files))
    rows[0].suptitle(f"Action labels -- {rates.describe()}", fontsize=10)
    for sf, f in zip(np.atleast_1d(rows[1:]), files):
        d = np.load(f)
        axes = sf.subplots(4, 1, sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.1, 1.1, .35],
                                        "hspace": .12})
        plot_clip(d, f.stem[:58], axes)


    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[viz] {len(files)} clips -> {args.out}")


if __name__ == "__main__":
    main()
