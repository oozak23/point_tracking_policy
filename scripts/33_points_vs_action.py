#!/usr/bin/env python3
"""Link the action curve to the POINTS: which points produced that motion?

Fatma hoca's question. The translation row of the action plot is one aggregate curve --
the end effector's displacement per frame. But we also track 221 points per clip, and the
robot half of them sit ON the gripper. So their image motion should BE that same motion,
projected. If it is not, either the tracking or the calibration is wrong.

Object points are the interesting case: they should move with the gripper only while
grasped, and independently (or not at all) otherwise. That is exactly the attachment test
`geom.rigid_residuals` already implements in 3D, seen here in 2D and for free.

Produces per clip:
  * correlation between |EE delta| and mean per-point image speed, split by role
  * a figure adding two rows to the action plot: robot-point speed and object-point speed

    python scripts/33_points_vs_action.py --per-lab 1
Run with any env with numpy + matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"


def point_speed(tracks, mask):
    """mean per-frame image displacement (px) over the points selected by `mask`."""
    d = np.full(len(tracks), np.nan)
    if mask.sum() == 0:
        return d
    p = tracks[:, mask]                                   # (T, n, 2)
    step = np.linalg.norm(np.diff(p, axis=0), axis=2)     # (T-1, n)
    with np.errstate(invalid="ignore"):
        d[:-1] = np.nanmean(step, axis=1)
    return d


def analyse(tk_npz: Path, act_npz: Path):
    d, a = np.load(tk_npz), np.load(act_npz)
    role = d["role"] if "role" in d.files else np.zeros(d["tracks"].shape[1], np.int8)
    T = min(len(d["tracks"]), len(a["valid"]))
    tracks = d["tracks"][:T]
    ee = np.linalg.norm(a["delta_pos_world"][:T], axis=1) * 100.0        # cm/frame
    rob = point_speed(tracks, role == 1)[:T]
    obj = point_speed(tracks, role == 0)[:T]
    grip = a["gripper_open_next"][:T, 0]
    v = a["valid"][:T] & np.isfinite(rob) & np.isfinite(obj) & (ee > 1e-6)

    def corr(x, y, m):
        if m.sum() < 10:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    closed = v & (grip < 0.5)
    return dict(T=int(T), n_valid=int(v.sum()),
                r_robot=corr(ee, rob, v), r_object=corr(ee, obj, v),
                r_object_grasping=corr(ee, obj, closed),
                n_grasping=int(closed.sum())), (ee, rob, obj, grip, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--actions", default="actions",
                    help="subdir holding action labels within each shard")
    ap.add_argument("--per-lab", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT / "points_vs_action")
    args = ap.parse_args()

    picked, by_lab = [], {}
    for npz in sorted(args.gen.glob("*/trackon/tracks/*.npz")):
        lab = npz.stem.split("+")[0]
        if len(by_lab.setdefault(lab, [])) < args.per_lab:
            by_lab[lab].append(npz); picked.append(npz)

    rows, series = [], []
    for tk in picked:
        act = tk.parents[2] / args.actions / "labels" / f"{tk.stem}.npz"
        if not act.exists():
            continue
        try:
            r, s = analyse(tk, act)
        except Exception as e:
            print(f"  {tk.stem[:44]} -- {type(e).__name__}: {e}"); continue
        r["video_id"] = tk.stem; rows.append(r); series.append((tk.stem, s))
        print(f"  {tk.stem.split('+')[0]:9s} r(EE, robot pts)={r['r_robot']:+.3f}   "
              f"r(EE, object pts)={r['r_object']:+.3f}   "
              f"while grasping={r['r_object_grasping']:+.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out / "report.json", "w"), indent=1)
    if rows:
        rr = np.array([r["r_robot"] for r in rows])
        ro = np.array([r["r_object"] for r in rows])
        rg = np.array([r["r_object_grasping"] for r in rows if np.isfinite(r["r_object_grasping"])])
        print(f"\n  robot points  vs EE : median r = {np.nanmedian(rr):+.3f}")
        print(f"  object points vs EE : median r = {np.nanmedian(ro):+.3f}")
        print(f"  object points, GRASPING only : median r = {np.nanmedian(rg):+.3f}")

    n = min(len(series), 4)
    fig = plt.figure(figsize=(11, 2.9 * n + .4))
    subs = np.atleast_1d(fig.subfigures(n + 1, 1, height_ratios=[.14] + [1] * n, hspace=.05))
    subs[0].suptitle("Which points produced the motion?  EE displacement vs point image speed",
                     fontsize=10)
    for sf, (name, (ee, rob, obj, grip, v)) in zip(subs[1:], series[:n]):
        ax = sf.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, .5],
                                                         "hspace": .12})
        t = np.arange(len(ee))
        ax[0].plot(t, np.where(v, ee, np.nan), lw=.9, color="#264653")
        ax[0].set_ylabel("EE delta\ncm/frame", fontsize=8)
        ax[0].set_title(name[:58], fontsize=9, loc="left")
        ax[1].plot(t, np.where(v, rob, np.nan), lw=.9, color="#2a9d8f", label="robot pts")
        ax[1].plot(t, np.where(v, obj, np.nan), lw=.9, color="#e9a13b", label="object pts")
        ax[1].set_ylabel("point speed\npx/frame", fontsize=8)
        ax[1].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
        ax[2].fill_between(t, 0, (grip < .5).astype(float), step="mid",
                           color="#8d99ae", alpha=.6)
        ax[2].set_ylabel("grasping", fontsize=8); ax[2].set_yticks([])
        ax[2].set_xlabel("frame", fontsize=8)
        for a_ in ax:
            a_.grid(alpha=.15, lw=.5); a_.margins(x=0); a_.tick_params(labelsize=7)
    fig.savefig(args.out / "points_vs_action.png", dpi=130)
    print(f"\n[pva] -> {args.out}")


if __name__ == "__main__":
    main()
