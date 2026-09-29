#!/usr/bin/env python3
"""When a tracked point is occluded and reappears, does the tracker come back to the RIGHT
point? -- one of the four questions Fatma hoca wants examples of.

This one can be ANSWERED rather than illustrated. Robot points carry exact FK ground truth
at every frame (joint encoders -> URDF -> calibrated projection), so at the moment a point
returns from occlusion we know precisely where it should be. No squinting at a video.

An event is: visible -> invisible for >= --min-gap frames -> visible again. At the return
frame we measure the distance between what Track-On says and what FK says. A tracker that
re-identifies correctly lands within a few px, at roughly its steady-state error (~4.5 px
median, see 34_gt_trajectories.py); one that has latched onto a different piece of the
scene lands far away and STAYS wrong, which is the failure worth showing.

Object points are deliberately excluded -- nothing observed where they really were, so
there is no ground truth to check a re-acquisition against.

    python pipeline/40_occlusion_recovery.py --gt out/gt_all
Run with any env with numpy (+ cv2 for the figure).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def events(vis_col: np.ndarray, min_gap: int):
    """-> list of (t_lost, t_back) for visible -> gap >= min_gap -> visible."""
    s = vis_col.astype(np.int8)
    d = np.diff(s)
    offs = np.where(d == -1)[0]          # last visible frame
    ons = np.where(d == 1)[0]            # frame before it returns
    out = []
    for o in offs:
        nxt = ons[ons > o]
        if len(nxt) and (nxt[0] - o) >= min_gap:
            out.append((int(o), int(nxt[0] + 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=OUT / "gt_all")
    ap.add_argument("--min-gap", type=int, default=3,
                    help="frames a point must stay invisible to count as an occlusion")
    ap.add_argument("--bad-px", type=float, default=20.0,
                    help="a return landing further than this is treated as a failure. "
                         "Track-On's steady-state median is ~4.5 px, so 20 is well "
                         "outside normal tracking error rather than a borderline call.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.gt

    files = sorted(args.gt.glob("*.npz"))
    if not files:
        raise SystemExit(f"no ground-truth npz under {args.gt} -- run 39_export_gt.sbatch")

    per_clip, err_all, gap_all = [], [], []
    for f in files:
        d = np.load(f)
        need = {"robot_gt", "robot_gt_ok", "robot_tracked", "robot_tracked_vis"}
        if not need.issubset(set(d.files)):
            continue
        gt, ok, tr, tv = (d["robot_gt"], d["robot_gt_ok"],
                          d["robot_tracked"], d["robot_tracked_vis"])
        errs, gaps = [], []
        for j in range(tv.shape[1]):
            for t_lost, t_back in events(tv[:, j], args.min_gap):
                if t_back >= len(gt) or not ok[t_back, j]:
                    continue          # no ground truth at the return frame
                e = float(np.linalg.norm(tr[t_back, j] - gt[t_back, j]))
                if np.isfinite(e):
                    errs.append(e); gaps.append(t_back - t_lost)
        if errs:
            e = np.array(errs)
            per_clip.append({"video_id": f.stem, "n_events": len(e),
                             "median_px": float(np.median(e)),
                             "frac_bad": float((e > args.bad_px).mean())})
            err_all.append(e); gap_all.append(np.array(gaps))

    if not err_all:
        raise SystemExit("no occlusion-and-return events found")
    e = np.concatenate(err_all); g = np.concatenate(gap_all)

    print(f"{len(e):,} occlusion->return events over {len(per_clip)} clips "
          f"(gap >= {args.min_gap} frames)\n")
    print(f"  error at the frame the point RETURNS:")
    print(f"    median {np.median(e):6.2f} px   p90 {np.percentile(e, 90):7.2f} px")
    print(f"    within  5 px: {(e <= 5).mean()*100:5.1f}%")
    print(f"    within 10 px: {(e <= 10).mean()*100:5.1f}%")
    print(f"    beyond {args.bad_px:.0f} px: {(e > args.bad_px).mean()*100:5.1f}%   "
          f"<- came back to the WRONG place")

    print(f"\n  does a longer occlusion make recovery worse?")
    for lo, hi in [(3, 5), (5, 10), (10, 20), (20, 50), (50, 10**6)]:
        m = (g >= lo) & (g < hi)
        if m.sum() > 20:
            lbl = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
            print(f"    gap {lbl:>7s} frames : median {np.median(e[m]):6.2f} px  "
                  f"bad {np.mean(e[m] > args.bad_px)*100:5.1f}%  (n={m.sum():,})")

    per_clip.sort(key=lambda r: -r["frac_bad"])
    print(f"\n  worst clips (highest share of wrong re-acquisitions):")
    for r in per_clip[:6]:
        print(f"    {r['video_id'][:46]:46s} {r['n_events']:4d} events  "
              f"{r['frac_bad']*100:5.1f}% bad  median {r['median_px']:6.2f} px")

    json.dump({"per_clip": per_clip,
               "overall": {"n_events": int(len(e)), "median_px": float(np.median(e)),
                           "frac_bad": float((e > args.bad_px).mean()),
                           "bad_px": args.bad_px, "min_gap": args.min_gap}},
              open(out / "occlusion_recovery.json", "w"), indent=1)
    print(f"\n[occ] -> {out / 'occlusion_recovery.json'}")


if __name__ == "__main__":
    main()
