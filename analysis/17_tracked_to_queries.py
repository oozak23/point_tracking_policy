#!/usr/bin/env python3
"""Turn Track-On's TRACKED positions at a chosen frame back into a scoreable point set.

Needed to evaluate a frame-0 seed. Seeding at frame 0 removes the blind window (7-40% of
each clip had an empty scene memory, because the detection frame comes from the motion
window and Track-On is causal). But a frame-0 seed cannot be scored where it is placed:
at frame 0 the gripper holds nothing, so the rigid-motion test is undefined.

So we score it later -- take where Track-On has CARRIED the points to at the original
detection frame, and hand those to the usual scorer. That measures grounding and
tracking together, and lands on exactly the frame `mm_sam` was scored at, so the two are
directly comparable.

Invisible points are dropped: a point Track-On has lost should not count as evidence
that we are on the object.

    python analysis/17_tracked_to_queries.py \
        --tracks out/trackon_mm_sam_f0/tracks --ref out/mm_sam/query_points \
        --out out/tracked_f0_at_det
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, required=True,
                    help="dir of <vid>.npz from pipeline/16_run_trackon.py")
    ap.add_argument("--ref", type=Path, required=True,
                    help="query-point dir naming the frame to evaluate at (e.g. out/mm_sam/query_points)")
    ap.add_argument("--out", type=Path, default=OUT / "tracked_at_ref")
    args = ap.parse_args()

    ref_frame = {}
    for p in sorted(args.ref.glob("*.npz")):
        m = NAME_RE.match(p.name)
        if m:
            ref_frame[m["vid"]] = int(m["frame"])

    qdir = args.out / "query_points"
    qdir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    n_kept = []
    for p in sorted(args.tracks.glob("*.npz")):
        vid = p.stem
        if vid not in ref_frame:
            skipped += 1
            continue
        f = ref_frame[vid]
        d = np.load(p)
        tracks, vis = d["tracks"], d["visibility"]
        if f >= tracks.shape[0]:
            skipped += 1
            continue
        pts = tracks[f]
        keep = vis[f] & np.isfinite(pts).all(axis=1)
        if keep.sum() < 5:
            skipped += 1
            continue
        pts = pts[keep]
        qp = np.concatenate(
            [np.full((len(pts), 1), f), np.rint(pts)], axis=1).astype(np.int32)
        np.savez(qdir / f"{vid}_f{f}.npz",
                 query_points=qp, dim=np.array([480, 854], dtype=np.int32))
        written += 1
        n_kept.append(int(keep.sum()))

    print(f"[tracked] wrote {written}, skipped {skipped}")
    if n_kept:
        print(f"[tracked] visible points carried to the reference frame: "
              f"median {np.median(n_kept):.0f} of 100 (min {min(n_kept)}, max {max(n_kept)})")
    print(f"[tracked] -> {qdir}")


if __name__ == "__main__":
    main()
