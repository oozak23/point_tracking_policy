#!/usr/bin/env python3
"""Use MolmoMotion's released tracks as QUERY POINTS, and score them like ours.

This is a different use from the one rejected earlier. As *labels* their tracks are
unusable (ViPE-lifted 3D, 13-20% teleport). As *query points* we borrow only one row per
track -- the birth position (t, x, y) -- and Track-On does its own tracking, so none of
their trajectory data comes along.

Attractive because it is free: no GPU, no MolmoPoint, no SAM, and it covers all 14,083
overlap clips versus the 212 we have run our own pipeline on. The open question is
whether their GROUNDING is any good; their tracks sit on background shelving in
out/diag_weird.png. This converts them into our npz format at the SAME detection frame
our own runs used, so scripts/05_score_points.py judges both on identical footing.

No mask is stored, so the scorer falls back to "within 12 px of a query point", which
approximates the region their k-means points were drawn from.

    python scripts/14_molmomotion_as_queries.py --like out/nat_geo --out out/mm_queries
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRACKS_TAR = Path("/frozen/avg/ozan/molmo-motion-1m/droid/tracks/tracks-0000.tar")
OUT = Path(__file__).resolve().parent.parent / "out"
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids-from", type=Path, default=None,
                    help="json list of {video_id: ...} -- e.g. out/overlap_manifest.json. "
                         "Use INSTEAD of --like to seed clips that have no run dir yet, "
                         "which is the case for all 14,083 at scale. Requires --frame.")
    ap.add_argument("--like", type=Path, default=None,
                    help="run dir whose clips+detection frames to mirror, e.g. out/nat_geo")
    ap.add_argument("--out", type=Path, default=OUT / "mm_queries")
    ap.add_argument("--max-points", type=int, default=100)
    ap.add_argument("--frame", type=int, default=None,
                    help="Override the frame to sample at. Use 0 to seed at the start of "
                         "the clip: their AllTracker is BIDIRECTIONAL so it has object "
                         "positions at frame 0, whereas our detection frame comes from "
                         "the motion window and leaves 7-40%% of each clip with an empty "
                         "scene memory (Track-On is causal). Default mirrors --like.")
    args = ap.parse_args()

    if (args.like is None) == (args.video_ids_from is None):
        ap.error("pass exactly one of --like or --video-ids-from")

    if args.video_ids_from:
        if args.frame is None:
            ap.error("--video-ids-from needs --frame (there is no run dir to inherit "
                     "detection frames from); --frame 0 is the seeding path we use")
        want = {r["video_id"]: args.frame
                for r in json.load(open(args.video_ids_from))}
        print(f"[mm] {len(want)} clips from {args.video_ids_from} at frame {args.frame}")
    else:
        want = {}
        for p in sorted((args.like / "query_points").glob("*.npz")):
            m = NAME_RE.match(p.name)
            if m:
                want[m["vid"]] = int(m["frame"])
        print(f"[mm] mirroring {len(want)} clips from {args.like}")

    # The tar is streamed ONCE and filtered by `want` (see the loop below), so this scales
    # to all 14,083 clips in a single 2.3 GB sequential pass. Never shard this stage: 141
    # array tasks would each re-stream the whole tar off /frozen for a 100-clip slice.

    qdir = args.out / "query_points"
    qdir.mkdir(parents=True, exist_ok=True)

    written = missing = empty = 0
    seen = set()
    with tarfile.open(TRACKS_TAR) as t:
        for mem in t:
            if not mem.name.endswith("_2d.npz"):
                continue
            vid = Path(mem.name).name[: -len("_2d.npz")]
            if vid not in want:
                continue
            seen.add(vid)
            d = np.load(io.BytesIO(t.extractfile(mem).read()))
            tracks, vis = d["tracks_2d"], d["visibility"]
            f = want[vid] if args.frame is None else args.frame
            if f >= tracks.shape[0]:
                # Their T axis is the release clip; our detection frame can exceed it.
                empty += 1
                continue
            pts = tracks[f][vis[f]]
            if len(pts) == 0:
                empty += 1
                continue
            if len(pts) > args.max_points:
                idx = np.linspace(0, len(pts) - 1, args.max_points).astype(int)
                pts = pts[idx]
            qp = np.concatenate(
                [np.full((len(pts), 1), f), np.rint(pts)], axis=1).astype(np.int32)
            np.savez(qdir / f"{vid}_f{f}.npz",
                     query_points=qp, dim=np.array([480, 854], dtype=np.int32))
            written += 1
            if len(seen) == len(want):
                break

    missing = len(want) - len(seen)
    print(f"[mm] wrote {written} | no released tracks {missing} | "
          f"no visible points at that frame {empty}")
    print(f"[mm] -> {qdir}")


if __name__ == "__main__":
    main()
