#!/usr/bin/env python3
"""Split the 14,083 overlap clips into shards, one per SLURM array task.

Shard granularity is the unit of failure and of resume. 100 clips is roughly an hour of
work, which is well inside the 8 h partition limit and small enough that re-running a
failed shard is cheap. Finer shards would multiply the fixed per-task cost (loading SAM
twice and Track-On once); coarser ones risk the wall clock.

Assignment is by a STABLE HASH of the video_id, not by position, so adding or reordering
clips later does not reshuffle every shard and invalidate finished work.

Clips are also stratified-checked here rather than at read time: the report prints the lab
distribution, because every clip we have looked at so far is AUTOLab and the position
probe (22_probe_position.py) cannot be trusted until it runs across labs.

    python scripts/27_build_shards.py --shard-size 100
Run with any env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"


def stable_shard(video_id: str, n_shards: int) -> int:
    h = hashlib.blake2b(video_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=OUT / "overlap_manifest.json")
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N clips")
    ap.add_argument("--out", type=Path, default=OUT / "shards")
    args = ap.parse_args()

    recs = json.load(open(args.manifest))
    if args.limit:
        recs = recs[: args.limit]
    n_shards = max(1, -(-len(recs) // args.shard_size))

    shards = [[] for _ in range(n_shards)]
    for r in recs:
        shards[stable_shard(r["video_id"], n_shards)].append({"video_id": r["video_id"]})

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.json"):
        old.unlink()
    sizes = []
    for i, s in enumerate(shards):
        json.dump(s, open(args.out / f"{i:04d}.json", "w"), indent=1)
        sizes.append(len(s))

    labs = Counter(r["video_id"].split("+")[0] for r in recs)
    print(f"[shards] {len(recs)} clips -> {n_shards} shards of "
          f"{min(sizes)}-{max(sizes)} (target {args.shard_size})")
    print(f"[shards] array range: --array=0-{n_shards - 1}")
    print(f"\nlab distribution ({len(labs)} labs):")
    for lab, n in labs.most_common(12):
        print(f"  {lab:24s} {n:6d}  {n / len(recs) * 100:5.1f}%")
    if len(labs) > 12:
        print(f"  ... and {len(labs) - 12} more")
    json.dump({"n_shards": n_shards, "n_clips": len(recs), "shard_size": args.shard_size,
               "labs": dict(labs)}, open(args.out / "index.json", "w"), indent=1)
    print(f"\n[shards] -> {args.out}")


if __name__ == "__main__":
    main()
