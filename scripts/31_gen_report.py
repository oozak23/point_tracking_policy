#!/usr/bin/env python3
"""Which shards finished, which failed, and what to re-submit.

A 141-task array will not complete cleanly on the first attempt, and the useful output is
not "how many succeeded" but the exact --array expression to rerun. Prints one, compressed
into ranges, so re-submission is copy-paste rather than arithmetic.

Also totals the artifacts, because a shard can exit 0 having processed far fewer clips
than it was given -- a clip is dropped whenever grounding, FK coverage or the SAM filter
rejects it, and those are silent by design. Comparing clips-out against clips-in is the
only check that catches a systematic drop.

    python scripts/31_gen_report.py
Run with any env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"


def compress(ids: list[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8', the form sbatch --array wants."""
    if not ids:
        return ""
    ids = sorted(ids)
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=OUT / "gen")
    ap.add_argument("--shards", type=Path, default=OUT / "shards")
    args = ap.parse_args()

    idx = json.load(open(args.shards / "index.json"))
    n_shards = idx["n_shards"]

    done, missing, partial = [], [], []
    clips_in = clips_out = n_tokens = 0
    for i in range(n_shards):
        s = f"{i:04d}"
        want = len(json.load(open(args.shards / f"{s}.json")))
        clips_in += want
        d = args.gen / s
        if (d / "DONE").exists():
            got = len(list((d / "trackon" / "tracks").glob("*.npz"))) \
                if (d / "trackon" / "tracks").is_dir() else 0
            clips_out += got
            n_tokens += sum(p.stat().st_size for p in
                            (d / "trackon" / "tracks").glob("*.npz")) \
                if (d / "trackon" / "tracks").is_dir() else 0
            done.append(i)
            if got < 0.5 * want:
                partial.append((i, got, want))
        else:
            missing.append(i)

    print(f"shards   : {len(done)}/{n_shards} complete")
    print(f"clips    : {clips_out} of {clips_in} attempted "
          f"({clips_out / max(clips_in, 1) * 100:.1f}% yield)")
    print(f"artifacts: {n_tokens / 1e9:.1f} GB of token npz")

    if partial:
        print(f"\n{len(partial)} shard(s) finished but kept under half their clips --")
        print("  a systematic drop looks like this, not like a crash:")
        for i, got, want in partial[:10]:
            print(f"    shard {i:04d}: {got}/{want}")

    if missing:
        print(f"\n{len(missing)} shard(s) incomplete. Re-submit with:")
        print(f"  sbatch --array={compress(missing)}%16 scripts/29_gen_shard.sbatch")
    else:
        print("\nall shards complete")


if __name__ == "__main__":
    main()
