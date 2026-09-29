#!/usr/bin/env python3
"""Step 2 -- run Qwen3-0.6B noun extraction over the DROID caption vocabulary.

DROID has only 454 *unique* captions across 23,532 clips, so this is keyed by caption
text rather than by clip: ~454 Qwen3 calls instead of 23,532. The result is cached to
out/caption_nouns.json and reused by every later step.

Also reports how many clips fall on each side of the vague/non-vague split, which
decides what can run on the 16 GB login GPU (non-vague) versus what needs Molmo2-8B
re-captioning on SLURM (vague).

    python scripts/02_extract_nouns.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp import grounding
from droid_qp.index import load_clips

OUT = Path(__file__).resolve().parent.parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the N most common captions")
    ap.add_argument("--out", type=Path, default=OUT / "caption_nouns.json")
    args = ap.parse_args()

    recs = load_clips()
    counts = collections.Counter(r.caption for r in recs)
    captions = [c for c, _ in counts.most_common(args.limit)]
    print(f"[nouns] {len(recs)} clips, {len(counts)} unique captions; processing {len(captions)}")

    cache = json.load(open(args.out)) if args.out.exists() else {}
    todo = [c for c in captions if c not in cache]
    print(f"[nouns] {len(cache)} cached, {len(todo)} to extract")

    if todo:
        model, tok = grounding.load_noun_extractor()
        for i, cap in enumerate(todo, 1):
            obj = grounding.extract_noun(model, tok, cap)
            cache[cap] = {
                "noun": obj,
                "vague": grounding.is_vague(obj) or not obj,
                "hallucinated": grounding.is_hallucinated(obj, cap),
                "needs_recaption": grounding.needs_recaption(obj, cap),
            }
            if i % 20 == 0 or i == len(todo):
                print(f"  [nouns] {i}/{len(todo)}")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(cache, open(args.out, "w"), indent=1)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(args.out, "w"), indent=1)

    n_direct = sum(counts[c] for c in cache if not cache[c]["vague"] and c in counts)
    n_vague = sum(counts[c] for c in cache if cache[c]["vague"] and c in counts)
    tot = n_direct + n_vague
    print(f"\n[nouns] clip-weighted split over {tot} clips:")
    print(f"  non-vague (login GPU, no Molmo2-8B) : {n_direct:6d}  {100*n_direct/max(tot,1):5.1f}%")
    print(f"  vague     (needs Molmo2-8B, SLURM)  : {n_vague:6d}  {100*n_vague/max(tot,1):5.1f}%")

    print("\n[nouns] most common non-vague captions -> noun:")
    shown = 0
    for cap, n in counts.most_common():
        e = cache.get(cap)
        if e and not e["vague"]:
            print(f"  {n:6d}  {cap[:62]:62s} -> {e['noun']!r}")
            shown += 1
            if shown >= 20:
                break
    print(f"\n[nouns] wrote {args.out}")


if __name__ == "__main__":
    main()
