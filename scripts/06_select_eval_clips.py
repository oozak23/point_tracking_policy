#!/usr/bin/env python3
"""Pick the fixed evaluation set both approaches are judged on.

Constraints, so that every selected clip is actually scorable:
  - in the MolmoMotion/PointWorld overlap (the scorer needs PointWorld's dense points)
  - decent PointWorld timeline coverage (median is 0.76 but the tail goes to 0.03)
  - has a motion range, since approach 1 needs one to trim to
  - has at least one window with real gripper displacement, or the rigid-motion test
    is undefined there
  - spread over categories, so the set is not 40 variations of "Move object"

Writes a list in the same shape as the other reports so `--video-ids-from` accepts it.

    python scripts/06_select_eval_clips.py --n 40
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--min-grip-motion", type=float, default=0.05,
                    help="metres of EE displacement in at least one window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--natural", action="store_true",
                    help="sample by NATURAL category frequency instead of round-robin. "
                         "Round-robin is a diagnostic set -- it under-samples 'Move "
                         "object' (40.6%% of data) 5x and over-samples 'Push/Pull' 25x, "
                         "so its headline score is pessimistic and does not predict "
                         "performance on the real 14,083. Use --natural for that number "
                         "and the stratified set to find WHICH categories fail.")
    ap.add_argument("--manifest", type=Path, default=OUT / "overlap_manifest.json")
    ap.add_argument("--out", type=Path, default=OUT / "eval_clips.json")
    args = ap.parse_args()

    rows = json.load(open(args.manifest))
    print(f"[select] overlap manifest: {len(rows)} videos")

    def usable(r):
        if r.get("coverage", 0) < args.min_coverage:
            return False
        if not r.get("mm_motion_ranges"):
            return False
        mags = r.get("ee_pos_motion", {}) or {}
        return any(v >= args.min_grip_motion for v in mags.values())

    cand = [r for r in rows if usable(r)]
    print(f"[select] usable after filters: {len(cand)}")

    by_cat = collections.defaultdict(list)
    for r in cand:
        by_cat[r["category"]].append(r)
    print(f"[select] categories available: {len(by_cat)}")

    random.seed(args.seed)
    if args.natural:
        chosen = random.sample(cand, min(args.n, len(cand)))
    else:
        # Round-robin across categories so no single one dominates.
        for v in by_cat.values():
            random.shuffle(v)
        cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
        chosen, i = [], 0
        while len(chosen) < args.n and any(by_cat[c] for c in cats):
            c = cats[i % len(cats)]
            if by_cat[c]:
                chosen.append(by_cat[c].pop())
            i += 1

    picked = collections.Counter(r["category"] for r in chosen)
    print(f"\n[select] {len(chosen)} clips:")
    for k, v in picked.most_common():
        print(f"  {v:3d}  {k}")

    payload = [{"video_id": r["video_id"], "episode": r["episode"], "cam": r["cam"],
                "category": r["category"], "caption": r["caption"],
                "coverage": r["coverage"], "pw_windows": r["pw_windows"],
                "mm_motion_ranges": r["mm_motion_ranges"]}
               for r in chosen]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=1)
    print(f"\n[select] wrote {args.out}")


if __name__ == "__main__":
    main()
