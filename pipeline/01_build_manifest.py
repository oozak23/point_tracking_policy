#!/usr/bin/env python3
"""Step 1 -- join the release annotations to the frozen raw mirror and report coverage.

Answers: how many of the 23,532 clips actually have a resolvable source MP4 and a
trajectory.h5? Only one lab (AUTOLab) had ever been spot-checked.

Writes out/manifest.json (all resolvable clips) and prints per-lab coverage.

    python pipeline/01_build_manifest.py [--check-decode 8]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.index import SUBSAMPLE, load_clips, resolvable

from droid_qp.paths import DATA  # noqa: E402
OUT = DATA  # the dataset root: droid_qp/paths.py
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-decode", type=int, default=0,
                    help="Decode N random resolvable clips and assert src_frames ~= 4*num_frames")
    ap.add_argument("--out", type=Path, default=OUT / "manifest.json")
    args = ap.parse_args()

    recs = load_clips()
    print(f"[manifest] {len(recs)} clips in droid_clips.json with a source-path entry")

    per_lab = collections.defaultdict(lambda: collections.Counter())
    good = []
    for rec in recs:
        r = resolvable(rec)
        c = per_lab[rec.lab]
        c["total"] += 1
        c["mp4"] += r["mp4"]
        c["h5"] += r["h5"]
        if r["mp4"]:
            c["usable"] += 1
            good.append((rec, r))

    print(f"\n{'lab':12s} {'total':>7s} {'mp4':>7s} {'h5':>7s} {'mp4%':>6s}")
    for lab in sorted(per_lab):
        c = per_lab[lab]
        print(f"{lab:12s} {c['total']:7d} {c['mp4']:7d} {c['h5']:7d} "
              f"{100*c['mp4']/c['total']:5.1f}%")
    tot = sum(c["total"] for c in per_lab.values())
    tm = sum(c["mp4"] for c in per_lab.values())
    th = sum(c["h5"] for c in per_lab.values())
    print(f"{'ALL':12s} {tot:7d} {tm:7d} {th:7d} {100*tm/tot:5.1f}%")

    cats = collections.Counter(rec.category for rec, _ in good)
    print("\ncategories among resolvable clips:")
    for k, v in cats.most_common():
        print(f"  {v:6d}  {k}")

    if args.check_decode:
        import random

        import cv2

        random.seed(0)
        print(f"\n[verify] decoding {args.check_decode} clips: src_frames vs {SUBSAMPLE}*num_frames")
        bad = 0
        for rec, _ in random.sample(good, min(args.check_decode, len(good))):
            cap = cv2.VideoCapture(str(rec.source_mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            ratio = n / max(rec.num_frames, 1)
            flag = "" if 3.8 <= ratio <= 4.2 else "  <-- VIOLATION"
            bad += bool(flag)
            print(f"  {rec.video_id:52s} nf={rec.num_frames:4d} src={n:5d} ratio={ratio:.2f}{flag}")
        print(f"[verify] {bad} violation(s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "video_id": rec.video_id, "cam": rec.cam, "lab": rec.lab,
            "num_frames": rec.num_frames, "caption": rec.caption,
            "category": rec.category, "motion_ranges": rec.motion_ranges,
            "source_mp4": str(rec.source_mp4), "trajectory_h5": str(rec.trajectory_h5),
            "has_h5": r["h5"],
        }
        for rec, r in good
    ]
    json.dump(payload, open(args.out, "w"))
    print(f"\n[manifest] wrote {len(payload)} entries -> {args.out}")


if __name__ == "__main__":
    main()
