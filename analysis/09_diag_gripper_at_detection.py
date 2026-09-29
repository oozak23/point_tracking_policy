#!/usr/bin/env python3
"""Was the gripper actually holding anything at the detection frame?

21 of 40 clips in the approach-1 re-run scored `too_few_attached_points`, many with
zero. Two very different causes:

  (a) nothing was grasped yet at that frame  -> a flaw in WHEN we ground
  (b) the object was held but did not reconstruct in the 320x180 depth  -> a limit of
      the geometric signal

`gripper_open` at that exact frame separates them. This matters because (a) is fixable
by grounding at the grasp instead of the motion-window start, while (b) would mean the
geometric approach cannot see small objects at all.

    python analysis/09_diag_gripper_at_detection.py --score out/score_a1_mw40.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.geom import find_window, release_to_pw_step, window_bounds

PW = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
          "/droid/flows-fs-optimized")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, required=True)
    args = ap.parse_args()

    d = json.load(open(args.score))
    rows = []
    for r in d["rows"]:
        if r.get("status") not in ("too_few_attached_points", "ok"):
            continue
        ep, f = r["episode"], r["detection_frame"]
        step = release_to_pw_step(f)
        try:
            with h5py.File(PW / f"{ep}_flows.h5", "r") as h:
                wins = sorted((k for k in h.keys() if ":" in k),
                              key=lambda k: window_bounds(k)[0])
                w = find_window(wins, step)
                if w is None:
                    continue
                a, _ = window_bounds(w)
                go = np.asarray(h[w]["gripper_open"][:]).ravel().astype(bool)
                li = int(np.clip(step - a, 0, len(go) - 1))
                # Does the gripper ever close later in the episode?
                closes_later = False
                for w2 in wins:
                    a2, _ = window_bounds(w2)
                    if a2 < a:
                        continue
                    g2 = np.asarray(h[w2]["gripper_open"][:]).ravel().astype(bool)
                    if (~g2).any():
                        closes_later = True
                        break
                rows.append({
                    "video_id": r["video_id"], "status": r["status"],
                    "frame": f, "open_at_detection": bool(go[li]),
                    "n_attached": int(r.get("n_attached_visible", 0)),
                    "closes_later": closes_later,
                })
        except Exception as e:
            print(f"  [skip] {ep}: {type(e).__name__}: {e}")

    print(f"clips checked: {len(rows)}")
    for st in ("ok", "too_few_attached_points"):
        sub = [x for x in rows if x["status"] == st]
        if not sub:
            continue
        op = sum(x["open_at_detection"] for x in sub)
        cl = sum(x["closes_later"] for x in sub if x["open_at_detection"])
        print(f"\n{st}: {len(sub)} clips")
        print(f"  gripper OPEN (holding nothing) at detection frame: "
              f"{op}/{len(sub)} = {100*op/len(sub):.0f}%")
        if op:
            print(f"  ...of those, the gripper DOES close later in the episode: "
                  f"{cl}/{op} = {100*cl/op:.0f}%")
            print(f"     -> those are fixable by grounding at the grasp instead.")

    bad = [x for x in rows
           if x["status"] == "too_few_attached_points" and not x["open_at_detection"]]
    if bad:
        print(f"\ngripper CLOSED but still no attached points: {len(bad)} clips")
        print("  (held something that did not reconstruct, or closed on nothing)")
        for x in bad[:8]:
            print(f"    {x['video_id'][:52]:52s} f{x['frame']:<4d} n_att={x['n_attached']}")


if __name__ == "__main__":
    main()
