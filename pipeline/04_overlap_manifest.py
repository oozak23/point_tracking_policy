#!/usr/bin/env python3
"""Step 4 -- join MolmoMotion clips to PointWorld episodes.

PointWorld stores one HDF5 per *episode* (both exterior cameras inside); MolmoMotion
stores one entry per *episode + camera*. Join by stripping the `__<cam>` suffix, then
verify the specific camera MolmoMotion chose actually exists in the PointWorld file
(checked 200/200 on a sample, but verified per row here).

For each matched row we record what PointWorld can supply that DROID's own annotations
cannot:
  - exact intrinsic/extrinsic per window per camera (so 3D->2D needs no fitting;
    measured: 100% of visible, depth-valid points project inside the frame)
  - dense scene_flows (~26,600 points/window) covering table, background and robot
  - joint/gripper state in the same file, for URDF FK robot points
  - canonical_timestamps, the only absolute clock in any of these datasets

Writes out/overlap_manifest.json.

    python pipeline/04_overlap_manifest.py [--limit N] [--jobs 8]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
from droid_qp.paths import DATA, DROID_ANNOTATIONS  # noqa: E402
ANN = DROID_ANNOTATIONS / "droid_clips.json"
OUT = DATA  # the dataset root: droid_qp/paths.py
def episode_of(video_id: str) -> str:
    return video_id.rsplit("__", 1)[0]


def read_episode(ep: str, cams_wanted: set[str]) -> dict | None:
    """Read one PointWorld episode's index. /frozen is read-only: open r, never write."""
    path = PW_FLOWS / f"{ep}_flows.h5"
    try:
        with h5py.File(path, "r") as f:
            wins = sorted((k for k in f.keys() if ":" in k),
                          key=lambda k: int(k.split(":")[0]))
            if not wins:
                return None
            cams = set()
            for w in wins:
                cams |= {k.split("_")[1] for k in f[w].keys() if k.startswith("camera_")}

            mag = f["ee_pos_motion_magnitudes"].attrs
            rot = f["ee_rot_motion_magnitudes"].attrs
            grip = f["has_gripper_movement"].attrs

            ts = np.asarray(eval(f.attrs["canonical_timestamps"]))
            traj_len = int(f.attrs["trajectory_length"])
            span_s = float(ts[-1] - ts[0]) / 1000.0

            covered = set()
            for w in wins:
                s, e = map(int, w.split(":"))
                covered |= set(range(s, e))

            return {
                "episode": ep,
                "pw_cams": sorted(cams),
                "n_windows": len(wins),
                "windows": wins,
                "pw_steps": int(len(ts)),
                "trajectory_length": traj_len,
                "span_s": span_s,
                # PointWorld indexes windows in its own subsampled step space
                # (len(ts) ~= trajectory_length / 2 -> ~7.2 Hz).
                "pw_hz": (len(ts) / span_s) if span_s > 0 else None,
                "traj_hz": (traj_len / span_s) if span_s > 0 else None,
                "coverage": len(covered) / max(len(ts), 1),
                "ee_pos_motion": {w: float(mag[w]) for w in wins if w in mag},
                "ee_rot_motion": {w: float(rot[w]) for w in wins if w in rot},
                "gripper_movement": {w: bool(grip[w]) for w in wins if w in grip},
                "current_task": str(f.attrs.get("current_task", "")),
            }
    except Exception as e:
        return {"episode": ep, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT / "overlap_manifest.json")
    args = ap.parse_args()

    clips = json.load(open(ANN))
    have = {fn[: -len("_flows.h5")] for fn in os.listdir(PW_FLOWS)
            if fn.endswith("_flows.h5")}
    print(f"[overlap] PointWorld episodes: {len(have)}")
    print(f"[overlap] MolmoMotion clips:   {len(clips)}")

    by_ep: dict[str, list] = collections.defaultdict(list)
    for e in clips:
        ep = episode_of(e["file"])
        if ep in have:
            by_ep[ep].append(e)
    eps = sorted(by_ep)
    if args.limit:
        eps = eps[: args.limit]
    print(f"[overlap] overlapping episodes: {len(eps)} "
          f"covering {sum(len(by_ep[e]) for e in eps)} clips")

    rows, errors, cam_missing = [], [], 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(read_episode, ep, {c["cam"] for c in by_ep[ep]}): ep
                for ep in eps}
        for i, fut in enumerate(as_completed(futs), 1):
            info = fut.result()
            ep = futs[fut]
            if info is None or "error" in (info or {}):
                errors.append(info or {"episode": ep, "error": "no windows"})
            else:
                for c in by_ep[ep]:
                    if c["cam"] not in info["pw_cams"]:
                        cam_missing += 1
                        continue
                    rows.append({
                        "video_id": c["file"], "episode": ep, "cam": c["cam"],
                        "caption": c["caption"], "category": c["category"],
                        "mm_num_frames": c["num_frames"],
                        "mm_motion_ranges": [list(r) for rs in
                                             c["clips_by_object"].values() for r in rs],
                        "pw_windows": info["windows"],
                        "pw_n_windows": info["n_windows"],
                        "pw_steps": info["pw_steps"],
                        "trajectory_length": info["trajectory_length"],
                        "span_s": info["span_s"],
                        "pw_hz": info["pw_hz"],
                        "traj_hz": info["traj_hz"],
                        "coverage": info["coverage"],
                        "ee_pos_motion": info["ee_pos_motion"],
                        "gripper_movement": info["gripper_movement"],
                    })
            if i % 500 == 0 or i == len(eps):
                print(f"  {i}/{len(eps)}  rows={len(rows)} errors={len(errors)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"))

    print("\n===== summary =====")
    print(f"matched videos (episode+camera) : {len(rows)}")
    print(f"distinct episodes               : {len({r['episode'] for r in rows})}")
    print(f"camera present in PointWorld    : {len(rows)} matched, {cam_missing} missing")
    print(f"unreadable episodes             : {len(errors)}")
    if rows:
        w = np.array([r["pw_n_windows"] for r in rows])
        cov = np.array([r["coverage"] for r in rows])
        hz = np.array([r["traj_hz"] for r in rows if r["traj_hz"]])
        print(f"windows/video   : mean {w.mean():.1f} median {np.median(w):.0f} "
              f"total {w.sum():,}")
        print(f"timeline coverage: median {np.median(cov):.2f} min {cov.min():.2f}")
        print(f"trajectory rate : median {np.median(hz):.2f} Hz  "
              f"(NOT the 60 fps the release metadata claims)")
        cats = collections.Counter(r["category"] for r in rows)
        print("categories:")
        for k, v in cats.most_common(8):
            print(f"  {v:6d}  {k}")
    for e in errors[:5]:
        print(f"  [ERR] {e}")
    print(f"\n[overlap] wrote {args.out}")


if __name__ == "__main__":
    main()
