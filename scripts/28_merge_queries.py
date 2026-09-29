#!/usr/bin/env python3
"""Combine object and robot query points into ONE Track-On query tensor.

Track-On takes queries as (N,3) rows (t, x, y) and births each when `query_times == t`
(trackon_predictor.py:193-275), so points with different birth frames coexist natively in
a single pass. Object points are born at frame 0 (frame-0 seeding); robot points at their
first step with >= --min-visible FK points in frame. One pass rather than two costs
nothing extra and yields the FK drift measurement for free.

A `role` array rides along (0 = object, 1 = robot). It is NOT a stored artifact of the
dataset -- downstream the stream is split into `scene_tokens` and `robot_tokens`, two
files, so the role is implicit in which one you read. It exists only to carry the split
through this pass.

    python scripts/28_merge_queries.py --obj out/gen/0000/obj --robot out/gen/0000/robot_sam
Run with any env that has numpy.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out"
NAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)\.npz$")

ROLE_OBJECT, ROLE_ROBOT = 0, 1


def index_dir(d: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(d.glob("*.npz")):
        m = NAME_RE.match(p.name)
        if m:
            out[m["vid"]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=Path, required=True, help="object run dir")
    ap.add_argument("--robot", type=Path, required=True, help="robot SAM-filtered run dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-robot", action="store_true",
                    help="skip clips with no robot points instead of keeping object-only")
    args = ap.parse_args()

    obj = index_dir(args.obj / "query_points")
    rob = index_dir(args.robot / "query_points")
    qdir = args.out / "query_points"
    qdir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] object {len(obj)} | robot {len(rob)} clips")

    report, n_both, n_obj_only = [], 0, 0
    for vid, po in obj.items():
        do = np.load(po)
        qo = do["query_points"].astype(np.int32)
        parts = [qo]
        roles = [np.full(len(qo), ROLE_OBJECT, dtype=np.int8)]
        extra = {}

        pr = rob.get(vid)
        if pr is not None:
            dr = np.load(pr)
            qr = dr["query_points"].astype(np.int32)
            parts.append(qr)
            roles.append(np.full(len(qr), ROLE_ROBOT, dtype=np.int8))
            # FK ground truth travels with the clip: it is what the tracker's drift is
            # measured against, and under the action-decoder design it is also where the
            # robot tokens are sampled from.
            for k in ("fk_uv", "fk_ok", "fk_release_frame", "point_index"):
                if k in dr.files:
                    extra[k] = dr[k]
            n_both += 1
        elif args.require_robot:
            continue
        else:
            n_obj_only += 1

        q = np.concatenate(parts, axis=0)
        role = np.concatenate(roles, axis=0)
        f0 = int(q[:, 0].min())
        np.savez(qdir / f"{vid}_f{f0}.npz", query_points=q, role=role,
                 dim=do["dim"] if "dim" in do.files
                 else np.array([480, 854], dtype=np.int32), **extra)
        report.append({"video_id": vid, "n_object": int(len(qo)),
                       "n_robot": int(len(q) - len(qo)), "n_total": int(len(q))})

    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print(f"[merge] wrote {len(report)} clips | with robot {n_both} | "
          f"object-only {n_obj_only}")
    if report:
        tot = np.array([r["n_total"] for r in report])
        rb = np.array([r["n_robot"] for r in report])
        print(f"[merge] N total: median {np.median(tot):.0f}  "
              f"robot half: median {np.median(rb):.0f}")
    print(f"[merge] -> {qdir}")


if __name__ == "__main__":
    main()
