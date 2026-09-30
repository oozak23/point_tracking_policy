#!/usr/bin/env python3
"""Per-horizon scales for the future-point loss (docs/POINT_POLICY_PLAN.md).

scale_j = RMS length, in pixels, of the FK displacement of a tracked robot point from frame t to
t+j, over all TRAINING clips, weighted by clip quality weight (as policy_loss_scales does for the
action). Dividing horizon j's squared error by scale_j^2 makes "predict no motion" cost ~1 at every
horizon, so far horizons, where points move ~15x further, do not dominate the loss.

A displacement counts only if the point is inside the image at both t and t+j, and t+j is a
stored frame. Validation labels are never read.

    $PY_POINTWORLD -B pipeline/85_point_loss_scales.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402

H = 16
_W = {}


def _init(palm):
    _W["palm"] = palm
    _W["shards"] = json.loads((DATA / "shard_index.json").read_text())


def clip_sums(args):
    vid, w = args
    (mg,) = sorted((DATA / "gen" / _W["shards"][vid] / "merged/query_points").glob(f"{vid}_f*.npz"))
    with np.load(mg) as a:
        ids = a["point_index"]
        fk, ok = a["fk_uv"][:, ids].astype(np.float64), a["fk_ok"][:, ids]
    palm = _W["palm"][ids]
    out = np.zeros((H, 3, 2))            # horizon x (all, palm, finger) x (weighted sum sq, weight)
    for j in range(1, H + 1):
        m = ok[j:] & ok[:-j]
        sq = ((fk[j:] - fk[:-j]) ** 2).sum(-1)
        for g, sel in enumerate((np.ones_like(palm), palm, ~palm)):
            mm = m & sel[None, :]
            out[j - 1, g] = (w * sq[mm].sum(), w * mm.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DATA / "point_loss_scales.json")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    from droid_qp.pose_fit import gripper_model
    from droid_qp.robot_points import load_sampler
    _, palm = gripper_model(load_sampler(128, gripper_only=True, seed=0))
    raw = (DATA / "dataset_manifest.json").read_bytes()
    train = [(c["video_id"], float(c["weight"])) for c in json.loads(raw)["clips"] if c["split"] == "train"]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(palm,)) as pool:
        tot = sum(pool.imap_unordered(clip_sums, train, chunksize=16))
    rms = lambda g: [float(np.sqrt(tot[j, g, 0] / tot[j, g, 1])) for j in range(H)]  # noqa: E731
    res = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train", "n_clips": len(train),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "horizons": list(range(1, H + 1)),
        "scale_px": rms(0), "scale_px_palm": rms(1), "scale_px_finger": rms(2),
        "weighted_point_frames": [float(tot[j, 0, 1]) for j in range(H)],
        "formula": "scale_j = sqrt(sum(w*||fk[t+j]-fk[t]||^2) / sum(w)), both ends inside the image",
        "units": "pixels, 854x480 frames",
    }
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print("scale_px by horizon:", " ".join(f"{s:.1f}" for s in res["scale_px"]))
    print("->", args.out)


if __name__ == "__main__":
    main()
