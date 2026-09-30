#!/usr/bin/env python3
"""The bar for the future-point head: trivial predictions of where robot points will be.

On the frozen validation windows (same frames as every other result), for each tracked robot
point and horizon j = 1..16, the target is its FK displacement from t to t+j (both ends inside the
image, t+j a stored frame). Three baselines:

    zero            the point does not move
    last_tracked    the point repeats its last tracked motion, j * (track[t] - track[t-1]);
                    the same information the model gets. No motion if either track is hidden.
    last_fk         the same with FK positions: idealised, the model does not see these

Reported per horizon: mean and median error in pixels, and the normalised loss
mean(|error|^2) / scale_j^2, the value the point loss would give (pipeline/85_... scales).

    $PY_POINTWORLD -B evaluation/86_point_baselines.py --out $PTP_OUT/point_baselines
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402

H, WIN = 16, 48
NAMES = ("zero", "last_tracked", "last_fk")
_W = {}


def _init(palm):
    _W["palm"] = palm
    _W["shards"] = json.loads((DATA / "shard_index.json").read_text())


def clip_errors(rec):
    vid, start = rec["video_id"], rec["start"]
    base = DATA / "gen" / _W["shards"][vid]
    (mg,) = sorted((base / "merged/query_points").glob(f"{vid}_f*.npz"))
    with np.load(mg) as a:
        ids = a["point_index"]
        fk, ok = a["fk_uv"][:, ids].astype(np.float64), a["fk_ok"][:, ids]
    with np.load(base / "trackon/tracks" / f"{vid}.npz") as t:
        cols = np.flatnonzero(t["role"] == 1)
        trk, vis = t["tracks"][:, cols].astype(np.float64), t["visibility"][:, cols]
    n = min(len(fk), len(trk))
    palm = _W["palm"][ids]
    rows = []                       # (j, is_palm, err_zero, err_last_tracked, err_last_fk, tracked_fallback)
    for t0 in range(max(start, 1), min(start + WIN, n)):
        v_trk = trk[t0] - trk[t0 - 1]
        fb = ~(vis[t0] & vis[t0 - 1])
        v_trk[fb] = 0.0
        v_fk = fk[t0] - fk[t0 - 1]
        v_fk[~(ok[t0] & ok[t0 - 1])] = 0.0
        for j in range(1, H + 1):
            if t0 + j >= n:
                break
            m = ok[t0] & ok[t0 + j]
            d = (fk[t0 + j] - fk[t0])[m]
            e = [np.linalg.norm(d, axis=1),
                 np.linalg.norm(d - j * v_trk[m], axis=1),
                 np.linalg.norm(d - j * v_fk[m], axis=1)]
            rows.append(np.column_stack([np.full(m.sum(), j), palm[m], *e, fb[m]]))
    return np.concatenate(rows) if rows else np.zeros((0, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=DATA / "validation_plan_v1.json")
    ap.add_argument("--scales", type=Path, default=DATA / "point_loss_scales.json")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    from droid_qp.pose_fit import gripper_model
    from droid_qp.robot_points import load_sampler
    _, palm = gripper_model(load_sampler(128, gripper_only=True, seed=0))
    recs = json.loads(args.plan.read_text())["records"]
    scale = np.array(json.loads(args.scales.read_text())["scale_px"])
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(palm,)) as pool:
        A = np.concatenate(list(pool.imap_unordered(clip_errors, recs, chunksize=8)))
    res = {"windows": len(recs), "horizons": {}}
    for j in range(1, H + 1):
        a = A[A[:, 0] == j]
        r = {"n": int(len(a)), "tracked_fallback_share": float(a[:, 5].mean())}
        for i, name in enumerate(NAMES):
            e = a[:, 2 + i]
            r[name] = {"mean_px": float(e.mean()), "median_px": float(np.median(e)),
                       "norm_loss": float((e ** 2).mean() / scale[j - 1] ** 2),
                       "palm_mean_px": float(e[a[:, 1] == 1].mean()),
                       "finger_mean_px": float(e[a[:, 1] == 0].mean())}
        res["horizons"][j] = r
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"{'j':>3s} {'n':>9s} | " + " | ".join(f"{nm:>22s}" for nm in NAMES))
    for j in (1, 2, 4, 8, 12, 16):
        r = res["horizons"][j]
        print(f"{j:3d} {r['n']:9d} | " + " | ".join(
            f"{r[nm]['mean_px']:7.2f}px  loss {r[nm]['norm_loss']:5.2f}" for nm in NAMES))
    print("->", args.out)


if __name__ == "__main__":
    main()
