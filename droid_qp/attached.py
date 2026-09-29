"""Locate the points that provably move with the gripper, in 2D.

Shared by the scorer (which uses it to judge) and the pipeline (which uses it to
correct). Deliberately depends only on h5py + numpy -- no pytorch_kinematics -- so it
can run inside the molmo-motion env next to MolmoPoint and SAM. FK is only needed for
fingertip positions, which this does not use.

Validated on TRI+52ca9b6a+2024-01-29-13h-27m-52s__25947356: MolmoPoint seeded SAM on a
book lying in a basket's mouth while the robot was manipulating the basket. This
returned 1,842 points covering the basket, and the mask's recall against them was 2.8%
-- the metric caught, unprompted, an error a human had spotted by eye.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geom import (attached_mask, carrying_span, find_window, project,
                   release_to_pw_step, rigid_residuals, window_bounds)

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")

MIN_ATTACHED = 20


def attached_cluster_2d(video_id: str, release_frame: int,
                        out_size: tuple[int, int] = (854, 480),
                        r_max: float = 0.03, s_frac: float = 0.3,
                        pw_flows: Path = PW_FLOWS) -> dict:
    """-> {'ok': bool, 'reason': str, 'uv': (M,2), 'centroid': (2,), 'medoid': (2,), ...}

    `medoid` -- the attached point closest to the cluster's own centre -- is what to
    seed SAM with. The mean can land off the object when the cluster is L-shaped or
    split across a handle, in the same way a k-means centre can fall outside a
    non-convex mask.
    """
    import h5py

    out = {"ok": False, "reason": "", "n_attached": 0}
    episode, cam = video_id.rsplit("__", 1)
    path = Path(pw_flows) / f"{episode}_flows.h5"
    if not path.exists():
        out["reason"] = "no_pointworld_episode"
        return out

    step = release_to_pw_step(release_frame)
    try:
        with h5py.File(path, "r") as f:
            wins = sorted((k for k in f.keys() if ":" in k),
                          key=lambda k: window_bounds(k)[0])
            w = find_window(wins, step)
            if w is None:
                out["reason"] = "frame_outside_pointworld_windows"
                return out
            cam_key = f"camera_{cam}_ext"
            if cam_key not in f[w]:
                out["reason"] = "camera_not_in_window"
                return out
            g = f[w][cam_key]
            sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
            vis = np.asarray(g["scene_visibility"][:])
            dvm = np.asarray(g["scene_depth_valid_mask"][:])
            E = np.asarray(g["extrinsic"][:], dtype=np.float64)
            K = np.asarray(g["intrinsic"][:], dtype=np.float64)
            pose = np.asarray(f[w]["gripper_pose"][:], dtype=np.float64)
            gopen = np.asarray(f[w]["gripper_open"][:]).ravel().astype(bool)
    except Exception as e:
        out["reason"] = f"read_error:{type(e).__name__}"
        return out

    a, _ = window_bounds(w)
    local = int(np.clip(step - a, 0, sf.shape[0] - 1))
    span = carrying_span(gopen, ref=local)
    if span is None:
        out["reason"] = "gripper_never_closed_in_window"
        return out

    r, s, grip = rigid_residuals(sf, pose, ref=local, span=span)
    if grip < 0.03:
        out["reason"] = "gripper_static_undefined"
        return out

    # Same clipping as scripts/05_score_points.py, so the corrector and the scorer
    # evaluate the identical frame -- otherwise the pipeline could "fix" a mask against
    # one frame and be graded against another.
    local = int(np.clip(local, span[0], span[1]))
    good = vis[local] & dvm[local]
    att = attached_mask(r, s, grip, r_max=r_max, s_frac=s_frac) & good

    uv, z = project(sf[local], E, K, out_size=out_size)
    W, H = out_size
    keep = att & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    n = int(keep.sum())
    out["n_attached"] = n
    out["grip_disp_m"] = float(grip)
    out["window"] = w
    if n < MIN_ATTACHED:
        out["reason"] = "too_few_attached_points"
        return out

    pts = uv[keep]
    c = pts.mean(axis=0)
    medoid = pts[int(np.argmin(((pts - c) ** 2).sum(axis=1)))]
    out.update({"ok": True, "uv": pts, "centroid": c, "medoid": medoid})
    return out


def mask_recall(mask_bool: np.ndarray, uv: np.ndarray) -> float:
    """Fraction of the attached cluster lying inside a mask."""
    if uv is None or len(uv) == 0:
        return 0.0
    h, w = mask_bool.shape
    x = np.clip(uv[:, 0].round().astype(int), 0, w - 1)
    y = np.clip(uv[:, 1].round().astype(int), 0, h - 1)
    return float(mask_bool.astype(bool)[y, x].mean())
