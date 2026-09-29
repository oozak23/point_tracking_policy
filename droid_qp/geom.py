"""Geometry over PointWorld: projection, and the rigid-motion test.

Two calibrated facts, measured by scripts/00_calibrate_gripper.py over 79 windows
(see out/gripper_calibration.json) -- do not re-derive them, and do not guess:

  * `gripper_pose[:, 3:7]` is a quaternion in **XYZW** order.
    Measured: 0.00 deg median geodesic error vs URDF FK, against 166.31 deg for WXYZ.
    Reading it as WXYZ still produces plausible-looking numbers, so this is silent.

  * `gripper_pose[:, :3]` is **`panda_link8`** -- the Franka *flange*, not the fingertips.
    Measured: 0.0000 m error, unanimous over all 79 windows. The Robotiq 2F-85 extends
    past it: outer_knuckle 0.070 m, outer_finger 0.088 m, **inner_finger 0.125 m**.
    A grasped object therefore sits ~0.13-0.17 m from `gripper_pose`, which is exactly
    the gap measured to the nearest dense scene point. Any "near the gripper" test must
    use the inner_finger links from FK, never `gripper_pose`.
"""
from __future__ import annotations

import numpy as np

from .rates import release_to_pw_step as _release_to_pw_step

# The scene point cloud is stored against a 320x180 intrinsic.
PW_W, PW_H = 320, 180


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(...,4) unit quaternion in XYZW order -> (...,3,3) rotation matrices."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def project(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
            out_size: tuple[int, int] | None = None):
    """World/robot-base 3D -> pixels. `extrinsic` is world->camera (verified).

    Returns (uv (...,2), z (...,)). Points with z <= 0 are behind the camera; callers
    must mask them. With `out_size=(W,H)` the result is rescaled from 320x180.
    """
    P = np.asarray(points_world, dtype=np.float64)
    E = np.asarray(extrinsic, dtype=np.float64)
    K = np.asarray(intrinsic, dtype=np.float64)

    cam = P @ E[:3, :3].T + E[:3, 3]
    z = cam[..., 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[..., 0] / safe + K[0, 2]
    v = K[1, 1] * cam[..., 1] / safe + K[1, 2]

    if out_size is not None:
        u = u * (out_size[0] / PW_W)
        v = v * (out_size[1] / PW_H)
    return np.stack([u, v], axis=-1), z


def carrying_span(gripper_open: np.ndarray, ref: int | None = None) -> tuple[int, int] | None:
    """Longest run of CLOSED frames in a window -> (start, end) inclusive.

    Residuals must be measured over this span, not the whole window. A window that
    straddles the grasp spends its first half with the object unattached, and since the
    residual is a median over frames, that half dominates and the object looks static.
    Measuring over the whole window is what produced a spurious "geometric coverage is
    43%" -- the same data gives ~87% when restricted to the carrying span (median 108
    attached points, zero in only 13% of carrying windows).

    With `ref` given, prefer a run containing it. Returns None if never closed.
    """
    closed = ~np.asarray(gripper_open).ravel().astype(bool)
    if not closed.any():
        return None
    runs, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(closed) - 1))
    if ref is not None:
        containing = [r for r in runs if r[0] <= ref <= r[1]]
        if containing:
            return max(containing, key=lambda r: r[1] - r[0])
    return max(runs, key=lambda r: r[1] - r[0])


def rigid_residuals(scene_flows: np.ndarray, gripper_pose: np.ndarray,
                    ref: int = 0, span: tuple[int, int] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Test each dense point against "am I rigidly attached to the gripper?".

    scene_flows  (T, N, 3) world-frame tracks
    gripper_pose (T, 7)    xyz + XYZW quaternion

    Returns (r, s, grip_disp):
      r (N,)  median over t of || X_i(t) - predicted_if_attached ||, metres
      s (N,)  median over t of || X_i(t) - X_i(ref) ||, i.e. did it move at all
      grip_disp  how far the gripper travelled -- the test is only discriminative
                 when this is large, since with a still gripper every point has a
                 small residual and r carries no information.

    A point on the held object has small r and large s. A point on the table has
    s ~= 0 and r ~= grip_disp. A point on some other moving thing has both large.
    """
    X = np.asarray(scene_flows, dtype=np.float64)
    G = np.asarray(gripper_pose, dtype=np.float64)

    if span is not None:
        a, b = span
        a = int(np.clip(a, 0, X.shape[0] - 1))
        b = int(np.clip(b, a, X.shape[0] - 1))
        ref = int(np.clip(ref, a, b)) - a
        X, G = X[a:b + 1], G[a:b + 1]
    if X.shape[0] < 2:
        n = X.shape[1]
        return np.full(n, np.inf), np.zeros(n), 0.0

    R = quat_to_R(G[:, 3:7])          # (T,3,3)
    p = G[:, :3]                      # (T,3)
    ref = int(np.clip(ref, 0, len(p) - 1))
    R0, p0 = R[ref], p[ref]

    # Relative gripper motion since `ref`: R_rel = R_t R_0^T, p_rel = p_t - R_rel p_0
    R_rel = np.einsum("tij,kj->tik", R, R0)
    p_rel = p - np.einsum("tij,j->ti", R_rel, p0)

    pred = np.einsum("tij,nj->tni", R_rel, X[ref]) + p_rel[:, None, :]
    r = np.median(np.linalg.norm(X - pred, axis=-1), axis=0)
    s = np.median(np.linalg.norm(X - X[ref][None], axis=-1), axis=0)

    grip_disp = float(np.max(np.linalg.norm(p - p0, axis=1)))
    return r, s, grip_disp


def attached_mask(r: np.ndarray, s: np.ndarray, grip_disp: float,
                  r_max: float = 0.02, s_frac: float = 0.3) -> np.ndarray:
    """Points that moved AND moved like the gripper.

    Both conditions are needed. Without the `s` condition a point that never moves
    scores a small residual whenever the gripper also barely moved, so the test would
    silently pass on static scenes -- the exact failure mode we are trying to detect.
    """
    return (r < r_max) & (s > max(r_max, s_frac * grip_disp))


def window_bounds(key: str) -> tuple[int, int]:
    """'100:111' -> (100, 111); the window covers PointWorld steps 100..110."""
    a, b = key.split(":")
    return int(a), int(b)


def release_to_pw_step(release_frame: int) -> int:
    """Release frame -> PointWorld step. Re-exported from `rates` so the subsample factor
    lives in exactly one place; kept here under its original name because
    `05_score_points.py`, `09_diag_gripper_at_detection.py`, `12_viz_attached.py` and
    `attached.py` all import it from `geom`.

    The factor is `rates.RATIO` -- 1 at 7.2 Hz (release space == PointWorld space), 2 at
    the old 3.6 Hz. Verified over 12 random episodes: trajectory/pw_steps = 2.00,
    pw_steps/release_frames = 1.99 under SUBSAMPLE=4.
    """
    return int(_release_to_pw_step(release_frame))


def find_window(windows: list[str], pw_step: int) -> str | None:
    """Window containing pw_step. Windows overlap (11 frames, stride 5), so prefer
    the one where the step sits most centrally -- that leaves the most frames on both
    sides for the residual to be measured over."""
    best, best_d = None, None
    for w in windows:
        a, b = window_bounds(w)
        if a <= pw_step < b:
            d = abs(pw_step - (a + b - 1) / 2.0)
            if best_d is None or d < best_d:
                best, best_d = w, d
    return best
