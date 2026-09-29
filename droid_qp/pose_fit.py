"""Recover the gripper's pose from its points, and the action from two poses.

The points-to-actions experiment (evaluation/83_points_to_actions.py) asks whether an action
can be read back from where the gripper's points are. Only the 68 palm points are used: they
are rigid with the wrist flange (panda_link8), the frame actions are defined in, while the 60
finger and knuckle points move by up to 58 mm as the gripper opens and closes.

Given the palm's fixed 3D shape in the flange frame (`model`):
    3D points  -> pose by a rigid least-squares fit (Kabsch)
    2D pixels  -> pose by PnP, using the camera intrinsics
Two poses in the same static frame give the body-frame action, exactly as the labels define it
(actions.pose_deltas): dR = R_a^T R_b, dp = R_a^T (p_b - p_a). A static camera cancels out, so
PnP poses need no extrinsics.
"""
from __future__ import annotations

import cv2
import numpy as np

PALM_LINK = "robotiq_85_base_link"
FLANGE_LINK = "panda_link8"
MIN_POINTS = 6          # PnP needs 4 in general position; 6 keeps it stable


def gripper_model(sampler):
    """-> (model (128, 3) in the flange frame, rigid (128,) bool: True for palm points).

    Reads the sampler's own mesh-to-link assignment, in the order `compute_points` writes
    points, so index i here is FK point i everywhere else.
    """
    import torch
    from robot_sampler import build_robotiq_joint_dict
    from .robot_points import PANDA_JOINTS, robot_points_3d

    links = []
    for mesh, pts in sampler._presampled_points.items():
        links += [sampler._mesh_to_link.get(mesh)] * pts.shape[0]
    q = np.array([[0, -0.6, 0, -2.2, 0, 1.6, 0.8]])       # any pose: palm points are rigid
    g = np.zeros((1, 1))
    P = robot_points_3d(sampler, q, g)[0]
    jd = {n: torch.tensor(q[:, i], dtype=sampler.dtype) for i, n in enumerate(PANDA_JOINTS)}
    jd.update(build_robotiq_joint_dict(torch.tensor(g[:, 0], dtype=sampler.dtype), sampler.joint_names))
    T = sampler.fk(jd, link_names=[FLANGE_LINK])[FLANGE_LINK][0].detach().numpy().astype(np.float64)
    model = (P - T[:3, 3]) @ T[:3, :3]                     # R^T (P - t), row-wise
    return model, np.array(links) == PALM_LINK


def kabsch(model: np.ndarray, observed: np.ndarray):
    """Least-squares rigid fit: observed ~= R @ model + t. (n, 3) each -> (R, t)."""
    mc, oc = model.mean(0), observed.mean(0)
    H = (model - mc).T @ (observed - oc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))                 # never return a reflection
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, oc - R @ mc


def pnp(model: np.ndarray, uv: np.ndarray, K: np.ndarray, min_points: int = MIN_POINTS):
    """Pose of the model in the camera frame from its pixels: uv ~ project(R @ model + t).

    SQPnP gives a globally optimal start, Levenberg-Marquardt refines it. Returns
    (R, t), or None when there are too few points, the points are degenerate (e.g. a
    tracker that collapsed them onto one pixel: OpenCV refuses those), or the solve fails.
    """
    if len(model) < min_points:
        return None
    obj = np.ascontiguousarray(model, dtype=np.float64)
    img = np.ascontiguousarray(uv, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    except cv2.error:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def relative(Ra, ta, Rb, tb):
    """Body-frame motion from pose a to pose b (same static frame): (dR, dp)."""
    return Ra.T @ Rb, Ra.T @ (tb - ta)


def repeat(dR, dp, k: int):
    """Apply the same body-frame step k times: the constant-motion (momentum) prediction."""
    R, p = np.eye(3), np.zeros(3)
    for _ in range(k):
        p = p + R @ dp
        R = R @ dR
    return R, p


def rotation_error_deg(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def scaled_intrinsics(K: np.ndarray, out_size=(854, 480), calib_size=(320, 180)) -> np.ndarray:
    """PointWorld's intrinsics are for 320x180 images; our frames are 854x480 (geom.project)."""
    sx, sy = out_size[0] / calib_size[0], out_size[1] / calib_size[1]
    K = np.asarray(K, dtype=np.float64).copy()
    K[0] *= sx
    K[1] *= sy
    return K


def pnp_ransac(model: np.ndarray, uv: np.ndarray, K: np.ndarray, threshold_px: float = 8.0,
               min_points: int = MIN_POINTS):
    """PnP that ignores points far from the consensus pose (a tracker that jumped), then
    refines on the rest. Same return as `pnp`."""
    if len(model) < min_points:
        return None
    obj = np.ascontiguousarray(model, dtype=np.float64)
    img = np.ascontiguousarray(uv, dtype=np.float64)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP,
                                                 reprojectionError=threshold_px,
                                                 iterationsCount=100, confidence=0.99)
        if not ok or inl is None or len(inl) < min_points:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
    except cv2.error:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)
