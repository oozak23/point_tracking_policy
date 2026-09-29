"""Action labels: stitch PointWorld's per-window proprioception into episode series,
then turn poses into the deltas a policy is actually supervised on.

Three things here are easy to get wrong and expensive to find later.

1. QUATERNION ORDER IS XYZW.
   Measured over 79 windows (scripts/00_calibrate_gripper.py): 0.00 deg median geodesic
   error vs URDF FK for XYZW, against 166.31 deg for WXYZ. Reading it as WXYZ still
   produces plausible-looking numbers, so this is silent. `geom.quat_to_R` already
   encodes it; never re-derive it here.

2. QUATERNION DOUBLE COVER.
   q and -q are the SAME rotation, and the raw data flips sign mid-episode -- visible in
   AUTOLab+0d4edc83+2023-10-21-19h-02m-35s window 5:16, where consecutive rows go
   -0.9982 -> -0.9989 -> -0.9980 -> +0.9972. An L2 loss on raw quaternions reads that as a
   near-maximal jump and would train the model to predict a violent flick that never
   happened. `canonicalize_quats` fixes the series; `rot_to_6d` sidesteps the issue
   entirely and is what the head should actually regress.

3. `gripper_pose` IS `panda_link8`, THE FLANGE -- not the fingertips, which are the
   `*_inner_finger` links 0.125 m further out. A grasped object sits 0.13-0.17 m from
   `gripper_pose`. Fine as an action target (it is what DROID commands), but never treat
   it as "where the gripper grasps".

Rotation representation. The head should emit **6D continuous** (Zhou et al., CVPR 2019):
the first two columns of R, flattened. Quaternions have the double cover above; Euler
angles have gimbal lock and discontinuity. 6D is continuous everywhere, which is the
property gradient descent needs.
"""
from __future__ import annotations

import numpy as np

from .geom import quat_to_R, window_bounds
from .robot_points import GRIPPER_URDF_SCALE

#: fields carried per window by the PointWorld flows h5, and their per-step shape
FIELDS = {
    "gripper_pose": 7,          # xyz + quat XYZW, panda_link8 (flange)
    "joint_positions": 7,
    "joint_velocities": 7,
    "joint_torques": 7,
    "gripper_positions": 1,     # continuous finger width
    "gripper_open": 1,          # bool
}


def episode_action_series(h5file) -> tuple[dict, np.ndarray]:
    """Stitch overlapping windows into one series per field, indexed by PointWorld step.

    Windows are 11 frames on stride 5, so they overlap; later windows overwrite earlier
    ones on shared steps, where the values are identical. Returns (fields, filled) with
    `filled` (S,) bool marking steps NO window covers -- PointWorld covers a median 76%
    of an episode timeline, so those gaps are normal and must be masked, not zero-filled.
    """
    wins = sorted((k for k in h5file.keys() if ":" in k),
                  key=lambda k: window_bounds(k)[0])
    if not wins:
        return {}, np.zeros(0, dtype=bool)

    S = max(window_bounds(w)[1] for w in wins)
    out = {k: np.zeros((S, d), dtype=np.float64) for k, d in FIELDS.items()}
    filled = np.zeros(S, dtype=bool)

    for w in wins:
        a, b = window_bounds(w)
        g = h5file[w]
        n = None
        for k, d in FIELDS.items():
            if k not in g:
                continue
            v = np.asarray(g[k][:], dtype=np.float64).reshape(len(g[k]), -1)
            m = min(len(v), b - a, S - a)
            out[k][a:a + m] = v[:m, :d]
            n = m if n is None else min(n, m)
        if n:
            filled[a:a + n] = True
    return out, filled


#: DROID stores EE orientation as Euler angles, not a quaternion. The convention is
#: scipy's extrinsic "xyz" -- measured against PointWorld's quaternion for the same steps:
#: 0.0000 deg median geodesic error, next-best convention 9.03 deg, worst 179.82 deg. So
#: this is unambiguous, but it is silent if wrong: several conventions produce
#: plausible-looking rotations.
DROID_EULER_SEQ = "xyz"

#: threshold reproducing PointWorld's `gripper_open` from the continuous width
GRIPPER_OPEN_THR = 0.10


def droid_action_series(h5file, subsample: int) -> tuple[dict, np.ndarray]:
    """Read proprioception straight from DROID's `trajectory.h5`, at our frame rate.

    PointWorld's copy of these fields is bit-identical (verified: joint_positions,
    joint_velocities, cartesian_position and gripper_positions all match to 0.000e+00),
    but it is WINDOWED and covers only a median 75% of each episode -- so taking the
    labels from PointWorld inherits gaps that do not exist in the source. DROID has every
    step. Coverage goes ~75% -> ~100% for free, and clips PointWorld barely covered (the
    worst pilot clip was 6.9% valid) become fully usable.

    PointWorld is still needed for CAMERA CALIBRATION, which DROID's raw file does not
    carry in the fitted form we project through. Only the action labels move.

    Returns the same (fields, filled) contract as `episode_action_series`, with `filled`
    all True -- kept so the two sources are drop-in interchangeable.
    """
    g = h5file["observation/robot_state"]
    sel = np.arange(0, len(g["joint_positions"]), subsample)

    cart = np.asarray(g["cartesian_position"][:], dtype=np.float64)[sel]   # (S,6) xyz+euler
    quat = _euler_to_quat_xyzw(cart[:, 3:6])
    # DROID's normalised 0-1 width -> the URDF joint angle PointWorld stores. Without
    # this, GRIPPER_OPEN_THR (calibrated against PointWorld's scale) is off by 1/0.725 and
    # the open/close label flips on every partially-closed frame.
    grip = (np.asarray(g["gripper_position"][:], dtype=np.float64)[sel].reshape(-1, 1)
            * GRIPPER_URDF_SCALE)

    out = {
        "gripper_pose": np.concatenate([cart[:, :3], quat], axis=1),
        "joint_positions": np.asarray(g["joint_positions"][:], dtype=np.float64)[sel],
        "joint_velocities": np.asarray(g["joint_velocities"][:], dtype=np.float64)[sel],
        "joint_torques": np.asarray(g["joint_torques_computed"][:],
                                    dtype=np.float64)[sel],
        "gripper_positions": grip,
        # DROID records a continuous width, not a boolean; PointWorld's `gripper_open` is
        # derived from it. Reproduce THEIR rule rather than inventing one, so the two
        # sources are interchangeable.
        #
        # Recovered by search over 4,378 steps from 25 episodes:
        #     gripper_open == (gripper_positions < 0.10)     100.00% agreement
        # The safe range is roughly (0.01, 0.26) -- open sits at p90 = 0.010 and closed at
        # p10 = 0.262, so the classes are well separated and any threshold in the gap
        # works. GRIPPER_OPEN_THR is the middle of it.
        #
        # An earlier guess of 0.5 passed a spot check only because that window's gripper
        # sat at exactly 0.0 the whole time; across the pilot it mislabelled 9.4% of
        # frames -- every frame with the gripper part-way closed, which is precisely the
        # grasp transition the policy has to learn.
        "gripper_open": (grip < GRIPPER_OPEN_THR).astype(np.float64),
    }
    return out, np.ones(len(sel), dtype=bool)


def _euler_to_quat_xyzw(eul: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler -> XYZW quaternion, matching `geom.quat_to_R`'s convention."""
    from scipy.spatial.transform import Rotation as _R

    return _R.from_euler(DROID_EULER_SEQ, eul).as_quat()      # scipy returns XYZW


def canonicalize_quats(q: np.ndarray) -> np.ndarray:
    """Remove double-cover sign flips along a trajectory: force dot(q[t], q[t+1]) >= 0.

    Operates in place on a copy, sequentially -- a vectorised version would be wrong,
    because each decision depends on the already-corrected previous frame.
    """
    q = np.array(q, dtype=np.float64, copy=True)
    for t in range(1, len(q)):
        if float(q[t - 1] @ q[t]) < 0:
            q[t] = -q[t]
    return q


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two COLUMNS of R, flattened. Zhou et al. 2019."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_rot(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt. Inverse of `rot_to_6d`; the model's head
    emits 6D and this is how it becomes a rotation at evaluation time."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pose_deltas(gripper_pose: np.ndarray, filled: np.ndarray | None = None) -> dict:
    """(S,7) xyz+quatXYZW -> the step-to-step deltas a policy predicts.

    `filled` (S,) marks steps a PointWorld window covers. It is NOT optional in practice:
    a delta spans steps t and t+1, so it is meaningful only when BOTH are covered.
    Checking only `filled[t]` records the arm's motion across an entire uncovered gap as
    one action -- measured on AUTOLab+5d05c5aa+2023-11-17-22h-46m-01s, that produced
    7 "actions" of 51-75 cm per frame, i.e. >5 m/s at the end effector, which no Franka
    and no teleoperator produces. All 7 sat exactly on coverage boundaries and none had
    both ends covered. Deltas failing this test are zeroed as well as masked, so a
    forgotten mask downstream degrades to a no-op instead of a 70 cm lunge.

    Both conventions are returned because they suit different heads and both are free:

      WORLD (base) frame   dp = p[t+1] - p[t],  dR = R[t+1] @ R[t].T
          matches DROID's own cartesian command space.
      BODY (EE) frame      dp = R[t].T @ (p[t+1] - p[t]),  dR = R[t].T @ R[t+1]
          translation-invariant, so the same motion has the same label wherever in the
          workspace it happens. Usually the better-conditioned target.

    The final step has no successor, so `valid_delta[-1]` is False. Deltas are computed
    from CANONICALISED quaternions, otherwise a double-cover sign flip appears as a ~180
    degree rotation that never occurred.
    """
    p = gripper_pose[:, :3]
    R = quat_to_R(canonicalize_quats(gripper_pose[:, 3:7]))
    S = len(p)

    dp_w = np.zeros((S, 3)); dR_w = np.tile(np.eye(3), (S, 1, 1))
    dp_b = np.zeros((S, 3)); dR_b = np.tile(np.eye(3), (S, 1, 1))
    if S > 1:
        dp_w[:-1] = p[1:] - p[:-1]
        dR_w[:-1] = R[1:] @ R[:-1].transpose(0, 2, 1)
        dp_b[:-1] = np.einsum("sij,sj->si", R[:-1].transpose(0, 2, 1), p[1:] - p[:-1])
        dR_b[:-1] = R[:-1].transpose(0, 2, 1) @ R[1:]

    valid = np.ones(S, dtype=bool)
    valid[-1] = False
    if filled is not None:
        f = np.asarray(filled, dtype=bool)
        valid &= f
        valid[:-1] &= f[1:]          # the delta needs BOTH ends covered
    for a in (dp_w, dp_b):
        a[~valid] = 0.0
    for a in (dR_w, dR_b):
        a[~valid] = np.eye(3)
    return {
        "position": p, "rot_6d": rot_to_6d(R),
        "delta_pos_world": dp_w, "delta_rot6d_world": rot_to_6d(dR_w),
        "delta_pos_body": dp_b, "delta_rot6d_body": rot_to_6d(dR_b),
        "valid_delta": valid,
    }
