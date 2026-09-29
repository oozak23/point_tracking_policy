"""Robot query points by URDF forward kinematics, projected to pixels.

WHAT IS OURS vs POINTWORLD'S
    theirs : RobotSampler, build_robotiq_joint_dict, deterministic_sample_surface
             (PointWorld/robot_sampler.py), the URDF, and the joint/gripper data
    ours   : calling their sampler on their dataset's joints, projecting 3D -> 2D
             pixels, and stitching overlapping windows into one episode series
PointWorld works entirely in 3D and never needs pixels; Track-On queries are 2D, so the
projection step has to exist somewhere and this is it. Nothing is copied from them.

Everything below is exact rather than fitted:

  * FK reproduces PointWorld's own `gripper_pose` to **0.0000 m** over 79 windows
    (pipeline/00_calibrate_gripper.py) -- so the URDF chain, the panda_joint1..7 mapping
    and the Robotiq mimic joints are verified end to end.
  * `gripper_pose` is `panda_link8`, the FLANGE. Fingertips are the `*_inner_finger`
    links, 0.125 m further out. Never treat `gripper_pose` as "where the gripper grasps".
  * The extrinsics we project through were fitted BY PointWorld by aligning FK robot
    points to the image (`cameras/*.json`: ~5,309 robot points, loss 0.23 -> 0.06), so
    they are tuned for exactly this operation. The copy in `cameras/*.json` is bit-identical
    to the one in the flows h5, so either source works.

Unlike a segmentation-derived point set, `presample(n, seed)` is deterministic: point
index *n* is the SAME physical spot on the robot in every clip and every episode. For a
fixed-dimension policy input that stable index semantics matters -- a k-means point set
has arbitrary per-clip ordering and is close to useless as a consistent action
representation.

Needs pytorch_kinematics -> $PY_POINTWORLD, which
has no `transformers`; stage results to .npz for the tracking step.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .paths import POINTWORLD, PW_FLOWS  # noqa: F401  (PW_FLOWS re-exported for pipeline scripts)
if str(POINTWORLD) not in sys.path:
    sys.path.insert(0, str(POINTWORLD))

from .geom import project, window_bounds

URDF = POINTWORLD / "assets/franka_description/franka_panda_robotiq_2f85.urdf"

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

#: DROID records the gripper as a NORMALISED width in [0, 1]; the URDF (and therefore
#: `build_robotiq_joint_dict`) wants the finger JOINT ANGLE in radians. PointWorld converts
#: between them by this factor, measured as exactly 0.725000 -- min = max = median over 11
#: random episodes, no spread at all. 0.725 rad = 41.5 deg, the Robotiq 2F-85 finger limit.
#:
#: Feeding the raw 0-1 value straight to FK opens the fingers 1/0.725 = 1.38x too far.
#: Every one of our query points sits on the gripper, so that moves ALL of them: measured
#: up to 16 px of error against the PointWorld-sourced geometry.
GRIPPER_URDF_SCALE = 0.725


def load_sampler(num_points: int = 128, gripper_only: bool = False,
                 seed: int = 0, device: str = "cpu"):
    from robot_sampler import RobotSampler

    s = RobotSampler(str(URDF), gripper_only=gripper_only, device=device)
    s.presample(num_points, seed=seed)
    return s


def robot_points_3d(sampler, joint_positions: np.ndarray,
                    gripper_positions: np.ndarray, device="cpu") -> np.ndarray:
    """-> (T, N, 3) in the robot base frame (== PointWorld's world frame).

    Mirrors PointWorld's own DROID path (dataset_components/robot.py:393) so the point
    set matches what their pipeline produces.
    """
    from robot_sampler import build_robotiq_joint_dict

    dtype = sampler.dtype
    jp = torch.as_tensor(joint_positions, device=device, dtype=dtype)
    gp = torch.as_tensor(gripper_positions, device=device, dtype=dtype).reshape(-1)

    missing = [n for n in PANDA_JOINTS if n not in sampler.joint_names]
    if missing:
        raise KeyError(f"URDF missing expected DROID joints: {missing}")

    jd = {n: jp[:, i].reshape(-1) for i, n in enumerate(PANDA_JOINTS)}
    jd.update(build_robotiq_joint_dict(gp, sampler.joint_names))

    pts, _c, _n = sampler.compute_points(jd)
    return pts.detach().cpu().numpy().astype(np.float64)


def robot_points_2d(points3d: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
                    out_size=(854, 480)):
    """(T,N,3) world -> (uv (T,N,2), in_frame (T,N) bool).

    `in_frame` is the filter a query set needs: Track-On has nothing to latch onto for a
    point projecting outside the image or behind the camera. Self-occlusion (a point on
    the far side of the arm) is NOT detected here -- it needs a depth test, and those
    points will simply be hard to track.
    """
    T, N, _ = points3d.shape
    uv, z = project(points3d.reshape(-1, 3), extrinsic, intrinsic, out_size=out_size)
    uv, z = uv.reshape(T, N, 2), z.reshape(T, N)
    W, H = out_size
    ok = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & \
         (uv[..., 1] >= 0) & (uv[..., 1] < H)
    return uv, ok


def window_calibration(h5file, window: str, cam: str):
    g = h5file[window][f"camera_{cam}_ext"]
    return (np.asarray(g["extrinsic"][:], dtype=np.float64),
            np.asarray(g["intrinsic"][:], dtype=np.float64))


def droid_joint_series(h5file, subsample: int):
    """Joints straight from DROID's `trajectory.h5`, at our frame rate. No gaps.

    Prefer this over `episode_joint_series`. PointWorld's copy of these fields is
    bit-identical (verified 0.000e+00) but WINDOWED, covering a median ~75% of an episode,
    and `episode_joint_series` ZERO-FILLS the rest. Zeros are not "missing" to FK -- they
    are a valid pose, the arm standing straight up, and FK happily projects it.

    Usually that lands outside the image so `fk_ok` rejects it, which is luck rather than
    design: it depends on where the camera sits. Measured over 80 pilot clips, **1.085% of
    uncovered point-steps still pass the visibility test**, and in the 3 affected clips it
    is near-total (one TRI clip: 2,304 of 2,304). Those frames enter the "ground truth"
    as a plausible pixel coordinate for a pose the arm was never in.

    Reading DROID removes the failure mode rather than masking it, and takes FK coverage
    from ~75% to 100%. PointWorld is still needed for the CAMERA CALIBRATION.

    Returns (joint_positions (S,7), gripper_positions (S,), filled (S,) all-True) to match
    `episode_joint_series`, so the two are drop-in interchangeable.
    """
    g = h5file["observation/robot_state"]
    sel = np.arange(0, len(g["joint_positions"]), subsample)
    jp = np.asarray(g["joint_positions"][:], dtype=np.float64)[sel]
    gp = np.asarray(g["gripper_position"][:], dtype=np.float64)[sel].reshape(-1)
    # -> PointWorld's convention (URDF joint angle), so everything downstream --
    # build_robotiq_joint_dict here, GRIPPER_OPEN_THR in actions.py -- keeps working.
    return jp, gp * GRIPPER_URDF_SCALE, np.ones(len(sel), dtype=bool)


def episode_joint_series(h5file):
    """Stitch per-window joint state into one series indexed by PointWorld step.

    Windows overlap (11 frames, stride 5); later windows overwrite earlier ones on the
    shared frames, which carry identical values. Returns
    (joint_positions (S,7), gripper_positions (S,), filled (S,) bool) -- `filled` marks
    steps no window covers, since PointWorld covers a median 76% of each timeline.
    """
    wins = sorted((k for k in h5file.keys() if ":" in k),
                  key=lambda k: window_bounds(k)[0])
    if not wins:
        return None, None, None
    S = max(window_bounds(w)[1] for w in wins)
    jp = np.zeros((S, 7), dtype=np.float64)
    gp = np.zeros((S,), dtype=np.float64)
    filled = np.zeros((S,), dtype=bool)
    for w in wins:
        a, b = window_bounds(w)
        j = np.asarray(h5file[w]["joint_positions"][:], dtype=np.float64)
        g = np.asarray(h5file[w]["gripper_positions"][:], dtype=np.float64).reshape(-1)
        n = min(len(j), b - a, S - a)
        jp[a:a + n], gp[a:a + n], filled[a:a + n] = j[:n], g[:n], True
    return jp, gp, filled
