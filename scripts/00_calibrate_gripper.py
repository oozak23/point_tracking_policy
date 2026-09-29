#!/usr/bin/env python3
"""Phase 0 -- calibrate the two unknowns in PointWorld's gripper data.

Both silently corrupt the rigid-motion scorer if wrong, so they are settled first.

  (1) What IS `gripper_pose`?  Its position sits ~17 cm from the nearest dense scene
      point, which suggests the flange rather than the fingertips. We do not guess:
      run URDF forward kinematics from `joint_positions`, evaluate EVERY link, and
      report which link's origin best matches `gripper_pose[:, :3]`.

  (2) Quaternion order -- `wxyz` or `xyzw`?  `gripper_pose[:, 3:7]` is a unit
      quaternion (verified: norm 1.0), but the order is undocumented. Once (1) gives
      the matching link, compare that link's FK rotation against both readings and
      report the geodesic angle error of each. The wrong choice corrupts every
      residual in the scorer while still looking plausible.

Reads only; writes a JSON report. /frozen is read-only.

    python scripts/00_calibrate_gripper.py --episodes 40
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python (pytorch_kinematics).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

POINTWORLD = Path("/scratch/oozak23/PointWorld")
sys.path.insert(0, str(POINTWORLD))

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
URDF = POINTWORLD / "assets/franka_description/franka_panda_robotiq_2f85.urdf"
OUT = Path(__file__).resolve().parent.parent / "out"

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def quat_to_R(q: np.ndarray, order: str) -> np.ndarray:
    """(...,4) unit quaternion -> (...,3,3). order is 'wxyz' or 'xyzw'."""
    q = np.asarray(q, dtype=np.float64)
    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    elif order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        raise ValueError(order)
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


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle of Ra^T Rb, in degrees. 0 => identical orientation."""
    M = np.einsum("...ji,...jk->...ik", Ra, Rb)
    tr = np.trace(M, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def build_chain():
    from utils import build_pk_chain_from_urdf

    chain = build_pk_chain_from_urdf(URDF.read_bytes())
    return chain.to(dtype=torch.float64)


def fk_all_links(chain, joint_positions, gripper_positions):
    """-> {link_name: (T,4,4)} in the robot base frame."""
    from robot_sampler import build_robotiq_joint_dict

    T = joint_positions.shape[0]
    jp = torch.as_tensor(joint_positions, dtype=torch.float64)
    gp = torch.as_tensor(gripper_positions, dtype=torch.float64).reshape(-1)

    names = set(chain.get_joint_parameter_names())
    jd = {n: jp[:, i] for i, n in enumerate(PANDA_JOINTS) if n in names}
    try:
        jd.update(build_robotiq_joint_dict(gp, list(names)))
    except Exception as e:
        print(f"    (robotiq joints unavailable: {type(e).__name__}: {e})")

    ret = chain.forward_kinematics(jd)
    out = {}
    for name, tf in ret.items():
        M = tf.get_matrix() if hasattr(tf, "get_matrix") else tf
        M = np.asarray(M.detach().cpu().numpy(), dtype=np.float64)
        if M.ndim == 2:
            M = M[None].repeat(T, axis=0)
        out[name] = M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--windows-per-episode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "gripper_calibration.json")
    args = ap.parse_args()

    print(f"[calib] URDF: {URDF}")
    chain = build_chain()
    print(f"[calib] chain joints: {len(chain.get_joint_parameter_names())}")

    files = sorted(f for f in os.listdir(PW_FLOWS) if f.endswith("_flows.h5"))
    random.seed(args.seed)
    files = random.sample(files, min(args.episodes, len(files)))

    pos_err = {}     # link -> list of per-frame position errors
    off_local = {}   # link -> list of (offset expressed in the link frame)
    rot_err = {"wxyz": [], "xyzw": []}
    best_link_votes = {}
    n_used = 0

    for k, fn in enumerate(files, 1):
        try:
            with h5py.File(PW_FLOWS / fn, "r") as f:
                wins = sorted((w for w in f.keys() if ":" in w),
                              key=lambda w: int(w.split(":")[0]))
                if not wins:
                    continue
                for w in wins[: args.windows_per_episode]:
                    c = f[w]
                    jp = np.asarray(c["joint_positions"][:], dtype=np.float64)
                    gp = np.asarray(c["gripper_positions"][:], dtype=np.float64)
                    pose = np.asarray(c["gripper_pose"][:], dtype=np.float64)
                    if jp.shape[0] != pose.shape[0]:
                        continue
                    links = fk_all_links(chain, jp, gp)

                    p_ref = pose[:, :3]
                    per_link = {}
                    for name, M in links.items():
                        d = np.linalg.norm(M[:, :3, 3] - p_ref, axis=1)
                        per_link[name] = float(d.mean())
                        pos_err.setdefault(name, []).append(float(d.mean()))
                        # offset expressed in the link's own frame -- constant if it is
                        # a fixed flange->TCP translation rather than noise
                        R = M[:, :3, :3]
                        local = np.einsum("tji,tj->ti", R, p_ref - M[:, :3, 3])
                        off_local.setdefault(name, []).append(local.mean(axis=0))

                    best = min(per_link, key=per_link.get)
                    best_link_votes[best] = best_link_votes.get(best, 0) + 1

                    R_fk = links[best][:, :3, :3]
                    for order in ("wxyz", "xyzw"):
                        R_q = quat_to_R(pose[:, 3:7], order)
                        rot_err[order].append(float(np.median(geodesic_deg(R_fk, R_q))))
                    n_used += 1
        except Exception as e:
            print(f"  [skip] {fn}: {type(e).__name__}: {e}")
        if k % 10 == 0:
            print(f"  {k}/{len(files)} episodes, {n_used} windows")

    if not n_used:
        sys.exit("no usable windows")

    print(f"\n===== (1) which link is `gripper_pose`?  ({n_used} windows) =====")
    ranked = sorted(pos_err.items(), key=lambda kv: float(np.mean(kv[1])))
    print(f"{'link':32s} {'mean pos err (m)':>17s} {'std':>8s}")
    for name, errs in ranked[:10]:
        print(f"{name:32s} {np.mean(errs):17.4f} {np.std(errs):8.4f}")
    best_link, best_errs = ranked[0]
    print(f"\nbest match: {best_link}   mean {np.mean(best_errs):.4f} m")
    print(f"per-window vote: {sorted(best_link_votes.items(), key=lambda kv: -kv[1])[:5]}")

    off = np.stack(off_local[best_link])
    print(f"\noffset gripper_pose - {best_link}, in the link frame (m):")
    print(f"  mean {off.mean(axis=0)}  std {off.std(axis=0)}")
    print("  (small std => a FIXED offset we can apply; large std => not a rigid offset)")

    print(f"\n===== (2) quaternion order =====")
    for order in ("wxyz", "xyzw"):
        v = np.array(rot_err[order])
        print(f"  {order}: median geodesic error {np.median(v):8.2f} deg   "
              f"p90 {np.percentile(v, 90):7.2f} deg")
    winner = min(rot_err, key=lambda o: np.median(rot_err[o]))
    loser = "xyzw" if winner == "wxyz" else "wxyz"
    print(f"\n  -> `gripper_pose[:, 3:7]` is {winner.upper()}"
          f"  ({np.median(rot_err[winner]):.2f} deg vs {np.median(rot_err[loser]):.2f} deg)")
    if np.median(rot_err[winner]) > 15:
        print("  !! WARNING: even the better reading is far off. `gripper_pose` may not be")
        print("     the same frame as any URDF link, or the FK joint mapping is wrong.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "n_windows": n_used,
        "best_link": best_link,
        "best_link_pos_err_m": float(np.mean(best_errs)),
        "link_ranking": [(n, float(np.mean(e))) for n, e in ranked[:15]],
        "offset_in_link_frame_m": off.mean(axis=0).tolist(),
        "offset_std_m": off.std(axis=0).tolist(),
        "quat_order": winner,
        "quat_median_err_deg": {o: float(np.median(rot_err[o])) for o in rot_err},
    }, open(args.out, "w"), indent=1)
    print(f"\n[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
