#!/usr/bin/env python3
"""Is the geometric 43% ceiling real, or is the gripper region simply not in the data?

20/21 unscorable clips had the gripper CLOSED and still produced ~0 points moving
rigidly with it. Three candidate explanations, and they demand different responses:

  (a) the held object is too small/thin to reconstruct at 320x180  -> hard ceiling
  (b) the signal is present but below the r<0.03 cut               -> retune, recoverable
  (c) `scene_flows` EXCLUDES the robot, and with it the gripper and much of the
      object region                                                -> design error on our side

(c) is testable and decisive: the gripper's own surface must move rigidly with the
gripper by definition, so if no dense points exist near the FK fingertips, the point
cloud is not describing that part of the scene at all.

Uses the Phase 0 calibration: `gripper_pose` is panda_link8 (the FLANGE), so the
fingertips come from FK on the `*_inner_finger` links, 0.125 m further out.

    python scripts/10_probe_gripper_region.py --episodes 25
Run with /home/oozak23/.conda/envs/pointworld-env/bin/python
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.geom import project, rigid_residuals, window_bounds

PW_FLOWS = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
                "/droid/flows-fs-optimized")
URDF = POINTWORLD / "assets/franka_description/franka_panda_robotiq_2f85.urdf"
OUT = Path(__file__).resolve().parent.parent / "out"

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_LINKS = ["left_inner_finger", "right_inner_finger"]
ARM_LINKS = ["panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_link8"]


def build_chain():
    from utils import build_pk_chain_from_urdf
    return build_pk_chain_from_urdf(URDF.read_bytes()).to(dtype=torch.float64)


def fk_links(chain, jp, gp, wanted):
    from robot_sampler import build_robotiq_joint_dict
    jpt = torch.as_tensor(jp, dtype=torch.float64)
    gpt = torch.as_tensor(gp, dtype=torch.float64).reshape(-1)
    names = set(chain.get_joint_parameter_names())
    jd = {n: jpt[:, i] for i, n in enumerate(PANDA_JOINTS) if n in names}
    try:
        jd.update(build_robotiq_joint_dict(gpt, list(names)))
    except Exception:
        pass
    ret = chain.forward_kinematics(jd)
    out = {}
    for n in wanted:
        if n in ret:
            M = ret[n].get_matrix().detach().cpu().numpy()
            out[n] = np.asarray(M, dtype=np.float64)[:, :3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--radius-3d", type=float, default=0.10, help="metres around fingertips")
    ap.add_argument("--radius-px", type=float, default=25.0, help="pixels, at 854x480")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT / "gripper_region_probe.json")
    args = ap.parse_args()

    chain = build_chain()
    files = sorted(f for f in os.listdir(PW_FLOWS) if f.endswith("_flows.h5"))
    random.seed(args.seed)
    files = random.sample(files, min(args.episodes, len(files)))

    rows = []
    for k, fn in enumerate(files, 1):
        try:
            with h5py.File(PW_FLOWS / fn, "r") as f:
                wins = sorted((w for w in f.keys() if ":" in w),
                              key=lambda w: window_bounds(w)[0])
                mags = f["ee_pos_motion_magnitudes"].attrs
                for w in wins:
                    if float(mags.get(w, 0.0)) < 0.08:
                        continue
                    go = np.asarray(f[w]["gripper_open"][:]).ravel().astype(bool)
                    if go.any():          # carrying windows only
                        continue
                    cams = [c for c in f[w].keys() if c.startswith("camera_")]
                    if not cams:
                        continue
                    g = f[w][cams[0]]
                    sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
                    vis = np.asarray(g["scene_visibility"][:])
                    dvm = np.asarray(g["scene_depth_valid_mask"][:])
                    E = np.asarray(g["extrinsic"][:], dtype=np.float64)
                    K = np.asarray(g["intrinsic"][:], dtype=np.float64)
                    jp = np.asarray(f[w]["joint_positions"][:], dtype=np.float64)
                    gp = np.asarray(f[w]["gripper_positions"][:], dtype=np.float64)
                    pose = np.asarray(f[w]["gripper_pose"][:], dtype=np.float64)

                    ref = sf.shape[0] // 2
                    links = fk_links(chain, jp, gp, FINGER_LINKS + ARM_LINKS)
                    if not all(n in links for n in FINGER_LINKS):
                        continue
                    tip = np.stack([links[n][ref] for n in FINGER_LINKS]).mean(axis=0)

                    r, s, grip = rigid_residuals(sf, pose, ref=ref)
                    good = vis[ref] & dvm[ref]
                    if good.sum() < 500 or grip < 0.08:
                        continue

                    X = sf[ref]
                    d3 = np.linalg.norm(X - tip, axis=1)
                    near3 = good & (d3 < args.radius_3d)

                    uv, z = project(X, E, K, out_size=(854, 480))
                    tip_uv, tip_z = project(tip[None], E, K, out_size=(854, 480))
                    d2 = np.linalg.norm(uv - tip_uv[0], axis=1)
                    near2 = good & (z > 0) & (d2 < args.radius_px)

                    # Are the ARM links represented at all? The arm is large and
                    # unambiguous, so its absence would mean the robot is masked out.
                    arm_near = 0
                    for n in ARM_LINKS:
                        if n in links:
                            arm_near += int((good & (np.linalg.norm(X - links[n][ref], axis=1)
                                                     < 0.08)).sum())

                    rows.append({
                        "episode": fn[:-len("_flows.h5")], "window": w,
                        "grip_disp": float(grip),
                        "n_valid": int(good.sum()),
                        "n_near_tip_3d": int(near3.sum()),
                        "n_near_tip_2d": int(near2.sum()),
                        "n_near_arm_3d": int(arm_near),
                        "tip_in_frame": bool(0 <= tip_uv[0, 0] < 854 and
                                             0 <= tip_uv[0, 1] < 480 and tip_z[0] > 0),
                        "r_near_tip_3d": (float(np.median(r[near3])) if near3.any() else None),
                        "r_near_tip_2d": (float(np.median(r[near2])) if near2.any() else None),
                        "n_attached_003": int(((r < 0.03) & (s > max(0.03, 0.3*grip)) & good).sum()),
                    })
        except Exception as e:
            print(f"  [skip] {fn}: {type(e).__name__}: {e}")
        if k % 5 == 0:
            print(f"  {k}/{len(files)} episodes, {len(rows)} carrying windows")

    if not rows:
        sys.exit("no usable windows")

    inf = [x for x in rows if x["tip_in_frame"]]
    print(f"\n===== {len(rows)} carrying windows ({len(inf)} with fingertips in frame) =====")

    n3 = np.array([x["n_near_tip_3d"] for x in inf])
    n2 = np.array([x["n_near_tip_2d"] for x in inf])
    na = np.array([x["n_near_arm_3d"] for x in inf])
    att = np.array([x["n_attached_003"] for x in inf])
    valid = np.array([x["n_valid"] for x in inf])

    print(f"\ndense points near the FK FINGERTIPS (within {args.radius_3d} m in 3D):")
    print(f"  median {np.median(n3):.0f}   mean {n3.mean():.0f}   zero in "
          f"{100*(n3 == 0).mean():.0f}% of windows")
    print(f"dense points near the fingertips in 2D (within {args.radius_px:.0f} px):")
    print(f"  median {np.median(n2):.0f}   mean {n2.mean():.0f}   zero in "
          f"{100*(n2 == 0).mean():.0f}% of windows")
    print(f"dense points near the ARM links (within 0.08 m):")
    print(f"  median {np.median(na):.0f}   mean {na.mean():.0f}   zero in "
          f"{100*(na == 0).mean():.0f}% of windows")
    print(f"\nvalid dense points per window: median {np.median(valid):.0f}")
    print(f"attached points (r<0.03): median {np.median(att):.0f}, "
          f"zero in {100*(att == 0).mean():.0f}% of windows")

    print(f"\nresidual of points NEAR the fingertips -- if these are large, the region")
    print(f"is reconstructed but not moving rigidly; if there are no points, it is absent:")
    rr = np.array([x["r_near_tip_3d"] for x in inf if x["r_near_tip_3d"] is not None])
    if len(rr):
        for p in [10, 25, 50, 75, 90]:
            print(f"  p{p:<3d} {np.percentile(rr, p):.4f} m")
        print(f"  fraction of those windows with median r < 0.03: {100*(rr < 0.03).mean():.0f}%")
    else:
        print("  NO windows have any dense point near the fingertips.")

    zero_tip = (n3 == 0)
    print(f"\n--- verdict ---")
    if zero_tip.mean() > 0.5 and np.median(na) < 50:
        print("  Neither the fingertips NOR the arm are represented in scene_flows.")
        print("  => the robot is masked out of the point cloud. Explanation (c).")
    elif zero_tip.mean() > 0.5:
        print("  The arm IS represented but the fingertip region is not.")
        print("  => held objects/gripper are occluded or unreconstructed. Explanation (a).")
    else:
        print("  The fingertip region IS represented; check the residuals above to see")
        print("  whether the signal is merely below threshold. Explanation (b).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
