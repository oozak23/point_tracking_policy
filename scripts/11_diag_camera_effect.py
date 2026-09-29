#!/usr/bin/env python3
"""Why do 18/40 eval clips have ~0 attached points when a random probe found only 13%?

The probe used whichever camera came first in the window; the scorer uses the specific
camera MolmoMotion chose. PointWorld stores both exterior views per window, and they see
the workspace from very different angles -- so one may reconstruct the held object while
the other does not.

For each failing clip this reports, PER CAMERA: valid dense points, points near the FK
fingertips, and points attached to the gripper. If the two cameras disagree sharply, the
geometric signal is a per-view property and the scorer must pick the better view rather
than inherit MolmoMotion's choice.

    python scripts/11_diag_camera_effect.py --score out/score_a1_mw40_fixed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

POINTWORLD = Path("/scratch/oozak23/PointWorld")
sys.path.insert(0, str(POINTWORLD))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.geom import (attached_mask, carrying_span, rigid_residuals, window_bounds)

PW = Path("/frozen/avg/bora_data/droid_datasets/pointworld_droid_restored"
          "/droid/flows-fs-optimized")
URDF = POINTWORLD / "assets/franka_description/franka_panda_robotiq_2f85.urdf"
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGERS = ["left_inner_finger", "right_inner_finger"]


def build_chain():
    from utils import build_pk_chain_from_urdf
    return build_pk_chain_from_urdf(URDF.read_bytes()).to(dtype=torch.float64)


def fingertip(chain, jp, gp, ref):
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
    pts = [np.asarray(ret[n].get_matrix().detach().cpu().numpy())[ref, :3, 3]
           for n in FINGERS if n in ret]
    return np.mean(pts, axis=0) if pts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--max-clips", type=int, default=20)
    args = ap.parse_args()

    chain = build_chain()
    rows = json.load(open(args.score))["rows"]
    targets = [r for r in rows
               if r.get("status") in ("too_few_attached_points", "ok") and "window" in r]
    targets = targets[: args.max_clips]

    print(f"{'video_id':44s} {'cam':>9s} {'valid':>7s} {'near_tip':>9s} {'attached':>9s} {'chosen':>7s}")
    disagree = 0
    per_status = {"ok": [], "too_few_attached_points": []}
    for r in targets:
        ep, w, chosen_cam = r["episode"], r["window"], r["cam"]
        try:
            with h5py.File(PW / f"{ep}_flows.h5", "r") as f:
                grp = f[w]
                cams = [c for c in grp.keys() if c.startswith("camera_")]
                jp = np.asarray(grp["joint_positions"][:], dtype=np.float64)
                gp = np.asarray(grp["gripper_positions"][:], dtype=np.float64)
                pose = np.asarray(grp["gripper_pose"][:], dtype=np.float64)
                gopen = np.asarray(grp["gripper_open"][:]).ravel().astype(bool)
                span = carrying_span(gopen, ref=r.get("local_index", 0))
                if span is None:
                    continue
                ref = int(np.clip(r.get("local_index", 0), span[0], span[1]))
                tip = fingertip(chain, jp, gp, ref)

                counts = {}
                for c in cams:
                    g = grp[c]
                    sf = np.asarray(g["scene_flows"][:], dtype=np.float64)
                    vis = np.asarray(g["scene_visibility"][:])
                    dvm = np.asarray(g["scene_depth_valid_mask"][:])
                    rr, ss, grip = rigid_residuals(sf, pose, ref=ref, span=span)
                    good = vis[int(np.clip(ref, 0, sf.shape[0]-1))] & \
                           dvm[int(np.clip(ref, 0, sf.shape[0]-1))]
                    att = attached_mask(rr, ss, grip) & good
                    near = 0
                    if tip is not None:
                        near = int((good & (np.linalg.norm(
                            sf[int(np.clip(ref, 0, sf.shape[0]-1))] - tip, axis=1) < 0.10)).sum())
                    serial = c.split("_")[1]
                    counts[serial] = (int(good.sum()), near, int(att.sum()))
                for serial, (v, n, a) in counts.items():
                    mark = "<--" if serial == chosen_cam else ""
                    print(f"{r['video_id'][:44]:44s} {serial:>9s} {v:7d} {n:9d} {a:9d} {mark:>7s}")
                vals = [a for _, _, a in counts.values()]
                if len(vals) > 1 and max(vals) >= 20 and min(vals) < 20:
                    disagree += 1
                per_status[r["status"]].append(
                    (counts.get(chosen_cam, (0, 0, 0))[2], max(vals) if vals else 0))
        except Exception as e:
            print(f"  [skip] {ep}: {type(e).__name__}: {e}")

    print(f"\nclips where one camera has >=20 attached and the other has <20: {disagree}")
    for st, v in per_status.items():
        if not v:
            continue
        chosen = np.array([x[0] for x in v]); best = np.array([x[1] for x in v])
        print(f"\n{st}: n={len(v)}")
        print(f"  attached with MolmoMotion's camera : median {np.median(chosen):.0f}, "
              f">=20 in {int((chosen >= 20).sum())}/{len(v)}")
        print(f"  attached with the BEST camera      : median {np.median(best):.0f}, "
              f">=20 in {int((best >= 20).sum())}/{len(v)}")


if __name__ == "__main__":
    main()
