#!/usr/bin/env python3
"""Compose saved policy actions and project verified gripper-only FK geometry."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.actions import sixd_to_rot
from droid_qp.geom import project
from droid_qp.policy_rollout import RolloutMetrics, integrate, place_local_points, pose_error
from droid_qp.robot_points import (PW_FLOWS, GRIPPER_URDF_SCALE, load_sampler,
                                   robot_points_3d, window_calibration)


from droid_qp.paths import DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation', type=Path, required=True)
    ap.add_argument('--plan', type=Path, default=DATA / 'validation_plan_v1.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='diagnostic subset only; zero means all')
    args = ap.parse_args()
    torch.set_num_threads(2)
    plan = json.loads(args.plan.read_text())
    manifest = DATA / 'dataset_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != plan['manifest_sha256']:
        raise ValueError('manifest differs from frozen plan')
    evaluation = json.loads((args.evaluation/'report.json').read_text())
    if evaluation['plan_sha256'] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
        raise ValueError('prediction evaluation uses another validation plan')
    shards = json.loads((DATA / 'shard_index.json').read_text())
    sampler = load_sampler(128, gripper_only=True, seed=0)
    horizons = [1, 5, 10, 20, 40]
    totals = {h: RolloutMetrics() for h in horizons}
    labs = {h: {} for h in horizons}
    qa_max_px = 0.
    oracle_max_mm = 0.
    oracle_max_deg = 0.
    n_clips = 0
    binary_local = None
    records = plan['records'][:args.limit] if args.limit else plan['records']
    for row in records:
        vid, start = row['video_id'], row['start']
        base = DATA / 'gen'/shards[vid]
        with np.load(base/'actions/labels'/f'{vid}.npz') as f:
            a = {k: f[k] for k in f.files}
        with np.load(args.evaluation/'predictions'/f'{vid}.npz') as f:
            prediction = {k: f[k] for k in f.files}
        if int(prediction['start']) != start:
            raise ValueError('prediction window differs from frozen plan')
        T = len(prediction['valid'])
        stop = min(start+T+1, len(a['position']))
        p = a['position'][start:stop].astype(np.float64)
        R = sixd_to_rot(a['rot_6d'][start:stop])
        gt_dp = a['delta_pos_body'][start:stop]
        gt_dR = sixd_to_rot(a['delta_rot6d_body'][start:stop])
        points = robot_points_3d(sampler, a['joint_positions'][start:stop],
                                 a['gripper_positions'][start:stop])
        # All sampled meshes descend from the flange. Finger articulation remains
        # in these local coordinates; the measured future width is an ORACLE input.
        local = np.einsum('tni,tij->tnj', points-p[:, None], R)
        if binary_local is None:
            jp = np.repeat(a['joint_positions'][start:start+1], 2, axis=0)
            endpoint_points = robot_points_3d(sampler, jp, np.array([GRIPPER_URDF_SCALE, 0.]))
            binary_local = (endpoint_points-p[0]) @ R[0]
        with h5py.File(PW_FLOWS/f"{row['episode']}_flows.h5", 'r') as f:
            window = sorted((k for k in f if ':' in k), key=lambda k: int(k.split(':')[0]))[0]
            E, K = window_calibration(f, window, vid.split('__')[-1])
        with np.load(base/'merged/query_points'/f'{vid}_f0.npz') as f:
            ids = f['point_index'].astype(int)
            gt_uv = f['fk_uv'][start:stop, ids]
            gt_ok = f['fk_ok'][start:stop, ids].astype(bool)
        reconstructed, depth = project(points[:, ids], E, K, out_size=(854, 480))
        qa_mask = gt_ok & (depth > 0)
        if qa_mask.any():
            maximum = float(np.linalg.norm(reconstructed-gt_uv, axis=-1)[qa_mask].max())
            qa_max_px = max(qa_max_px, maximum)
            if maximum > .1:
                raise ValueError(f'FK reprojection disagrees with saved targets: {vid}, {maximum} px')

        for h in horizons:
            lm = labs[h].setdefault(row['lab'], RolloutMetrics())
            def add(name, values):
                totals[h].add(name, values)
                lm.add(name, values)
            # Exclude origins without a known preceding action for ALL methods.
            for origin in range(min(T-h+1, len(p)-h)):
                t = start+origin
                if t < 1 or not a['valid'][t-1] or not prediction['valid'][origin:origin+h].all():
                    continue
                end = origin+h
                oracle_p, oracle_R = integrate(p[origin], R[origin], gt_dp[origin:end], gt_dR[origin:end])
                oracle_mm, oracle_deg = pose_error(oracle_p, oracle_R, p[end], R[end])
                oracle_max_mm = max(oracle_max_mm, oracle_mm)
                oracle_max_deg = max(oracle_max_deg, oracle_deg)
                if oracle_mm > .1 or oracle_deg > .1:
                    raise ValueError(f'GT delta composition fails: {vid}, {oracle_mm} mm, {oracle_deg} deg')
                previous_dp = a['delta_pos_body'][t-1]
                previous_dR = sixd_to_rot(a['delta_rot6d_body'][t-1])
                method_actions = {
                    'policy': (prediction['delta_pos_body'][origin:end], prediction['delta_rotmat_body'][origin:end]),
                    'recorded_previous': (a['delta_pos_body'][t-1:t+h-1], sixd_to_rot(a['delta_rot6d_body'][t-1:t+h-1])),
                    'hold_previous': (np.repeat(previous_dp[None], h, axis=0), np.repeat(previous_dR[None], h, axis=0)),
                    'zero': (np.zeros((h, 3)), np.repeat(np.eye(3)[None], h, axis=0))}
                for name, (dp, dR) in method_actions.items():
                    pred_p, pred_R = integrate(p[origin], R[origin], dp, dR)
                    mm, deg = pose_error(pred_p, pred_R, p[end], R[end])
                    add(name+'/position_mm', [mm])
                    add(name+'/rotation_deg', [deg])
                    width_variants = {'oracle_width': local[end, ids]}
                    if name == 'policy':
                        opening = int(prediction['gripper_logit'][end-1] >= 0)
                    elif name == 'recorded_previous':
                        opening = int(a['gripper_open'][t+h-1].item() >= .5)
                    else:
                        opening = int(a['gripper_open'][t].item() >= .5)
                    width_variants['binary_endpoint_width'] = binary_local[opening, ids]
                    for width, geometry in width_variants.items():
                        predicted_points = place_local_points(geometry, pred_p, pred_R)
                        pred_uv, pred_depth = project(predicted_points, E, K, out_size=(854, 480))
                        eligible = gt_ok[end]
                        positive = pred_depth > 0
                        # Never hide off-screen predictions by excluding them.
                        add(name+'/'+width+'/behind_camera_fraction', (~positive)[eligible])
                        add(name+'/'+width+'/pixel_error', np.linalg.norm(pred_uv-gt_uv[end], axis=-1)[eligible & positive])
                add('oracle_delta_roundtrip_mm', [oracle_mm])
        n_clips += 1
        if n_clips % 100 == 0 or n_clips == len(records):
            print(f'{n_clips}/{len(records)} rollout clips', flush=True)
    result = dict(checkpoint=evaluation['checkpoint'], checkpoint_step=evaluation['checkpoint_step'],
                  plan_sha256=evaluation['plan_sha256'], n_clips=n_clips, diagnostic_subset=bool(args.limit),
                  qa=dict(max_saved_fk_reprojection_px=qa_max_px,
                          max_ground_truth_composition_mm=oracle_max_mm,
                          max_ground_truth_composition_deg=oracle_max_deg),
                  horizons={h:{'aggregate': totals[h].result(),
                               'per_lab':{lab: m.result() for lab,m in sorted(labs[h].items())}}
                            for h in horizons},
                  limitations=[
                      'Offline integrated-action proxy: every policy prediction still uses the recorded demonstrator observations. No counterfactual Track-On frames or closed-loop robot execution.',
                      'All origins start at the measured pose. Hold-previous repeats the action known just before that origin for the entire horizon; it does not read future demonstrated actions.',
                      'Recorded-previous copies the preceding demonstrated action at EACH step and the current observed gripper state at that step. Like the policy evaluation, it uses successive recorded observations; hold-previous has less information and is an extrapolation comparison.',
                      'Oracle-width projection uses future measured finger articulation and isolates pose error. It is not a learned gripper rollout.',
                      'Binary-endpoint projection snaps predicted open/closed state to URDF angles 0/0.725 radians instantly. Zero/hold-previous persist the initial observed binary state; recorded-previous uses each recorded current state. This ignores continuous width and actuator dynamics.',
                      'FK uses gripper-only sampled geometry. EE poses do not determine the redundant arm joint configuration; no full-arm IK rollout is claimed.',
                      'Pixel means use ground-truth in-frame, positive-depth query points; predicted off-screen points remain included. Predicted nonpositive depth is reported separately. FK visibility is not an occlusion test.',
                      'Overlapping origins and multiple cameras from the same episode are correlated; counts are not independent trials. Maximum horizon is 40 because predictions cover 48 frames.'
                  ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['qa']), flush=True)


if __name__ == '__main__':
    main()
