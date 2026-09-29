#!/usr/bin/env python3
"""Freeze validation windows, baseline measurements, donors and pilot decision criteria."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_eval import ValidationMetrics


def main():
    torch.set_num_threads(2)
    out = ROOT / 'out/validation_plan_v1.json'
    if out.exists():
        raise FileExistsError(f'{out}: frozen protocol already exists')
    manifest = ROOT / 'out/dataset_manifest.json'
    raw = manifest.read_bytes()
    all_clips = json.loads(raw)['clips']
    clips = [c for c in all_clips if c['split'] == 'val']
    if {c['episode'] for c in clips} & {c['episode'] for c in all_clips if c['split'] == 'train'}:
        raise ValueError('episode leakage')
    shards = json.loads((ROOT / 'out/shard_index.json').read_text())
    rng = np.random.default_rng(202610)
    rows, metrics, labs, whole = [], ValidationMetrics(), {}, {}
    for i, c in enumerate(clips):
        vid, lab = c['video_id'], c['video_id'].split('+')[0]
        with np.load(ROOT / 'out/gen' / shards[vid] / 'actions/labels' / f'{vid}.npz') as a:
            target = np.concatenate([a['delta_pos_body'], a['delta_rot6d_body'], a['gripper_open_next']], -1)
            valid, current = a['valid'], a['gripper_open'][:, 0]
        start = int(rng.integers(max(0, len(valid)-48)+1))
        indices = np.arange(start, start+48)
        previous = np.clip(indices-1, 0, len(valid)-1)
        length = min(48, len(valid)-start)
        padded_target = np.zeros((48,10), np.float32)
        padded_target[:length] = target[start:start+length]
        padded_valid = np.zeros(48, bool)
        padded_valid[:length] = valid[start:start+length]
        padded_current = np.zeros(48, np.float32)
        padded_current[:length] = current[start:start+length]
        batch = dict(target=torch.from_numpy(padded_target).unsqueeze(0),
                     target_valid=torch.from_numpy(padded_valid).unsqueeze(0),
                     weight=torch.tensor([c['weight']]),
                     previous_action=torch.from_numpy(target[previous, :9]).unsqueeze(0),
                     previous_valid=torch.from_numpy((indices > 0) & (indices < len(valid)) & valid[previous]).unsqueeze(0),
                     gripper_current=torch.from_numpy(padded_current).unsqueeze(0))
        pred = dict(delta_pos_body=torch.zeros(1, 48, 1, 3),
                    delta_rotmat_body=torch.eye(3).expand(1, 48, 1, 3, 3),
                    gripper_logit=(batch['gripper_current']*40-20)[:, :, None, None])
        metrics.add(pred, batch, None)
        labs.setdefault(lab, ValidationMetrics()).add(pred, batch, None)
        ok = valid[1:] & valid[:-1]
        w = whole.setdefault(lab, dict(n=0, zero_sum_mm=0., copy_sum_mm=0.))
        w['n'] += int(ok.sum())
        w['zero_sum_mm'] += float(np.linalg.norm(target[1:, :3][ok], axis=1).sum())*1000
        w['copy_sum_mm'] += float(np.linalg.norm((target[1:, :3]-target[:-1, :3])[ok], axis=1).sum())*1000
        rows.append(dict(index=i, video_id=vid, episode=c['episode'], lab=lab, start=start,
                         n_robot=c['n_robot'], weight=c['weight'], n_frames=len(valid)))
    eligible = [i for i, r in enumerate(rows) if r['n_robot'] == 128 and r['start'] >= 23]
    cross_lab = 0
    for row in rows:
        candidates = [i for i in eligible if rows[i]['episode'] != row['episode']]
        same_lab = [i for i in candidates if rows[i]['lab'] == row['lab']]
        donor = int(rng.choice(same_lab or candidates))
        row['donor_row'] = donor
        cross_lab += rows[donor]['lab'] != row['lab']
    baseline = metrics.result()
    def whole_result(v):
        return dict(n=v['n'], zero_mm=v['zero_sum_mm']/v['n'], copy_previous_mm=v['copy_sum_mm']/v['n'])
    combined = {k: sum(v[k] for v in whole.values()) for k in ('n', 'zero_sum_mm', 'copy_sum_mm')}
    plan = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                manifest_sha256=hashlib.sha256(raw).hexdigest(), seed=202610, T_win=48, M=24,
                records=rows, baseline=baseline, baseline_per_lab={k:v.result() for k,v in sorted(labs.items())},
                whole_val_baselines={**whole_result(combined), 'per_lab':{k:whole_result(v) for k,v in sorted(whole.items())}},
                n_full_history_full_robot_donors=len(eligible), cross_lab_donors=cross_lab,
                criteria=dict(position_mm_strictly_below=min(4.35, baseline['copy_position_mm']),
                              position_common_mask='valid[t] and valid[t-1] and t>0',
                              scene_shuffle_relative_increase_strictly_above=.10,
                              transition_timing_cost_strictly_below=baseline['events']['persistence']['all']['timing_cost_frames'],
                              checkpoint_selection='lowest weighted validation loss, never training loss'),
                limitation='All inputs follow recorded demonstrator trajectories. No closed-loop evaluation.')
    out.write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(out), n_clips=len(rows), n_labs=len(labs),
                         window_baselines={k:baseline[k] for k in ('n_targets','n_copy_comparable','common_zero_mm','copy_position_mm')},
                         whole_val=whole_result(combined), criteria=plan['criteria'], cross_lab_donors=cross_lab), indent=2))


if __name__ == '__main__':
    main()
