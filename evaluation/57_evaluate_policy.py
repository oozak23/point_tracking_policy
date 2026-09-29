#!/usr/bin/env python3
"""Held-out metrics and paired scene/robot-history interventions on a saved policy."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_eval import ValidationMetrics, ValidationWindows, replace_robot_history
from droid_qp.policy_loss import PolicyLoss
from droid_qp.policy_train import ActionPolicy, to_device


from droid_qp.paths import DATA  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=DATA / 'validation_plan_v1.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ablate', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    data_cls, model_cls = PolicyWindows, ActionPolicy
    if cfg.get('proprioception', False):
        from droid_qp.policy_proprio import ProprioPolicyWindows, ProprioActionPolicy, PROPRIO_RECIPE
        if cfg['proprioception_recipe'] != PROPRIO_RECIPE:
            raise ValueError('unsupported proprioception recipe')
        data_cls, model_cls = ProprioPolicyWindows, ProprioActionPolicy
    manifest = DATA / 'dataset_manifest.json'
    plan = json.loads(args.plan.read_text())
    if plan['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('manifest differs from frozen validation plan')
    if cfg['manifest_sha256'] != plan['manifest_sha256'] or cfg['M'] != plan['M'] or cfg['T_win'] != plan['T_win']:
        raise ValueError('checkpoint incompatible with validation plan')
    model = model_cls(layers=cfg['layers'], M=cfg['M'], scene_slots=cfg['M'],
                         checkpoint=False, dropout=cfg['dropout']).to(device).eval()
    model.load_state_dict(ck['model'])
    criterion = PolicyLoss.from_json(DATA / 'policy_loss_scales.json', manifest=manifest).to(device)
    criterion.load_state_dict(ck['criterion'])
    data = data_cls('val', T_win=plan['T_win'], M=plan['M'], cache_clips=8)
    dataset = ValidationWindows(data, plan['records'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'predictions').mkdir(exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    totals = {k:ValidationMetrics() for k in ('clean', 'scene_shuffle', 'clean_anchor', 'scene_anchor', 'robot_history_anchor')}
    if cfg.get('proprioception', False):
        totals['proprio_shuffle'] = ValidationMetrics()
    labs = {k:{} for k in totals}

    def add(name, pred, batch, anchor=False):
        mask = batch['target_valid'].clone()
        if anchor:
            mask[:, :-1] = False
        terms = criterion(pred, batch['target'], mask, batch['weight'])
        totals[name].add(pred, batch, terms, anchor=anchor)
        for lab in set(batch['lab']):
            idx = torch.tensor([i for i, v in enumerate(batch['lab']) if v == lab], device=device)
            sub = {k:v[idx] for k,v in batch.items() if isinstance(v, torch.Tensor)}
            sp = {k:v[idx] for k,v in pred.items()}
            st = criterion(sp, sub['target'], mask[idx], sub['weight'])
            labs[name].setdefault(lab, ValidationMetrics()).add(sp, sub, st, anchor=anchor)

    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['amp']):
                pred = model(batch)
            add('clean', pred, batch)
            for i, vid in enumerate(batch['video_id']):
                np.savez(args.out/'predictions'/f'{vid}.npz', start=int(batch['start'][i]),
                         delta_pos_body=pred['delta_pos_body'][i, :, 0].float().cpu().numpy(),
                         delta_rotmat_body=pred['delta_rotmat_body'][i, :, 0].float().cpu().numpy(),
                         gripper_logit=pred['gripper_logit'][i, :, 0, 0].float().cpu().numpy(),
                         valid=batch['target_valid'][i].cpu().numpy())
            if args.ablate:
                records = plan['records'][offset:offset+len(batch['video_id'])]
                donors = to_device(default_collate([dataset[r['donor_row']] for r in records]), device)
                scene_batch = dict(batch, scene=donors['scene'])
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['amp']):
                    scene_pred = model(scene_batch)
                add('scene_shuffle', scene_pred, batch)
                add('clean_anchor', pred, batch, anchor=True)
                add('scene_anchor', scene_pred, batch, anchor=True)
                mappings = []
                for i, r in enumerate(records):
                    donor_r = plan['records'][r['donor_row']]
                    def ids(row):
                        vid = row['video_id']
                        f = data.gen/data.shard[vid]/'merged/query_points'/f'{vid}_f0.npz'
                        with np.load(f) as a:
                            return a['point_index'].tolist()
                    recipient_ids, donor_ids = ids(r), ids(donor_r)
                    if len(donor_ids) != 128 or not donors['hist_valid'][i].all():
                        raise ValueError('donor needs full robot coverage and full history')
                    mapping = torch.tensor([donor_ids.index(j) for j in recipient_ids], device=device)
                    mappings.append(mapping)
                robot = replace_robot_history(batch['robot'], donors['robot'], mappings)
                history_batch = dict(batch, robot=robot)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['amp']):
                    history_pred = model(history_batch)
                add('robot_history_anchor', history_pred, batch, anchor=True)
                if cfg.get('proprioception', False):
                    state_batch = dict(batch, proprio=donors['proprio'])
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['amp']):
                        state_pred = model(state_batch)
                    add('proprio_shuffle', state_pred, batch)
            offset += len(batch['video_id'])
            if offset % 160 == 0 or offset == len(dataset):
                print(f'{offset}/{len(dataset)} validation windows', flush=True)
    names = totals.keys() if args.ablate else ['clean']
    result = {name:{**totals[name].result(), 'per_lab':{k:v.result() for k,v in sorted(labs[name].items())}}
              for name in names}
    clean = result['clean']
    result.update(checkpoint=str(args.checkpoint), checkpoint_step=ck['step'],
                  proprioception=cfg.get('proprioception', False),
                  plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                  n_validation_episodes=len({r['episode'] for r in plan['records']}),
                  donor_provenance=dict(cross_lab_total=plan['cross_lab_donors'],
                      cross_lab_by_recipient_lab=dict(Counter(r['lab'] for r in plan['records']
                          if plan['records'][r['donor_row']]['lab'] != r['lab']))),
                  limitation=plan['limitation'],
                  ablation_caveat='Shuffles change joint compatibility. Current frozen robot tokens still contain prior tracking context.')
    if cfg.get('proprioception', False):
        result['proprioception_ablation_caveat'] = 'Scene and robot-history interventions retain current proprioception. Robot-history intervention tests visual history only. Proprio-shuffle replaces recorded states with donor states and is a separate compatibility-breaking diagnostic.'
    if args.ablate:
        relative = result['scene_shuffle']['common_position_mm']/clean['common_position_mm']-1
        criteria = plan['criteria']
        gates = dict(position=clean['common_position_mm'] < criteria['position_mm_strictly_below'],
                     scene=relative > criteria['scene_shuffle_relative_increase_strictly_above'],
                     transitions=clean['events']['policy']['all']['timing_cost_frames'] < criteria['transition_timing_cost_strictly_below'])
        result['pilot_decision'] = dict(gates=gates, pass_all=all(gates.values()),
                                        scene_position_relative_increase=relative, thresholds=criteria)
    (args.out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in totals}, indent=2))
    print('clean:',json.dumps({k:v for k,v in clean.items() if k not in ('events','per_lab')}))


if __name__ == '__main__':
    main()
