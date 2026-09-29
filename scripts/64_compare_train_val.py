#!/usr/bin/env python3
"""Evaluate the one-epoch pilot on its exact seen windows and frozen validation set."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_eval import ValidationWindows, evaluate_validation
from droid_qp.policy_loss import PolicyLoss
from droid_qp.policy_train import ActionPolicy, TrainingWindows


class SeenWindows(Dataset):
    """Use the training sampler itself to replay seed/epoch/index-selected starts."""
    def __init__(self, data, seed):
        self.data = data
        self.training = TrainingWindows(data, seed=seed)
        self.training.epoch = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        original = self.training[i]
        row = dict(index=i, start=original['start'], episode=self.data.clips[i]['episode'])
        # Reuse precisely the validation metric inputs and masks. The token file
        # is already cached by TrainingWindows, so this does not reload it.
        return ValidationWindows(self.data, [row])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'out/policy_pilot_1657753/best.pt')
    ap.add_argument('--plan', type=Path, default=ROOT/'out/validation_plan_v1.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N windows per split; 0 means all')
    ap.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.limit < 0:
        ap.error('invalid batch size, workers or limit')
    if args.out.exists():
        raise FileExistsError('Choose a new output directory to preserve previous diagnostics')
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    data_cls, model_cls = PolicyWindows, ActionPolicy
    if cfg.get('proprioception', False):
        from droid_qp.policy_proprio import ProprioPolicyWindows, ProprioActionPolicy, PROPRIO_RECIPE
        if cfg['proprioception_recipe'] != PROPRIO_RECIPE:
            raise ValueError('unsupported proprioception recipe')
        data_cls, model_cls = ProprioPolicyWindows, ProprioActionPolicy
    plan = json.loads(args.plan.read_text())
    manifest = ROOT/'out/dataset_manifest.json'
    scales = ROOT/'out/policy_loss_scales.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if cfg['manifest_sha256'] != sha(manifest) or plan['manifest_sha256'] != sha(manifest):
        raise ValueError('manifest differs from checkpoint/validation plan')
    if cfg['validation_plan_sha256'] != sha(args.plan) or cfg['scale_sha256'] != sha(scales):
        raise ValueError('validation protocol or loss scales changed')
    if (cfg['T_win'], cfg['M']) != (plan['T_win'], plan['M']):
        raise ValueError('window configuration differs')
    train = data_cls('train', T_win=cfg['T_win'], M=cfg['M'])
    expected_steps = (len(train)+cfg['batch_size']-1)//cfg['batch_size']
    if (cfg['mode'] != 'train' or ck['epoch'] != 0 or ck['cursor'] != len(train)
            or ck['step'] != expected_steps or cfg['steps'] != expected_steps):
        raise ValueError('This diagnostic requires the completed first-epoch checkpoint; partial/later epochs need another replay plan')
    val = data_cls('val', T_win=cfg['T_win'], M=cfg['M'])
    if {c['episode'] for c in train.clips} & {c['episode'] for c in val.clips}:
        raise ValueError('train/validation episode overlap')
    device = torch.device(args.device)
    model = model_cls(layers=cfg['layers'], M=cfg['M'], scene_slots=cfg['M'],
                         checkpoint=False, dropout=cfg['dropout']).to(device).eval()
    model.load_state_dict(ck['model'])
    criterion = PolicyLoss.from_json(scales, manifest=manifest).to(device)
    criterion.load_state_dict(ck['criterion'])
    datasets = dict(train_seen=SeenWindows(train, cfg['seed']), val=ValidationWindows(val, plan['records']))
    args.out.mkdir(parents=True)
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=ck['step'],
                  proprioception=cfg.get('proprioception', False),
                  training_epoch_replayed=0, seed=cfg['seed'], manifest_sha256=sha(manifest),
                  plan_sha256=sha(args.plan), diagnostic_subset=bool(args.limit),
                  updates_performed=0, amp=args.amp, metrics={})
    for name, dataset in datasets.items():
        if args.limit:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        windows = []
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=device.type == 'cuda')
        def batches():
            for batch in loader:
                windows.extend(dict(video_id=v, start=int(s), episode=e, lab=lab)
                               for v,s,e,lab in zip(batch['video_id'], batch['start'], batch['episode'], batch['lab']))
                if len(windows) % 160 == 0 or len(windows) == len(dataset):
                    print(f'{name}: loaded {len(windows)}/{len(dataset)} windows', flush=True)
                yield batch
        print(f'Evaluating {name}: {len(dataset)} windows; no optimizer or weight updates', flush=True)
        result = evaluate_validation(model, criterion, batches(), device, amp=args.amp)
        for m in [result, *result['per_lab'].values()]:
            m['position_to_copy_ratio'] = m['common_position_mm']/m['copy_position_mm'] if m['copy_position_mm'] else None
        report['metrics'][name] = result
        (args.out/f'{name}_windows.json').write_text(json.dumps(windows, indent=2)+'\n')
        (args.out/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def f(value):
        return 'n/a' if value is None else f'{value:.3f}'
    lines = ['# Seen-training versus held-out validation', '',
             f"Checkpoint update {ck['step']}; **zero training updates performed**. Subset smoke test: {bool(args.limit)}.", '',
             '| Split | Windows | Loss | Position mm | Copy mm | Policy/copy | Rotation deg | Copy deg | Event cost | Persistence cost |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in report['metrics'].items():
        values = [m[k] for k in ['loss', 'common_position_mm', 'copy_position_mm', 'position_to_copy_ratio',
                                 'common_rotation_deg', 'copy_rotation_deg']]
        values += [m['events'][who]['all']['timing_cost_frames'] for who in ['policy','persistence']]
        lines.append(f"| {name} | {m['n_windows']} | "+' | '.join(map(f, values))+' |')
    lines += ['', '| Lab | Train mm | Train copy mm | Train/copy | Val mm | Val copy mm | Val/copy |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for lab in sorted(set(report['metrics']['train_seen']['per_lab']) | set(report['metrics']['val']['per_lab'])):
        values = [report['metrics'][split]['per_lab'].get(lab, {}).get(k) for split in ['train_seen','val']
                  for k in ['common_position_mm','copy_position_mm','position_to_copy_ratio']]
        lines.append(f"| {lab} | "+' | '.join(map(f, values))+' |')
    lines += ['', 'Ratios below 1 beat the copy-previous baseline on that split. Compare against each split’s own baseline and within labs: train and validation motion distributions can differ.',
              '', 'Poor performance on both splits is consistent with insufficient fitting, but does not distinguish too little training from optimization or input limitations. Good training performance with poor validation suggests a generalization gap. These are diagnostics, not new pass/fail thresholds.',
              '', 'Dropout is disabled on both splits. Inputs follow recorded demonstrator trajectories; this does not evaluate closed-loop control. Exact replayed starts and full per-lab/event metrics are saved beside this summary.', '']
    summary = '\n'.join(lines)
    (args.out/'summary.md').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
