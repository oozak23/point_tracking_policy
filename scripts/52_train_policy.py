#!/usr/bin/env python3
"""Step 6: 10-window overfit gate first; full-split training is a separate mode."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_loss import PolicyLoss
from droid_qp.policy_eval import ValidationWindows, evaluate_validation
from droid_qp.policy_train import (ActionPolicy, TrainingWindows, evaluate, lr_factor, make_optimizer,
                                  save_checkpoint, select_overfit_windows, to_device, train_step)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('overfit', 'train'), default='overfit')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--clips', type=int, default=10)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--val-clips', type=int, default=64)
    p.add_argument('--validation-plan', type=Path)
    p.add_argument('--train-eval-plan', type=Path, help='optional fixed training windows evaluated alongside validation')
    p.add_argument('--proprioception', action='store_true', help='add current q, recorded qdot and gripper state at the readout')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--T-win', type=int, default=48)
    p.add_argument('--M', type=int, default=24)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    data_cls, model_cls = PolicyWindows, ActionPolicy
    if args.proprioception:
        from droid_qp.policy_proprio import ProprioPolicyWindows, ProprioActionPolicy, PROPRIO_RECIPE
        data_cls, model_cls = ProprioPolicyWindows, ProprioActionPolicy
    if min(args.steps, args.batch_size, args.clips, args.eval_every, args.val_clips, args.T_win, args.M, args.layers) <= 0:
        p.error('counts must be positive')
    if args.workers < 0 or args.lr <= 0 or args.warmup < 0 or args.warmup > args.steps:
        p.error('invalid workers, learning rate or warmup')
    if args.mode == 'train' and args.clips != 10:
        p.error('--clips selects overfit examples only; train mode uses the full split')
    if args.train_eval_plan and args.mode != 'train':
        p.error('--train-eval-plan requires train mode')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; submit the GPU sbatch or explicitly use --device cpu')
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(), flush=True)
        if args.amp and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 requires supported hardware; use --no-amp otherwise')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'config.json').exists() and args.resume is None:
        raise FileExistsError('run directory already has a config; choose a new directory or --resume')
    manifest = ROOT / 'out/dataset_manifest.json'
    scales = ROOT / 'out/policy_loss_scales.json'
    criterion = PolicyLoss.from_json(scales, manifest=manifest).to(device)
    data = data_cls(T_win=args.T_win, M=args.M, cache_clips=max(args.clips, 8), seed=args.seed)
    fixed = select_overfit_windows(data, args.clips, args.seed) if args.mode == 'overfit' else None
    dataset = TrainingWindows(data, fixed=fixed, seed=args.seed)
    if fixed is not None:
        eval_data = dataset
        eval_name = 'fit'
    else:
        val = data_cls('val', T_win=args.T_win, M=args.M, seed=args.seed)
        if args.validation_plan is None:
            p.error('train mode requires a frozen --validation-plan')
        plan = json.loads(args.validation_plan.read_text())
        if (plan['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest()
                or plan['T_win'] != args.T_win or plan['M'] != args.M):
            raise ValueError('validation plan does not match data/window configuration')
        records = plan['records']
        eval_data = ValidationWindows(val, records)
        fixed = records
        eval_name = 'validation_fixed_windows'
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers if args.mode == 'train' else 0,
                             generator=torch.Generator().manual_seed(args.seed))
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg.update(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
               scale_sha256=hashlib.sha256(scales.read_bytes()).hexdigest(),
               fixed_windows=fixed, evaluation=eval_name,
               dropout=0. if args.mode == 'overfit' else 0.1,
               weight_decay=0. if args.mode == 'overfit' else 1e-4,
               optimizer_recipe='adamw_physical_head_lr_v1',
               torch_version=torch.__version__)
    if args.validation_plan:
        cfg['validation_plan_sha256'] = hashlib.sha256(args.validation_plan.read_bytes()).hexdigest()
    if args.proprioception:
        cfg['proprioception_recipe'] = PROPRIO_RECIPE
    train_eval_loader = None
    if args.train_eval_plan:
        train_plan = json.loads(args.train_eval_plan.read_text())
        if (train_plan['manifest_sha256'] != cfg['manifest_sha256'] or train_plan['split'] != 'train'
                or train_plan['T_win'] != args.T_win or train_plan['M'] != args.M
                or train_plan['seed'] != args.seed or train_plan['training_epoch_replayed'] != 0):
            raise ValueError('fixed training evaluation plan differs from training configuration')
        for row in train_plan['records']:
            if data.clips[row['index']]['video_id'] != row['video_id']:
                raise ValueError('training evaluation clip index mismatch')
        train_eval_data = data_cls('train', T_win=args.T_win, M=args.M, cache_clips=8)
        train_eval_loader = DataLoader(ValidationWindows(train_eval_data, train_plan['records']),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            generator=torch.Generator().manual_seed(args.seed))
        cfg['train_eval_plan_sha256'] = hashlib.sha256(args.train_eval_plan.read_bytes()).hexdigest()
    model = model_cls(layers=args.layers, M=args.M, scene_slots=args.M,
                         checkpoint=args.checkpoint, dropout=cfg['dropout']).to(device)
    print(f'{sum(v.numel() for v in model.parameters()):,} trainable parameters; '
          f'{len(dataset)} training windows/epoch; evaluation={eval_name}', flush=True)
    optimizer = make_optimizer(model, criterion, lr=args.lr, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda s: lr_factor(s, args.steps, args.warmup))
    step = epoch = cursor = 0
    best = float('inf')
    initial = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ('mode', 'steps', 'batch_size', 'seed', 'lr', 'warmup', 'T_win', 'M',
                    'layers', 'amp', 'optimizer_recipe', 'manifest_sha256', 'scale_sha256', 'fixed_windows'):
            if ck['config'][key] != cfg[key]:
                raise ValueError(f'resume configuration mismatch: {key}')
        for key in ('validation_plan_sha256', 'train_eval_plan_sha256'):
            if ck['config'].get(key) != cfg.get(key):
                raise ValueError(f'resume evaluation plan mismatch: {key}')
        if ck['config'].get('proprioception', False) != args.proprioception or ck['config'].get('proprioception_recipe') != cfg.get('proprioception_recipe'):
            raise ValueError('resume proprioception configuration mismatch')
        model.load_state_dict(ck['model'])
        criterion.load_state_dict(ck['criterion'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        step, epoch, cursor, best = ck['step'], ck['epoch'], ck['cursor'], ck['best_loss']
        initial = ck['initial']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    if not (args.out / 'config.json').exists():
        (args.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print('configuration:', json.dumps({k: v for k, v in cfg.items() if k != 'fixed_windows'}), flush=True)
    started = time.monotonic()

    def record(event):
        with (args.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
        print(json.dumps(event, allow_nan=False), flush=True)

    def assess():
        nonlocal best, initial
        eval_fn = evaluate_validation if args.mode == 'train' else evaluate
        metrics = eval_fn(model, criterion, eval_loader, device, amp=args.amp)
        if initial is None:
            initial = metrics
        improved = metrics['loss'] < best
        best = min(best, metrics['loss'])
        gate = (args.mode == 'overfit' and metrics['position'] < 0.02 and
                metrics['rotation'] < 0.02 and metrics['gripper'] < 0.02 and
                metrics['loss'] < 0.05 and metrics['gripper_accuracy'] >= 0.995 and
                metrics['transition_frame_accuracy'] == 1.)
        event = {'event': eval_name, 'step': step, 'epoch': epoch,
                 'elapsed_s': time.monotonic() - started, 'metrics': metrics, 'overfit_gate': gate}
        record(event)
        train_metrics = None
        if train_eval_loader is not None:
            print('Evaluating fixed epoch-0 training windows...', flush=True)
            train_metrics = evaluate_validation(model, criterion, train_eval_loader, device, amp=args.amp)
            record(dict(event='training_fixed_windows', step=step, epoch=epoch,
                        elapsed_s=time.monotonic()-started, metrics=train_metrics))
        state = dict(config=cfg, model=model.state_dict(), criterion=criterion.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     step=step, epoch=epoch, cursor=cursor, best_loss=best, initial=initial,
                     metrics=metrics, train_metrics=train_metrics, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        save_checkpoint(args.out / 'last.pt', state)
        if improved:
            save_checkpoint(args.out / 'best.pt', state)
        summary = dict(status='overfit_passed' if gate else ('budget_finished' if step >= args.steps else 'running'),
                       step=step, best_loss=best, initial=initial, latest=metrics,
                       train_latest=train_metrics, config=cfg)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return gate

    if assess():
        return
    skipped = 0
    while step < args.steps:
        dataset.epoch = epoch
        order = np.random.default_rng([args.seed, epoch]).permutation(len(dataset)).tolist()
        if cursor >= len(order):
            epoch += 1
            cursor = 0
            continue
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=order[cursor:],
                            num_workers=args.workers, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        for batch in loader:
            cursor += len(batch['target'])
            model.train()
            batch = to_device(batch, device)
            used_lr = optimizer.param_groups[0]['lr']
            terms, grad_norm = train_step(model, criterion, batch, optimizer, amp=args.amp)
            if grad_norm is None:
                skipped += 1
                if skipped >= len(dataset):
                    raise RuntimeError('no supervised batches; refusing to loop without updates')
                continue
            skipped = 0
            step += 1
            scheduler.step()
            if step == 1 or step % 10 == 0:
                record({'event': 'train', 'step': step, 'epoch': epoch,
                        'elapsed_s': time.monotonic() - started,
                        'lr': used_lr, 'grad_norm_before_clip': grad_norm,
                        'losses': {k: float(terms[k].detach()) for k in ('loss', 'position', 'rotation', 'gripper')},
                        'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None})
            if step % args.eval_every == 0 or step == args.steps:
                if assess():
                    return
            if step >= args.steps:
                break
        if cursor >= len(order):
            epoch += 1
            cursor = 0


if __name__ == '__main__':
    main()
