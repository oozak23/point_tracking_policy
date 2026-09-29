#!/usr/bin/env python3
"""Backfill/follow policy JSONL metrics in W&B without touching the training process."""
import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path


def flatten(value, prefix):
    out = {}
    if isinstance(value, dict):
        for k,v in value.items():
            out.update(flatten(v, f'{prefix}/{k}'))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        out[prefix] = value
    return out


def payload(event):
    prefixes = {'train':'train_batch', 'training_fixed_windows':'train_fixed',
                'validation_fixed_windows':'val', 'fit':'overfit'}
    prefix = prefixes.get(event['event'], event['event'])
    result = {'optimizer_step':event['step']}
    if 'metrics' in event:
        result.update(flatten(event['metrics'], prefix))
    else:
        result.update(flatten(event.get('losses', {}), prefix))
        for key in ['lr','grad_norm_before_clip','peak_gpu_gb']:
            result.update(flatten(event.get(key), f'{prefix}/{key}'))
    result.update(flatten(event.get('elapsed_s'), 'recorded_elapsed_s'))
    return result


def complete_events(stream):
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line or not line.endswith('\n'):
            stream.seek(offset)
            return
        yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--project', default='action-decoder')
    p.add_argument('--entity', help='optional W&B team/account')
    p.add_argument('--follow', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--offline', action='store_true', help='write local W&B data only')
    args = p.parse_args()
    if args.poll_seconds <= 0:
        p.error('poll interval must be positive')
    args.run = args.run.resolve()
    config = json.loads((args.run/'config.json').read_text())
    state = args.run/'wandb_monitor'
    state.mkdir(exist_ok=True)
    lock = (state/'monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A W&B monitor is already running for this experiment')
    identity_path = state/'identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    if identity and (identity['project'] != args.project or
                     (args.entity and identity['entity'] != args.entity)):
        raise ValueError('Use the same W&B project/entity to resume this monitor')
    import wandb
    settings = dict(disable_git=True, console='off')
    available = wandb.Settings()
    for option in ['_disable_stats','_disable_meta','x_disable_stats']:
        if hasattr(available, option):
            settings[option] = True
    keys = ['mode','steps','batch_size','seed','lr','warmup','T_win','M','layers','amp',
            'dropout','weight_decay','optimizer_recipe','manifest_sha256',
            'validation_plan_sha256','train_eval_plan_sha256','proprioception','proprioception_recipe']
    wb = wandb.init(project=args.project, entity=identity.get('entity', args.entity),
        id=hashlib.sha256(str(args.run).encode()).hexdigest()[:16], name=args.run.name,
        resume=None if args.offline else 'allow', mode='offline' if args.offline else 'online',
        dir=str(state), save_code=False, settings=wandb.Settings(**settings),
        config={**{k:config[k] for k in keys if k in config}, 'source':'existing policy JSONL metrics'})
    if not args.offline:
        identity_path.write_text(json.dumps(dict(project=wb.project,entity=wb.entity,id=wb.id,url=wb.url),indent=2)+'\n')
        print('Dashboard:',wb.url,flush=True)
    wb.define_metric('optimizer_step')
    wb.define_metric('*', step_metric='optimizer_step')
    # Source-record index is the monotonically increasing W&B history step. Train
    # and validation can share an optimizer step without dropping either record.
    next_step = int(wb.step)
    source_index = 0
    try:
        with (args.run/'metrics.jsonl').open() as stream:
            while True:
                logged = 0
                for event in complete_events(stream):
                    if source_index >= next_step:
                        wb.log(payload(event), step=source_index)
                        logged += 1
                    source_index += 1
                if logged:
                    print(f'Synced {logged} records; read {source_index} source records',flush=True)
                if not args.follow:
                    break
                summary_path = args.run/'summary.json'
                try:
                    summary = json.loads(summary_path.read_text())
                except (FileNotFoundError,json.JSONDecodeError):
                    summary = {}
                if summary.get('status') in ('budget_finished','overfit_passed'):
                    if stream.tell() < (args.run/'metrics.jsonl').stat().st_size:
                        continue  # Drain final records written between EOF and summary read.
                    print('Training completed; metrics synchronized.',flush=True)
                    break
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print('Stopping metric monitor; the training process is unaffected.',flush=True)
    finally:
        wb.finish()
        lock.close()


if __name__ == '__main__':
    main()
