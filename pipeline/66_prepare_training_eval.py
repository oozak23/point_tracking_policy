#!/usr/bin/env python3
"""Freeze the epoch-0 windows recovered by the completed train/val diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic', type=Path, default=DATA / 'policy_fit_diagnostic_1658338')
    p.add_argument('--out', type=Path, default=DATA / 'training_eval_plan_v1.json')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Fixed training evaluation plan already exists; refusing to replace it')
    manifest = DATA / 'dataset_manifest.json'
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    comparison = json.loads((args.diagnostic/'comparison.json').read_text())
    if (comparison['manifest_sha256'] != manifest_sha or comparison['diagnostic_subset']
            or comparison['training_epoch_replayed'] != 0 or comparison['updates_performed'] != 0):
        raise ValueError('need a full, unmodified epoch-0 diagnostic')
    clips = [c for c in json.loads(manifest.read_text())['clips'] if c['split'] == 'train']
    saved = json.loads((args.diagnostic/'train_seen_windows.json').read_text())
    if len(saved) != len(clips) or len({r['video_id'] for r in saved}) != len(clips):
        raise ValueError('training windows are incomplete or duplicated')
    index = {c['video_id']:i for i,c in enumerate(clips)}
    rows = []
    for row in saved:
        i = index[row['video_id']]
        if row['episode'] != clips[i]['episode'] or row['start'] < 0:
            raise ValueError('invalid training window')
        rows.append(dict(row, index=i))
    cfg = json.loads((Path(comparison['checkpoint']).parent/'config.json').read_text())
    result = dict(schema_version=1, split='train', manifest_sha256=manifest_sha,
        T_win=cfg['T_win'], M=cfg['M'], seed=comparison['seed'], training_epoch_replayed=0,
        source_diagnostic=str(args.diagnostic.resolve()),
        source_windows_sha256=hashlib.sha256((args.diagnostic/'train_seen_windows.json').read_bytes()).hexdigest(),
        records=rows)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Frozen {len(rows)} training windows: {args.out}')


if __name__ == '__main__':
    main()
