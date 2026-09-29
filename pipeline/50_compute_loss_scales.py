#!/usr/bin/env python3
"""Compute step-5 loss scales from all valid, quality-weighted training labels."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_loss import compute_loss_scales


from droid_qp.paths import DATA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DATA / 'dataset_manifest.json')
    parser.add_argument('--gen', type=Path, default=DATA / 'gen')
    parser.add_argument('--shard-index', type=Path, default=DATA / 'shard_index.json')
    parser.add_argument('--out', type=Path, default=DATA / 'policy_loss_scales.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = compute_loss_scales(args.manifest, args.gen, args.shard_index,
                                 progress=lambda i, n: print(f'{i}/{n} training clips', flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
