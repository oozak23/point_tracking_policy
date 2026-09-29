#!/usr/bin/env python3
"""Paired fixed-window learning curves and the unchanged pilot gates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.run/'config.json').read_text())
    rows = [json.loads(s) for s in (args.run/'metrics.jsonl').read_text().splitlines()]
    paired = {name:{r['step']:r['metrics'] for r in rows if r['event'] == event}
              for name,event in [('train', 'training_fixed_windows'), ('val', 'validation_fixed_windows')]}
    steps = sorted(paired['train'].keys() & paired['val'].keys())
    if not steps:
        raise ValueError('no completed paired evaluations yet')
    best_step = min(paired['val'], key=lambda s: paired['val'][s]['loss'])
    lines = ['# Five-epoch fitting experiment', '',
             f"Fresh initialization, seed {cfg['seed']}; cosine schedule over {cfg['steps']} updates. Best validation-loss checkpoint: update {best_step}.", '',
             'Training uses newly sampled windows each epoch; evaluation always uses the same epoch-0 training windows and frozen validation windows. At initialization the training windows have not yet been seen.', '',
             '| Update | Train loss | Val loss | Train mm | Train copy mm | Val mm | Val copy mm | Train event cost | Val event cost |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in steps:
        t,v = paired['train'][s],paired['val'][s]
        nums = [t['loss'],v['loss'],t['common_position_mm'],t['copy_position_mm'],
                v['common_position_mm'],v['copy_position_mm'],
                t['events']['policy']['all']['timing_cost_frames'], v['events']['policy']['all']['timing_cost_frames']]
        lines.append(f'| {s} | '+' | '.join('n/a' if x is None else f'{x:.3f}' for x in nums)+' |')
    latest = steps[-1]
    lines += ['', f'Per-lab position comparison at the latest paired evaluation (update {latest}):', '',
              '| Lab | Train mm | Train copy mm | Val mm | Val copy mm |', '|---|---:|---:|---:|---:|']
    for lab,t in sorted(paired['train'][latest]['per_lab'].items()):
        v = paired['val'][latest]['per_lab'][lab]
        nums = [t['common_position_mm'],t['copy_position_mm'],v['common_position_mm'],v['copy_position_mm']]
        lines.append(f'| {lab} | '+' | '.join(f'{x:.3f}' for x in nums)+' |')
    evaluation = args.run/'evaluation/report.json'
    if evaluation.exists():
        r = json.loads(evaluation.read_text())
        d = r['pilot_decision']
        lines += ['', f"Best-checkpoint frozen pilot gates: {d['gates']}; all passed: {d['pass_all']}.",
                  f"Full-window scene-shuffle position increase: {100*d['scene_position_relative_increase']:.2f}%.",
                  f"Paired anchor errors (mm): clean {r['clean_anchor']['common_position_mm']:.3f}, scene shuffle {r['scene_anchor']['common_position_mm']:.3f}, robot-history shuffle {r['robot_history_anchor']['common_position_mm']:.3f}."]
    else:
        lines += ['', 'Post-training memory ablations have not completed yet.']
    lines += ['', 'Compare each split against its own baseline and within labs. Similar poor train/validation performance does not identify whether optimization, inputs, or training duration is the limiting factor.',
              'The longer cosine schedule changes learning rates relative to the one-epoch pilot; this is a larger optimization-budget experiment, not an exact continuation or a duration-only control.',
              'These are recorded-observation predictions, not closed-loop task-success measurements. No run beyond this budget is launched automatically.', '',
              'All per-lab losses, rotation metrics and event counts at every evaluation are in metrics.jsonl. Final ablations and their per-lab breakdowns are in evaluation/report.json.', '']
    (args.run/'experiment_summary.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for split in ('train','val'):
        ms = [paired[split][s] for s in steps]
        for ax,ys,title in [(axes[0],[m['loss'] for m in ms],'Weighted loss'),
                            (axes[1],[m['common_position_mm'] for m in ms],'Position error (mm)'),
                            (axes[2],[m['events']['policy']['all']['timing_cost_frames'] for m in ms],'Transition cost (frames)')]:
            ax.plot(steps,ys,marker='o',label=split)
            ax.set(xlabel='Optimizer updates',ylabel=title)
            ax.grid(alpha=.2)
        axes[1].axhline(ms[-1]['copy_position_mm'],linestyle='--',label=f'{split} copy baseline')
        axes[2].axhline(ms[-1]['events']['persistence']['all']['timing_cost_frames'],linestyle='--',label=f'{split} persistence')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(args.run/'train_val_curve.png',dpi=170)
    plt.close(fig)
    print(args.run/'experiment_summary.md',flush=True)


if __name__ == '__main__':
    main()
