#!/usr/bin/env python3
"""Compare the one-epoch proprioception pilot with the original visual-only pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,default=ROOT/'out/policy_pilot_1657753')
    ap.add_argument('--baseline-fit',type=Path,default=ROOT/'out/policy_fit_diagnostic_1658338')
    args=ap.parse_args()
    read=lambda p:json.loads(p.read_text())
    cfg,bcfg=read(args.run/'config.json'),read(args.baseline/'config.json')
    if not cfg.get('proprioception') or bcfg.get('proprioception',False):
        raise ValueError('need a proprioception variant and visual-only baseline')
    for k in ['mode','steps','batch_size','seed','lr','warmup','T_win','M','layers','amp',
              'dropout','weight_decay','optimizer_recipe','manifest_sha256','scale_sha256',
              'validation_plan_sha256','eval_every']:
        if cfg[k]!=bcfg[k]:
            raise ValueError(f'comparison recipe mismatch: {k}')
    variant,baseline=read(args.run/'evaluation/report.json'),read(args.baseline/'evaluation/report.json')
    fit,bfit=read(args.run/'end_fit/comparison.json'),read(args.baseline_fit/'comparison.json')
    if not variant.get('proprioception') or not fit.get('proprioception') or baseline.get('proprioception',False) or bfit.get('proprioception',False):
        raise ValueError('evaluation artifacts have the wrong variant identity')
    for r in [fit,bfit]:
        if r['diagnostic_subset'] or r['checkpoint_step']!=758 or r['manifest_sha256']!=cfg['manifest_sha256']:
            raise ValueError('need complete epoch-end training/validation diagnostics')
    for filename in ['train_seen_windows.json','val_windows.json']:
        if read(args.run/'end_fit'/filename)!=read(args.baseline_fit/filename):
            raise ValueError('diagnostic evaluated different windows')
    def values(m):
        return [m['loss'],m['common_position_mm'],m['common_rotation_deg'],
                m['events']['policy']['all']['timing_cost_frames']]
    fmt=lambda x:'n/a' if x is None else f'{x:.3f}'
    lines=['# One-epoch proprioception comparison','',
           'Same one-epoch budget (758 updates), seed, window sampling, optimizer/loss recipe, dropout and one-epoch cosine schedule. Only the variant adds current q[7], recorded qdot[7], and gripper closure[1] through a late-fusion MLP.','',
           '| Epoch-end metric | Visual train | State+visual train | Visual val | State+visual val |',
           '|---|---:|---:|---:|---:|']
    models=[bfit['metrics']['train_seen'],fit['metrics']['train_seen'],bfit['metrics']['val'],fit['metrics']['val']]
    for i,label in enumerate(['Weighted loss','Position mm','Rotation degrees','Transition cost (frames)']):
        lines.append(f'| {label} | '+' | '.join(fmt(values(m)[i]) for m in models)+' |')
    lines+=['',f"Best validation-loss checkpoints: visual update {baseline['checkpoint_step']}, proprioception update {variant['checkpoint_step']}.",'',
            '| Best-checkpoint test | Visual only | State + visual |','|---|---:|---:|']
    for label,key in [('Position mm','common_position_mm'),('Rotation degrees','common_rotation_deg'),('Copy position baseline mm','copy_position_mm')]:
        lines.append(f"| {label} | {fmt(baseline['clean'][key])} | {fmt(variant['clean'][key])} |")
    for label,key in [('Scene-shuffle increase %','scene_position_relative_increase')]:
        lines.append(f"| {label} | {100*baseline['pilot_decision'][key]:.2f} | {100*variant['pilot_decision'][key]:.2f} |")
    lines+=['', '| Paired anchor position error (mm) | Visual only | State + visual |', '|---|---:|---:|']
    for key in ['clean_anchor','scene_anchor','robot_history_anchor']:
        lines.append(f"| {key} | {fmt(baseline[key]['common_position_mm'])} | {fmt(variant[key]['common_position_mm'])} |")
    if 'proprio_shuffle' in variant:
        lines+=['',f"Proprioception-shuffle full-window error: {variant['proprio_shuffle']['common_position_mm']:.3f} mm (clean {variant['clean']['common_position_mm']:.3f} mm)."]
    lines+=['', '| Lab, epoch-end val | Visual mm | State+visual mm | Copy mm |', '|---|---:|---:|---:|']
    for lab,b in sorted(bfit['metrics']['val']['per_lab'].items()):
        v=fit['metrics']['val']['per_lab'][lab]
        lines.append(f"| {lab} | {b['common_position_mm']:.3f} | {v['common_position_mm']:.3f} | {v['copy_position_mm']:.3f} |")
    lines+=['',f"Original frozen gates on the proprioception best checkpoint: {variant['pilot_decision']['gates']}.",
            '', 'Current measured state is retained during scene/robot-history shuffles. The history intervention now tests visual history, not removal of all motion information. State shuffling breaks visual/state compatibility, so it is a sensitivity diagnostic, not a unique causal explanation.',
            '', 'The MLP adds 69,888 parameters at D=256. Shared weights and initial predictions match the visual baseline, but there is no matched-capacity control. This is a single-seed, one-epoch comparison; improvements or failures do not establish eventual performance.',
            '', 'Metrics use recorded demonstrator observations and observed next-state targets, not closed-loop task success. Detailed losses, baselines and events by lab are in end_fit/comparison.json and evaluation/report.json.', '']
    (args.run/'proprio_comparison.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,run in [('Visual only',args.baseline),('State + visual',args.run)]:
        rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r['event']=='validation_fixed_windows']
        steps=[r['step'] for r in rows]
        for ax,ys,title in [(axes[0],[r['metrics']['loss'] for r in rows],'Validation loss'),
                            (axes[1],[r['metrics']['common_position_mm'] for r in rows],'Position error (mm)'),
                            (axes[2],[r['metrics']['events']['policy']['all']['timing_cost_frames'] for r in rows],'Transition cost (frames)')]:
            ax.plot(steps,ys,marker='o',label=name);ax.set(xlabel='Optimizer updates',ylabel=title);ax.grid(alpha=.2)
    axes[1].axhline(baseline['clean']['copy_position_mm'],linestyle='--',color='gray',label='Copy previous')
    axes[2].axhline(baseline['clean']['events']['persistence']['all']['timing_cost_frames'],linestyle='--',color='gray',label='Persistence')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(args.run/'proprio_comparison.png',dpi=170);plt.close(fig)
    print(args.run/'proprio_comparison.md',flush=True)


if __name__=='__main__':
    main()
