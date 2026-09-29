#!/usr/bin/env python3
"""Render pilot decisions, per-lab results and offline rollout curves from artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    report = json.loads((run/'evaluation/report.json').read_text())
    rollout = json.loads((run/'evaluation/rollout.json').read_text())
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    vals = [r for r in rows if r['event'] == 'validation_fixed_windows']
    c, decision = report['clean'], report['pilot_decision']
    e, persistence = c['events']['policy']['all'], c['events']['persistence']['all']
    steps = [r['step'] for r in vals]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [('loss', 'Weighted validation loss'), ('common_position_mm', 'Position error (mm)'),
              ('rotation_deg', 'Rotation error (degrees)'), ('timing', 'Transition timing cost (frames)')]
    for ax, (key, title) in zip(axes.flat, panels):
        y = [r['metrics'][key] if key != 'timing' else r['metrics']['events']['policy']['all']['timing_cost_frames'] for r in vals]
        ax.plot(steps, y, marker='o', label='Policy')
        if key == 'common_position_mm':
            ax.axhline(c['copy_position_mm'], color='tab:orange', linestyle='--', label='Copy previous')
            ax.axhline(decision['thresholds']['position_mm_strictly_below'], color='tab:red', linestyle=':', label='Gate')
        if key == 'rotation_deg':
            ax.axhline(c['copy_rotation_deg'], color='tab:orange', linestyle='--', label='Copy previous')
        if key == 'timing':
            ax.axhline(persistence['timing_cost_frames'], color='tab:orange', linestyle='--', label='Persistence')
        ax.axvline(report['checkpoint_step'], color='gray', alpha=.4, label='Selected checkpoint')
        ax.set(xlabel='Optimizer updates', ylabel=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('One-epoch pilot: recorded-observation validation (not closed-loop)')
    fig.savefig(run/'validation_curve.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    hs = sorted(map(int, rollout['horizons']))
    for name, label in [('policy', 'Policy'), ('recorded_previous', 'Copy each recorded previous action'),
                        ('hold_previous', 'Hold initial previous action'), ('zero', 'Zero action')]:
        for ax, key, title in [(axes[0], 'position_mm', 'Flange position error (mm)'),
                               (axes[1], 'oracle_width/pixel_error', 'Gripper projection error (px; oracle width)')]:
            ax.plot(hs, [rollout['horizons'][str(h)]['aggregate'][name+'/'+key]['mean'] for h in hs], marker='o', label=label)
            ax.set(xlabel='Integrated action horizon (frames)', ylabel=title)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Offline action integration; predictions still use recorded observations')
    fig.savefig(run/'rollout_curve.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    lab_names = sorted(c['per_lab'])
    positions = list(range(len(lab_names)))
    ax.barh([i-.18 for i in positions], [c['per_lab'][lab]['common_position_mm'] for lab in lab_names],
            height=.35, label='Policy')
    ax.barh([i+.18 for i in positions], [c['per_lab'][lab]['copy_position_mm'] for lab in lab_names],
            height=.35, label='Copy previous action')
    ax.set(yticks=positions, yticklabels=lab_names, xlabel='Mean position error (mm; same target mask)',
           title='Held-out prediction by lab at the selected checkpoint')
    ax.legend()
    ax.grid(axis='x', alpha=.2)
    fig.savefig(run/'per_lab_position.png', dpi=170)
    plt.close(fig)

    lines = ['# One-epoch action-decoder pilot', '',
             f"**Frozen gate: {'PASS' if decision['pass_all'] else 'FAIL'}.** Selected checkpoint: update {report['checkpoint_step']}, using lowest weighted validation loss.", '',
             f"{c['n_windows']:,} held-out clip windows from {report['n_validation_episodes']:,} episodes; {c['n_targets']:,} valid targets and {c['n_copy_comparable']:,} copy-comparable targets. All observations follow the demonstrator; these are not closed-loop results.", '',
             'Training samples one 48-frame window from each of 12,123 training clips in this pass. Targets are observed EE pose changes and next observed gripper state; the binary gripper target is not the recorded actuator command.', '',
             '| Predeclared condition | Measured | Required | Result |', '|---|---:|---:|---|',
             f"| Position error | {c['common_position_mm']:.3f} mm | < {decision['thresholds']['position_mm_strictly_below']:.3f} mm | {'PASS' if decision['gates']['position'] else 'FAIL'} |",
             f"| Scene-shuffle position increase | {100*decision['scene_position_relative_increase']:.2f}% | > 10% | {'PASS' if decision['gates']['scene'] else 'FAIL'} |",
             f"| Transition timing cost | {e['timing_cost_frames']:.3f} frames | < {persistence['timing_cost_frames']:.3f} frames | {'PASS' if decision['gates']['transitions'] else 'FAIL'} |", '',
             f"Same-mask baselines: zero translation {c['common_zero_mm']:.3f} mm; copy previous action {c['copy_position_mm']:.3f} mm. Policy rotation {c['common_rotation_deg']:.3f} degrees versus copy previous {c['copy_rotation_deg']:.3f} degrees.", '',
             f"Policy transition matches {e['matched']}/{e['true']}, missed {e['missed']}, extra {e['extra']}. Precision {e['precision']:.3f}, recall {e['recall']:.3f}, F1 {e['f1']:.3f}. Timing cost includes missed and extra events; matched-only timing cannot hide those failures.", '',
             '## Paired memory interventions', '',
             '| Last-frame anchors only | Targets | Position mm | Rotation degrees |', '|---|---:|---:|---:|']
    for name in ['clean_anchor', 'scene_anchor', 'robot_history_anchor']:
        r = report[name]
        lines.append(f"| {name} | {r['n_copy_comparable']} | {r['common_position_mm']:.3f} | {r['common_rotation_deg']:.3f} |")
    lines += ['', report['ablation_caveat'],
              'These interventions measure sensitivity, not proof of object semantics or copying. Robot-history intervention preserves the final current frame; its frozen tokens retain tracking context.', '',
              f"Cross-lab donors: {report['donor_provenance']['cross_lab_total']}; recipient counts {report['donor_provenance']['cross_lab_by_recipient_lab']}. All ILIAD and RPL recipients require cross-lab donors, so interpret their intervention effects with that confound.", '',
              '## Held-out results by lab', '',
              '| Lab | Targets | Policy mm | Copy previous mm | Policy degrees | Copy degrees | Transition cost | Persistence cost |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):
        return 'n/a' if v is None else f'{v:.3f}'
    for lab, r in c['per_lab'].items():
        lines.append(f"| {lab} | {r['n_copy_comparable']} | {r['common_position_mm']:.3f} | {r['copy_position_mm']:.3f} | {r['common_rotation_deg']:.3f} | {r['copy_rotation_deg']:.3f} | {fmt(r['events']['policy']['all']['timing_cost_frames'])} | {fmt(r['events']['persistence']['all']['timing_cost_frames'])} |")
    lines += ['', 'Every reported metric, event direction, baseline and intervention also has a per-lab breakdown in [evaluation/report.json](evaluation/report.json).', '',
              '## Offline integrated-action proxy', '',
              '| Horizon | Origins | Policy mm | Recorded previous mm | Hold previous mm | Zero mm | Policy px (oracle width) | Policy px (binary width) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for h in hs:
        r = rollout['horizons'][str(h)]['aggregate']
        keys = ['policy/position_mm', 'recorded_previous/position_mm', 'hold_previous/position_mm', 'zero/position_mm',
                'policy/oracle_width/pixel_error', 'policy/binary_endpoint_width/pixel_error']
        lines.append(f"| {h} | {r['policy/position_mm']['count']} | "+' | '.join(fmt(r[k]['mean']) for k in keys)+' |')
    missing_pose = c['n_copy_comparable'] - rollout['horizons']['1']['aggregate']['policy/position_mm']['count']
    lines += ['', f'Rollout evaluation additionally requires a stored successor pose: {missing_pose} one-step targets with an action label but no stored successor pose are excluded from every rollout method.']
    lines += ['', *[f'- {v}' for v in rollout['limitations']], '',
              f"Geometry checks: maximum saved-FK reprojection disagreement {rollout['qa']['max_saved_fk_reprojection_px']:.6f} px; ground-truth delta composition disagreement {rollout['qa']['max_ground_truth_composition_mm']:.6f} mm. Per-lab horizons, point counts and behind-camera rates: [evaluation/rollout.json](evaluation/rollout.json).", '',
              '![Validation curves](validation_curve.png)', '', '![Position error by lab](per_lab_position.png)', '',
              '![Offline rollout curves](rollout_curve.png)', '',
              'No extended run was started. '+('The frozen conditions passed; a longer run can now be considered.' if decision['pass_all'] else 'Investigate the failed gates before extending training; memorizing ten windows does not establish held-out policy quality.'), '']
    (run/'PILOT_REPORT.md').write_text('\n'.join(lines))
    print(run/'PILOT_REPORT.md')


if __name__ == '__main__':
    main()
