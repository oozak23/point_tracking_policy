#!/usr/bin/env python3
"""Does the policy use the scene WHERE IT MATTERS? Scene-shuffle, stratified by difficulty.

The one-epoch proprioception run reported a scene-shuffle effect of only +2.1% averaged
over every validation frame. But predicting 0.14 s ahead is dominated by momentum: on
steady-motion frames the right answer is "keep going" and the scene cannot improve it. The
scene can only matter at DECISION frames -- where the motion changes. An average over all
frames therefore dilutes any real effect with thousands of frames where none is possible.

So stratify. The difficulty of a frame is how badly "repeat the previous action" does on
it -- |a_t - a_{t-1}|, the change in the demonstrator's action. That is a property of the
recording, not of the model, so it is a fair axis to split on.

    scene effect large on hard frames  -> the model uses the scene where it counts;
                                          the +2.1% gate was measuring the wrong thing
    scene effect small even there      -> the model genuinely ignores the scene

REPRODUCTION FIRST. Same checkpoint, same frozen validation plan, same DONOR per window as
57_evaluate_policy.py, same unweighted per-frame metric. Aggregated over all valid frames
this must reproduce the reported clean / scene-shuffle / proprio-shuffle numbers, or the
stratified ones mean nothing -- they would be measuring a different intervention.

Per-frame arrays are saved, so the stratification can be redone without re-running the
model. Error bars are a CLUSTER bootstrap over windows: frames within a window are
correlated, and treating 63k frames as independent would make every interval too narrow.

    python evaluation/75_stratified_scene_shuffle.py --checkpoint out/policy_proprio_1658371/best.pt
Run with $PY_TRACKON
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_eval import ValidationWindows
from droid_qp.policy_train import ActionPolicy, to_device


def expected_from_report(checkpoint: Path) -> dict:
    """The numbers 57_evaluate_policy.py reported for THIS checkpoint.

    Read from the run's own evaluation/report.json rather than hard-coded: a hard-coded
    reference silently tests the wrong checkpoint the moment the script is pointed at a
    different run. `proprio` is None for a visual-only model.
    """
    r = json.loads((checkpoint.parent / 'evaluation/report.json').read_text())
    t = r.get('totals', r)
    return {'clean': t['clean']['position_mm'], 'scene': t['scene_shuffle']['position_mm'],
            'proprio': t.get('proprio_shuffle', {}).get('position_mm'),
            'n': t['clean']['n_targets']}


from droid_qp.paths import DATA, OUT  # noqa: E402


def collect(args):
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    data_cls, model_cls = PolicyWindows, ActionPolicy
    if cfg.get('proprioception', False):
        from droid_qp.policy_proprio import ProprioPolicyWindows, ProprioActionPolicy
        data_cls, model_cls = ProprioPolicyWindows, ProprioActionPolicy
    manifest = DATA / 'dataset_manifest.json'
    plan = json.loads(args.plan.read_text())
    if plan['manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('manifest differs from frozen validation plan')
    model = model_cls(layers=cfg['layers'], M=cfg['M'], scene_slots=cfg['M'],
                      checkpoint=False, dropout=cfg['dropout']).to(device).eval()
    model.load_state_dict(ck['model'])
    data = data_cls('val', T_win=plan['T_win'], M=plan['M'], cache_clips=8)
    dataset = ValidationWindows(data, plan['records'])
    # shuffle=False and the SAME batch size as 57, so `offset` addresses the same donor rows
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False)

    rows = {k: [] for k in ('clean', 'scene', 'proprio', 'copy', 'valid', 'common',
                            'transition', 'window', 'lab')}
    offset = 0
    has_proprio = cfg.get('proprioception', False)
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            records = plan['records'][offset:offset + len(batch['video_id'])]
            donors = to_device(default_collate([dataset[r['donor_row']] for r in records]),
                               device)

            def err(b):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg['amp']):
                    p = model(b)
                return (torch.linalg.vector_norm(p['delta_pos_body'][:, :, 0].float()
                                                 - batch['target'][..., :3].float(), dim=-1)
                        * 1000)

            rows['clean'].append(err(batch).cpu())
            rows['scene'].append(err(dict(batch, scene=donors['scene'])).cpu())
            rows['proprio'].append(err(dict(batch, proprio=donors['proprio'])).cpu()
                                   if has_proprio else torch.full_like(rows['clean'][-1], float('nan')))
            y = batch['target'].float()
            rows['copy'].append((torch.linalg.vector_norm(
                batch['previous_action'][..., :3].float() - y[..., :3], dim=-1) * 1000).cpu())
            valid = batch['target_valid'] & (batch['weight'][:, None] > 0)
            rows['valid'].append(valid.cpu())
            rows['common'].append((valid & batch['previous_valid']).cpu())
            rows['transition'].append((y[..., 9].bool() != batch['gripper_current'].bool()).cpu())
            n, T = valid.shape
            rows['window'].append(torch.arange(offset, offset + n)[:, None].expand(n, T).clone())
            rows['lab'].extend([[lab] * T for lab in batch['lab']])
            offset += n
            if offset % 320 == 0 or offset == len(dataset):
                print(f'{offset}/{len(dataset)} windows', flush=True)

    out = {k: torch.cat(v).numpy() for k, v in rows.items() if k != 'lab'}
    out['lab'] = np.array(rows['lab'])
    np.savez(args.out / 'per_frame.npz', **out)
    return out


def cluster_bootstrap(clean, shuffled, window, n_boot=1000, seed=0):
    """% increase with a 95% interval, resampling WINDOWS rather than frames."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(window, return_inverse=True)
    c = np.bincount(inv, weights=clean, minlength=len(ids))
    s = np.bincount(inv, weights=shuffled, minlength=len(ids))
    k = np.bincount(inv, minlength=len(ids)).astype(float)
    point = (s.sum() / c.sum() - 1) * 100
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        draws.append((s[pick].sum() / c[pick].sum() - 1) * 100)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def analyse(d, out_dir, EXPECTED, title):
    v, cm = d['valid'].ravel(), d['common'].ravel()
    clean, scene, prop = d['clean'].ravel(), d['scene'].ravel(), d['proprio'].ravel()
    copy, win = d['copy'].ravel(), d['window'].ravel()
    trans = d['transition'].ravel()
    lines = []
    say = lambda s='': (print(s), lines.append(s))

    # ---- reproduction ---------------------------------------------------------------
    say('## 1. Reproduction of the reported ablation (all valid frames)\n')
    say('| | reported | reproduced |')
    say('|---|---:|---:|')
    rep = {'clean': clean[v].mean(), 'scene': scene[v].mean(), 'proprio': prop[v].mean()}
    ok = int(v.sum()) == EXPECTED['n']
    for k in ('clean', 'scene', 'proprio'):
        if EXPECTED[k] is None:          # visual-only checkpoint: no proprioception to shuffle
            continue
        diff = abs(rep[k] - EXPECTED[k])
        ok &= diff < 0.02
        say(f'| {k} position mm | {EXPECTED[k]:.4f} | {rep[k]:.4f} |')
    say(f'| valid frames | {EXPECTED["n"]:,} | {int(v.sum()):,} |')
    inc = (rep['scene'] / rep['clean'] - 1) * 100
    say(f'\nScene-shuffle increase reproduced: **{inc:+.2f}%** '
        f'(reported {(EXPECTED["scene"] / EXPECTED["clean"] - 1) * 100:+.2f}%). '
        f'**{"REPRODUCED" if ok else "DOES NOT REPRODUCE -- stratified results below are not comparable"}**\n')

    # ---- stratified by copy-last difficulty --------------------------------------------
    say('## 2. Scene-shuffle effect, stratified by how badly copy-last does\n')
    say('Frames split into 5 bins of equal size by |a_t − a_{t−1}| (the change in the '
        "demonstrator's action). Bin 5 is where the motion changes most.\n")
    say('| difficulty | frames | copy-last mm | clean mm | scene-shuffled mm | '
        'scene effect | 95% CI | proprio effect |')
    say('|---|---:|---:|---:|---:|---:|---|---:|')
    edges = np.quantile(copy[cm], np.linspace(0, 1, 6))
    strata = []
    for i in range(5):
        m = cm & (copy >= edges[i]) & (copy <= edges[i + 1] if i == 4 else copy < edges[i + 1])
        p, lo, hi = cluster_bootstrap(clean[m], scene[m], win[m])
        pp = (prop[m].mean() / clean[m].mean() - 1) * 100 if EXPECTED['proprio'] else float('nan')
        strata.append((i + 1, p, lo, hi))
        label = ['1 easiest', '2', '3', '4', '5 hardest'][i]
        say(f'| {label} | {int(m.sum()):,} | {copy[m].mean():.2f} | {clean[m].mean():.2f} | '
            f'{scene[m].mean():.2f} | **{p:+.1f}%** | [{lo:+.1f}, {hi:+.1f}] | {pp:+.0f}% |')

    # ---- gripper transitions -----------------------------------------------------------
    say('\n## 3. Around gripper transitions\n')
    say('A second definition of a decision frame: within ±2 frames of the gripper '
        'changing state.\n')
    T = d['transition'].shape[1]
    tr2 = d['transition'].copy()
    near = np.zeros_like(tr2)
    for s in range(-2, 3):
        near |= np.roll(tr2, s, axis=1)
    near = near.ravel()
    say('| frames | n | clean mm | scene-shuffled mm | scene effect | 95% CI |')
    say('|---|---:|---:|---:|---:|---|')
    for name, m in (('near a transition', v & near), ('elsewhere', v & ~near)):
        p, lo, hi = cluster_bootstrap(clean[m], scene[m], win[m])
        say(f'| {name} | {int(m.sum()):,} | {clean[m].mean():.2f} | {scene[m].mean():.2f} | '
            f'**{p:+.1f}%** | [{lo:+.1f}, {hi:+.1f}] |')

    # ---- verdict -----------------------------------------------------------------------
    easy, hard = strata[0][1], strata[-1][1]
    hard_lo = strata[-1][2]
    say('\n## 4. Reading\n')
    if hard_lo > 10:
        verdict = ('The scene effect on the hardest frames is large and its interval clears '
                   '10%. The model DOES use the scene where the motion changes; the all-frame '
                   'average diluted it. The scene gate was measuring the wrong thing.')
    elif hard > 2 * max(easy, 0.5) and hard_lo > 0:
        verdict = ('The scene effect grows with difficulty and is clearly non-zero on the '
                   'hardest frames, but stays below 10%. The model uses the scene somewhat '
                   'where it matters -- partly a measurement problem, partly a real '
                   'under-use. Chunking is still worth trying.')
    else:
        verdict = ('The scene effect is small even on the hardest frames. The model '
                   'genuinely ignores the scene, and the one-step objective is the likely '
                   'cause. This is the case for action chunking.')
    say(verdict)
    say(f'\n(easiest bin {easy:+.1f}%, hardest bin {hard:+.1f}%)')
    (out_dir / 'STRATIFIED_SCENE_SHUFFLE.md').write_text(
        f'# Stratified scene-shuffle -- {title}\n\n' + '\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=DATA / 'validation_plan_v1.json')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--batch-size', type=int, default=16)   # must match 57 for donor alignment
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--reuse', action='store_true', help='re-analyse saved per_frame.npz')
    args = p.parse_args()
    args.out = args.out or OUT / args.checkpoint.parent.name / 'stratified_scene'
    args.out.mkdir(parents=True, exist_ok=True)
    d = (dict(np.load(args.out / 'per_frame.npz')) if args.reuse
         else collect(args))
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    title = (f"{args.checkpoint.parent.name} (step {ck.get('step')}, "
             f"{'proprioception' if ck['config'].get('proprioception') else 'visual only'})")
    analyse(d, args.out, expected_from_report(args.checkpoint), title)


if __name__ == '__main__':
    main()
