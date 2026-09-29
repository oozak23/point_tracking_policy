"""Offline validation: fixed windows, copy-action baselines, events and lab breakdowns."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .action_decoder import rotation_6d_to_matrix
from .policy_train import Metrics, to_device


class ValidationWindows(Dataset):
    """Lazy fixed windows; do not preload the entire validation token set into RAM."""

    def __init__(self, data, records):
        self.data, self.records = data, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        sample = self.data.sample(row['index'], start=row['start'])
        if sample['start'] != row['start']:
            raise ValueError('frozen window start does not match cached token/label lengths')
        vid, s, T = sample['video_id'], sample['start'], self.data.T_win
        path = self.data.gen / self.data.shard[vid] / 'actions/labels' / f'{vid}.npz'
        with np.load(path) as a:
            n = len(a['valid'])
            current = np.zeros(T, np.float32)
            current[:min(T,n-s)] = a['gripper_open'][s:s+T, 0]
            indices = np.arange(s, s+T)
            prev_i = np.clip(indices-1, 0, n-1)
            previous = np.concatenate([a['delta_pos_body'][prev_i], a['delta_rot6d_body'][prev_i]], -1)
            prev_valid = (indices > 0) & (indices < n) & a['valid'][prev_i]
        sample.update(gripper_current=current, previous_action=previous.astype(np.float32),
                      previous_valid=prev_valid, lab=vid.split('+')[0], episode=row['episode'])
        return sample


def replace_robot_history(robot, donor, mappings):
    """Copy FK-ID-aligned donor history, preserving the recipient's last frame exactly."""
    result = robot.clone()
    for i, mapping in enumerate(mappings):
        result[i, :-1, :len(mapping)] = donor[i, :-1, mapping]
    return result


def match_events(true, predicted, tolerance=2):
    """Chronological DP: maximize matches, then minimize absolute timing difference."""
    n, m = len(true), len(predicted)
    dp = [[(0, 0, []) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [dp[i-1][j], dp[i][j-1]]
            error = predicted[j-1] - true[i-1]
            if abs(error) <= tolerance:
                c, cost, errors = dp[i-1][j-1]
                options.append((c+1, cost+abs(error), errors+[error]))
            dp[i][j] = max(options, key=lambda x: (x[0], -x[1]))
    matched, cost, errors = dp[n][m]
    return dict(true=n, predicted=m, matched=matched, abs_error=cost,
                signed_error=sum(errors), missed=n-matched, extra=m-matched)


def events(states, current_initial, valid):
    previous = np.r_[current_initial, states[:-1]]
    ok = valid & np.r_[True, valid[:-1]]
    return {name: np.flatnonzero(ok & (states != previous) & (states == state)).tolist()
            for name, state in [('open', True), ('close', False)]}


def event_report(counts):
    true, pred, matched = counts['true'], counts['predicted'], counts['matched']
    return {**counts, 'precision': matched/pred if pred else None,
            'recall': matched/true if true else None,
            'f1': 2*matched/(true+pred) if true+pred else None,
            'matched_mae_frames': counts['abs_error']/matched if matched else None,
            'timing_cost_frames': (counts['abs_error']+3*(counts['missed']+counts['extra']))/true if true else None}


class ValidationMetrics:
    def __init__(self):
        self.n = self.common_n = self.mass = self.windows = 0
        self.sums = dict.fromkeys(('position_mm', 'zero_mm', 'rotation_deg', 'identity_deg',
                                  'common_position_mm', 'common_zero_mm', 'copy_position_mm',
                                  'common_rotation_deg', 'copy_rotation_deg', 'correct', 'persistence_correct'), 0.)
        self.losses = dict.fromkeys(('loss', 'position', 'rotation', 'gripper'), 0.)
        self.ev = {who: {direction: dict.fromkeys(('true', 'predicted', 'matched', 'abs_error',
                                                  'signed_error', 'missed', 'extra'), 0)
                         for direction in ('open', 'close')}
                   for who in ('policy', 'persistence')}

    def add(self, pred, batch, terms, *, anchor=False):
        valid = batch['target_valid'] & (batch['weight'][:, None] > 0)
        if anchor:
            valid = valid.clone()
            valid[:, :-1] = False
        common = valid & batch['previous_valid']
        y = batch['target'].float()
        R = pred['delta_rotmat_body'][:, :, 0].float()
        target_R = rotation_6d_to_matrix(y[..., 3:9])
        previous_R = rotation_6d_to_matrix(batch['previous_action'][..., 3:9].float())
        position = torch.linalg.vector_norm(pred['delta_pos_body'][:, :, 0].float() - y[..., :3], dim=-1)*1000
        zero = torch.linalg.vector_norm(y[..., :3], dim=-1)*1000
        copy = torch.linalg.vector_norm(batch['previous_action'][..., :3] - y[..., :3], dim=-1)*1000
        rotation = Metrics.angle(R.transpose(-1, -2) @ target_R)
        copy_rotation = Metrics.angle(previous_R.transpose(-1, -2) @ target_R)
        values = {'position_mm': (position, valid), 'zero_mm': (zero, valid),
                  'rotation_deg': (rotation, valid), 'identity_deg': (Metrics.angle(target_R), valid),
                  'common_position_mm': (position, common), 'common_zero_mm': (zero, common),
                  'copy_position_mm': (copy, common), 'common_rotation_deg': (rotation, common),
                  'copy_rotation_deg': (copy_rotation, common)}
        for key, (a, mask) in values.items():
            self.sums[key] += float(a[mask].sum())
        states = pred['gripper_logit'][:, :, 0, 0] >= 0
        truth, current = y[..., 9].bool(), batch['gripper_current'].bool()
        self.sums['correct'] += int((states[valid] == truth[valid]).sum())
        self.sums['persistence_correct'] += int((current[valid] == truth[valid]).sum())
        self.n += int(valid.sum())
        self.common_n += int(common.sum())
        self.windows += len(y)
        if terms is not None:
            mass = float(terms['weight_sum'])
            self.mass += mass
            for key in self.losses:
                self.losses[key] += mass*float(terms[key])
        if not anchor:
            for i in range(len(y)):
                gt = events(truth[i].cpu().numpy(), bool(current[i, 0]), valid[i].cpu().numpy())
                for who, seq in [('policy', states), ('persistence', current)]:
                    predicted = events(seq[i].cpu().numpy(), bool(current[i, 0]), valid[i].cpu().numpy())
                    for direction in ('open', 'close'):
                        row = match_events(gt[direction], predicted[direction])
                        for key, value in row.items():
                            self.ev[who][direction][key] += value

    def result(self):
        out = {'n_targets': self.n, 'n_copy_comparable': self.common_n, 'n_windows': self.windows,
               **{k: v/self.mass if self.mass else None for k, v in self.losses.items()}}
        for key, value in self.sums.items():
            denom = self.common_n if key.startswith(('common_', 'copy_')) else self.n
            out[key] = value/denom if denom else None
        out['gripper_accuracy'] = out.pop('correct')
        out['persistence_accuracy'] = out.pop('persistence_correct')
        out['events'] = {}
        for who in self.ev:
            total = {k: sum(v[k] for v in self.ev[who].values()) for k in self.ev[who]['open']}
            out['events'][who] = {'all': event_report(total),
                                  **{d: event_report(v) for d, v in self.ev[who].items()}}
        return out


@torch.no_grad()
def evaluate_validation(model, criterion, loader, device, *, amp=False):
    was_training = model.training
    model.eval()
    total, labs = ValidationMetrics(), {}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(batch)
            terms = criterion(pred, batch['target'], batch['target_valid'], batch['weight'])
        total.add(pred, batch, terms)
        for lab in set(batch['lab']):
            indices = torch.tensor([i for i, v in enumerate(batch['lab']) if v == lab], device=device)
            sub = {k: v[indices] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            subpred = {k: v[indices] for k, v in pred.items()}
            subterms = criterion(subpred, sub['target'], sub['target_valid'], sub['weight'])
            labs.setdefault(lab, ValidationMetrics()).add(subpred, sub, subterms)
    model.train(was_training)
    return {**total.result(), 'per_lab': {k: v.result() for k, v in sorted(labs.items())},
            'evaluation_type': 'one-step predictions conditioned on recorded demonstrator observations; not closed-loop'}
