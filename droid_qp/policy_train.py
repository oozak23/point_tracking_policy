"""Training support for the frozen-token action policy (step 6)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .action_decoder import ActionHead, ActionMemory, ActionReadout, rotation_6d_to_matrix
from .policy_data import PolicyWindows


class ActionPolicy(nn.Module):
    """Only trainable modules; the frozen tracker is never instantiated by training."""

    def __init__(self, D=256, layers=3, M=24, scene_slots=24, checkpoint=True, dropout=0.1, k=1):
        super().__init__()
        self.memory = ActionMemory(D=D, layers=layers, M=M, scene_slots=scene_slots,
                                   checkpoint=checkpoint)
        # k readout queries -> k actions per frame. k=1: the next action. k>1: an action chunk,
        # query j predicting the action at frame t+j (droid_qp/chunk_policy.py).
        self.readout = ActionReadout(D=D, k=k)
        self.head = ActionHead(D=D)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
            elif isinstance(module, nn.MultiheadAttention):
                module.dropout = dropout

    def forward(self, batch):
        # PolicyWindows uses True=real; both attention modules use True=padding.
        pad = ~batch['robot_pad']
        tokens = self.memory(batch['robot'].float(), batch['scene'].float(),
                             batch['hist_valid'], pad, batch['target'].shape[1])
        return self.head(self.readout(tokens, pad))


def select_overfit_windows(data: PolicyWindows, count=10, seed=0):
    """Select distinct training episodes with reliable tracking and a gripper change.

    Each fixed window is centered on a transition and must include valid motion.
    This intentional selection is a wiring/memorization test, not a population sample.
    """
    selected, episodes = [], set()
    for i in np.random.default_rng(seed).permutation(len(data.clips)):
        c = data.clips[int(i)]
        if c['weight'] < 0.75 or c['episode'] in episodes:
            continue
        vid = c['video_id']
        path = data.gen / data.shard[vid] / 'actions/labels' / f'{vid}.npz'
        with np.load(path) as a:
            valid = a['valid']
            pos = a['delta_pos_body']
            changes = np.flatnonzero(valid & (a['gripper_open'].ravel() != a['gripper_open_next'].ravel()))
        if len(valid) < data.T_win or not len(changes):
            continue
        start = int(np.clip(changes[0] - data.T_win // 2, 0, len(valid) - data.T_win))
        span = slice(start, start + data.T_win)
        ok = valid[span]
        if ok.sum() < data.T_win - 1 or np.linalg.norm(pos[span][ok], axis=1).max() < 0.01:
            continue
        selected.append({'index': int(i), 'video_id': vid, 'episode': c['episode'],
                         'start': start, 'weight': float(c['weight'])})
        episodes.add(c['episode'])
        if len(selected) == count:
            return selected
    raise ValueError(f'only found {len(selected)} eligible overfit episodes, need {count}')


class TrainingWindows(Dataset):
    """Epoch/index-seeded windows, reproducible across DataLoader worker counts.

    fixed is used for overfit/validation. Otherwise one random window per clip/epoch.
    Workers are recreated each epoch so they receive the current epoch value.
    """

    def __init__(self, data: PolicyWindows, *, fixed=None, seed=0):
        self.data, self.fixed, self.seed, self.epoch = data, fixed, seed, 0
        self.samples = [self._sample(r['index'], r['start']) for r in fixed] if fixed is not None else None

    def __len__(self):
        return len(self.samples) if self.samples is not None else len(self.data)

    def _sample(self, index, start):
        sample = self.data.sample(index, start=start)
        vid = sample['video_id']
        path = self.data.gen / self.data.shard[vid] / 'actions/labels' / f'{vid}.npz'
        with np.load(path) as a:
            current = a['gripper_open'].ravel()
        s, T = sample['start'], self.data.T_win
        grip = np.zeros(T, np.float32)
        grip[:len(current[s:s+T])] = current[s:s+T]
        sample['gripper_current'] = grip
        return sample

    def __getitem__(self, index):
        if self.samples is not None:
            return self.samples[index]
        clip = self.data.clips[index]
        obj, _, target, _ = self.data._load(clip['video_id'])
        hi = max(0, min(len(obj), len(target)) - self.data.T_win)
        start = int(np.random.default_rng([self.seed, self.epoch, index]).integers(hi + 1))
        return self._sample(index, start)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def lr_factor(step, total_steps, warmup):
    if warmup and step < warmup:
        return (step + 1) / warmup
    fraction = min(1., max(0., (step - warmup) / max(1, total_steps - warmup)))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * fraction))


def make_optimizer(model, criterion, lr=3e-4, weight_decay=0.):
    """AdamW steps respect physical output scales; moment normalization alone does not.

    A shared 3e-4 step on a meter-valued projection caused a ~10 cm jump in the first
    CPU smoke test. Scale those two projection learning rates while retaining physical
    outputs and the agreed loss. Backbone/readout/gripper use the base learning rate.
    """
    position = list(model.head.position.parameters())
    rotation = list(model.head.rotation.parameters())
    special = {id(p) for p in position + rotation}
    rest = [p for p in model.parameters() if id(p) not in special]
    return torch.optim.AdamW([
        {'params': rest, 'lr': lr, 'name': 'memory_readout_gripper'},
        {'params': position, 'lr': lr * float(criterion.position_scale_m), 'name': 'position'},
        {'params': rotation, 'lr': lr * float(criterion.rotation_scale), 'name': 'rotation'},
    ], weight_decay=weight_decay)


def train_step(model, criterion, batch, optimizer, *, amp=False, clip_grad=1.):
    """Return loss components and pre-clipping norm; empty batches never step AdamW."""
    optimizer.zero_grad(set_to_none=True)
    device_type = batch['robot'].device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        pred = model(batch)
        terms = criterion(pred, batch['target'], batch['target_valid'], batch['weight'])
    if terms['weight_sum'].item() == 0:
        return terms, None
    if not torch.isfinite(terms['loss']):
        raise FloatingPointError('nonfinite training loss')
    terms['loss'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
    optimizer.step()
    return terms, float(norm)


class Metrics:
    """Weighted loss components, unweighted physical errors and persistence baseline."""

    def __init__(self):
        self.mass = self.count = self.transitions = 0
        self.losses = dict.fromkeys(('loss', 'position', 'rotation', 'gripper'), 0.)
        self.sums = dict.fromkeys(('position_cm', 'rotation_deg', 'gripper_correct',
                                   'transition_correct', 'persistence_correct',
                                   'zero_position_cm', 'identity_rotation_deg'), 0.)

    @staticmethod
    def angle(R):
        skew = torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                            R[..., 1, 0] - R[..., 0, 1]), -1)
        sin = torch.linalg.vector_norm(skew, dim=-1) / 2
        cos = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        return torch.rad2deg(torch.atan2(sin, cos))

    def add(self, pred, batch, terms):
        mass = float(terms['weight_sum'])
        self.mass += mass
        for k in self.losses:
            self.losses[k] += float(terms[k]) * mass
        valid = batch['target_valid'] & (batch['weight'][:, None] > 0)
        y = batch['target'][valid].float()
        if not len(y):
            return
        p = pred['delta_pos_body'][:, :, 0][valid].float()
        R = pred['delta_rotmat_body'][:, :, 0][valid].float()
        target_R = rotation_6d_to_matrix(y[:, 3:9])
        g = pred['gripper_logit'][:, :, 0, 0][valid] >= 0
        target_g = y[:, 9].bool()
        current_g = batch['gripper_current'][valid].bool()
        change = current_g != target_g
        self.count += len(y)
        self.transitions += int(change.sum())
        self.sums['position_cm'] += float(torch.linalg.vector_norm(p - y[:, :3], dim=-1).sum()) * 100
        self.sums['zero_position_cm'] += float(torch.linalg.vector_norm(y[:, :3], dim=-1).sum()) * 100
        self.sums['rotation_deg'] += float(self.angle(R.transpose(-1, -2) @ target_R).sum())
        self.sums['identity_rotation_deg'] += float(self.angle(target_R).sum())
        self.sums['gripper_correct'] += int((g == target_g).sum())
        self.sums['transition_correct'] += int((g[change] == target_g[change]).sum())
        self.sums['persistence_correct'] += int((current_g == target_g).sum())

    def result(self):
        if self.mass <= 0 or self.count == 0:
            raise ValueError('evaluation has no active targets')
        return {**{k: v / self.mass for k, v in self.losses.items()},
                'n_valid': self.count, 'n_transitions': self.transitions,
                **{k: self.sums[k] / self.count for k in ('position_cm', 'rotation_deg',
                     'zero_position_cm', 'identity_rotation_deg')},
                'gripper_accuracy': self.sums['gripper_correct'] / self.count,
                'persistence_accuracy': self.sums['persistence_correct'] / self.count,
                'transition_frame_accuracy': self.sums['transition_correct'] / self.transitions if self.transitions else None}


@torch.no_grad()
def evaluate(model, criterion, loader, device, *, amp=False):
    was_training = model.training
    model.eval()
    metrics = Metrics()
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(batch)
            terms = criterion(pred, batch['target'], batch['target_valid'], batch['weight'])
        metrics.add(pred, batch, terms)
    model.train(was_training)
    return metrics.result()


def save_checkpoint(path: Path, state):
    """Replace only after a complete write, keeping the last checkpoint recoverable."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
