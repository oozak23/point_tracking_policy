"""Step 5: train-only group scales and masked, quality-weighted behavior cloning loss."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_decoder import rotation_6d_to_matrix


def compute_loss_scales(manifest: Path, gen: Path, shard_index: Path,
                        progress=None) -> dict:
    """Stream action labels only; never read validation labels or cached scene tokens.

    Average over valid TRAIN frames with manifest clip weights. Translation scale is
    RMS displacement norm in metres; rotation scale is RMS chordal distance / sqrt(2).
    Its square is approximately mean squared angle for small rotations. Neither scale
    centers the labels nor divides individual rotation coordinates by their variance.
    """
    manifest, gen, shard_index = Path(manifest), Path(gen), Path(shard_index)
    raw = manifest.read_bytes()
    clips = json.loads(raw)['clips']
    train = sorted((c for c in clips if c['split'] == 'train'), key=lambda c: c['video_id'])
    if not train or len({c['video_id'] for c in train}) != len(train):
        raise ValueError('manifest must contain nonempty, unique training clips')
    train_episodes = {c['episode'] for c in train}
    other_episodes = {c['episode'] for c in clips if c['split'] != 'train'}
    if train_episodes & other_episodes:
        raise ValueError('training episodes overlap another split')
    with shard_index.open() as f:
        shards = json.load(f)
    p_sum = r_sum = mass = 0.0
    n_valid = n_used = 0
    labels_hash = hashlib.sha256()
    for i, c in enumerate(train, 1):
        w = float(c['weight'])
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"invalid clip weight: {c['video_id']}")
        path = gen / shards[c['video_id']] / 'actions' / 'labels' / (c['video_id'] + '.npz')
        with np.load(path) as a:
            valid = a['valid']
            pos, d6 = a['delta_pos_body'], a['delta_rot6d_body']
        if valid.dtype != np.bool_ or valid.ndim != 1:
            raise ValueError(f'invalid validity mask: {path}')
        if pos.shape != (len(valid), 3) or d6.shape != (len(valid), 6):
            raise ValueError(f'invalid label shapes: {path}')
        n_valid += int(valid.sum())
        if w > 0:
            pos = pos[valid].astype(np.float64)
            d6 = d6[valid].astype(np.float64)
            if not np.isfinite(pos).all() or not np.isfinite(d6).all():
                raise ValueError(f'nonfinite valid labels: {path}')
            # Hash exactly the continuous labels used, together with their identity,
            # weight and validity pattern, for reproducibility of the scale artifact.
            labels_hash.update(json.dumps([c['video_id'], w]).encode())
            labels_hash.update(valid.tobytes())
            labels_hash.update(pos.tobytes())
            labels_hash.update(d6.tobytes())
            with torch.no_grad():
                R = rotation_6d_to_matrix(torch.from_numpy(d6))
                energy = 0.5 * (R - torch.eye(3, dtype=torch.float64)).square().sum().item()
            p_sum += w * float(np.square(pos).sum())
            r_sum += w * energy
            mass += w * len(pos)
            n_used += len(pos)
        if progress is not None and (i % 1000 == 0 or i == len(train)):
            progress(i, len(train))
    if mass <= 0 or p_sum <= 0 or r_sum <= 0:
        raise ValueError('positive valid weight and nonzero translation/rotation energy required')
    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': str(manifest.resolve()),
        'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'used_continuous_labels_sha256': labels_hash.hexdigest(),
        'gen': str(gen.resolve()), 'shard_index': str(shard_index.resolve()),
        'split': 'train', 'n_clips': len(train), 'n_episodes': len(train_episodes),
        'n_valid_frames': n_valid, 'n_used_frames': n_used, 'weight_sum': mass,
        'weighting': 'all valid training frames, weighted by manifest clip weight; not sampled windows',
        'position_scale_m': math.sqrt(p_sum / mass),
        'rotation_scale': math.sqrt(r_sum / mass),
        'position_scale_formula': 'sqrt(sum(w*valid*||delta_pos||^2)/sum(w*valid))',
        'rotation_scale_formula': 'sqrt(sum(w*valid*0.5*||delta_R-I||_F^2)/sum(w*valid))',
    }


class PolicyLoss(nn.Module):
    """L = weighted mean(||dp error||²/sp² + ||R error||²/(2*sr²) + BCE).

    Scales are registered buffers, so they travel with state_dict/device moves. The
    three components have unit coefficients. Reduction uses FP32 even under autocast.
    Excluded targets are selected out BEFORE geometry/BCE, so invalid-label NaNs are
    harmless. Nonfinite active predictions/labels and nonbinary gripper labels fail.
    """

    def __init__(self, position_scale_m: float, rotation_scale: float):
        super().__init__()
        for name, value in (('position_scale_m', position_scale_m), ('rotation_scale', rotation_scale)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    @classmethod
    def from_json(cls, path: Path, *, manifest: Path | None = None):
        with Path(path).open() as f:
            info = json.load(f)
        if info['schema_version'] != 1 or info['split'] != 'train':
            raise ValueError('expected a version-1 training-split scale artifact')
        if manifest is not None:
            digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
            if digest != info['manifest_sha256']:
                raise ValueError('scale artifact does not match the current manifest')
        return cls(info['position_scale_m'], info['rotation_scale'])

    def forward(self, predictions: dict, target: torch.Tensor,
                target_valid: torch.Tensor, weight: torch.Tensor) -> dict:
        """
        predictions: ActionHead output, leading shape (B,T,k)
        target: (B,T,10) for k=1, or explicit (B,T,k,10) future-aligned targets
        target_valid: bool, same leading axes as target
        weight: (B,), nonnegative manifest clip weights

        Returns differentiable loss/position/rotation/gripper scalars plus detached
        weight_sum and n_valid (active targets). If weight_sum=0, all losses are
        differentiable zeros. The training loop must SKIP the optimizer step then,
        since AdamW weight decay could otherwise change parameters without labels.
        """
        pos = predictions['delta_pos_body']
        rot = predictions['delta_rotmat_body']
        grip = predictions['gripper_logit']
        if pos.ndim != 4 or pos.shape[-1] != 3:
            raise ValueError('predicted position must have shape (B,T,k,3)')
        B, T, k, _ = pos.shape
        leading = (B, T, k)
        if rot.shape != (*leading, 3, 3) or grip.shape != (*leading, 1):
            raise ValueError('prediction shapes disagree')
        if target.ndim == 3:
            if k != 1 or target_valid.shape != (B, T):
                raise ValueError('single-step targets require k=1; do not broadcast future targets')
            target, target_valid = target.unsqueeze(2), target_valid.unsqueeze(2)
        if target.shape != (*leading, 10) or target_valid.shape != leading:
            raise ValueError('target and validity shapes must exactly match prediction axes')
        if target_valid.dtype != torch.bool:
            raise ValueError('target_valid must be boolean')
        if weight.shape != (B,) or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('weight must be finite, nonnegative, with shape (B,)')
        active = target_valid & (weight[:, None, None] > 0)
        p, R, g = pos[active].float(), rot[active].float(), grip[active].float()
        y = target[active].detach().float()
        w = weight.float()[:, None, None].expand(leading)[active]
        mass = w.sum()
        if not active.any():
            lp, lr, lg = p.sum(), R.sum(), g.sum()
        else:
            if not all(torch.isfinite(a).all() for a in (p, R, g, y)):
                raise ValueError('nonfinite active prediction or target')
            if not ((y[:, 9] == 0) | (y[:, 9] == 1)).all():
                raise ValueError('active gripper targets must be binary')
            target_R = rotation_6d_to_matrix(y[:, 3:9])
            ep = ((p - y[:, :3]) / self.position_scale_m).square().sum(-1)
            er = 0.5 * ((R - target_R) / self.rotation_scale).square().sum((-1, -2))
            eg = F.binary_cross_entropy_with_logits(g[:, 0], y[:, 9], reduction='none')
            lp, lr, lg = ((e * w).sum() / mass for e in (ep, er, eg))
        return {'loss': lp + lr + lg, 'position': lp, 'rotation': lr, 'gripper': lg,
                'weight_sum': mass.detach(), 'n_valid': active.sum().detach()}
