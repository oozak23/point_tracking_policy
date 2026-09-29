"""Current-state late fusion for the frozen-visual action policy.

Only q[t], recorded qdot[t], and gripper_positions[t] are inputs to the prediction
of movement t->t+1. No finite differences, future states, targets or commands are read.
"""
import math

import numpy as np
import torch
from torch import nn

from .policy_data import PolicyWindows
from .policy_train import ActionPolicy

PROPRIO_RECIPE = 'current_q7_qdot7_gripper1_late_add_zero_init_v1'
STATE_FIELDS = ('joint_positions', 'joint_velocities', 'gripper_positions')
# Labels store gripper closure scaled to URDF radians: 0=open, 0.725=closed.
# Fixed physical scales avoid fitting statistics to held-out data.
STATE_SCALES = [math.pi]*7 + [1.0]*7 + [0.725]


def current_state_window(labels, start, length, current_valid):
    if start < 0 or np.shape(current_valid) != (length,):
        raise ValueError('invalid state window or mask')
    result = np.zeros((length, 15), np.float32)
    widths, offset = (7, 7, 1), 0
    live = np.flatnonzero(current_valid)
    indices = start + live
    for key, width in zip(STATE_FIELDS, widths):
        source = np.asarray(labels[key])
        if source.ndim != 2 or source.shape[1] != width:
            raise ValueError(f'{key} must have shape (frames, {width})')
        if len(indices) and indices[-1] >= len(source):
            raise ValueError(f'missing current-state coverage: {key}')
        values = source[indices]
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite current robot state: {key}')
        result[live, offset:offset+width] = values
        offset += width
    return result


class ProprioPolicyWindows(PolicyWindows):
    def sample(self, i, start=None):
        sample = super().sample(i, start=start)
        vid = sample['video_id']
        with np.load(self.gen/self.shard[vid]/'actions/labels'/f'{vid}.npz') as labels:
            sample['proprio'] = current_state_window(labels, sample['start'], self.T_win,
                                                     sample['hist_valid'][-self.T_win:])
        return sample


class ProprioActionPolicy(ActionPolicy):
    def __init__(self, D=256, **kwargs):
        super().__init__(D=D, **kwargs)
        # CPU-only module initialization inside fork_rng preserves the parent's
        # initialization and subsequent training dropout RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.proprio_encoder = nn.Sequential(nn.Linear(15, D), nn.GELU(), nn.Linear(D, D))
            nn.init.zeros_(self.proprio_encoder[-1].weight)
            nn.init.zeros_(self.proprio_encoder[-1].bias)
        self.register_buffer('proprio_scales', torch.tensor(STATE_SCALES, dtype=torch.float32))

    def forward(self, batch):
        pad = ~batch['robot_pad']
        T = batch['target'].shape[1]
        state = batch['proprio'].float()
        live = batch['hist_valid'][:, -T:]
        if state.shape != (batch['robot'].shape[0], T, 15):
            raise ValueError('proprio must have shape (batch, supervised frames, 15)')
        if not torch.isfinite(state[live]).all():
            raise ValueError('nonfinite active proprioception')
        state = torch.where(live[..., None], state, 0.)
        tokens = self.memory(batch['robot'].float(), batch['scene'].float(),
                             batch['hist_valid'], pad, T)
        readout = self.readout(tokens, pad)
        encoded = self.proprio_encoder(state/self.proprio_scales)
        encoded = torch.where(live[..., None], encoded, 0.)
        return self.head(readout + encoded[:, :, None, :])
