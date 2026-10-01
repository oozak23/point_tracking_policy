"""Action chunking: run C of docs/POINT_POLICY_PLAN.md.

Today's policy predicts one action per frame, 0.14 s ahead, where momentum explains most of the
answer. Here the same model, with K = 16 readout queries instead of one, predicts the next K
actions at once (2.2 s): query j predicts the action taken at frame t+j, j = 0..K-1, from
observations up to frame t only. Step 0 is exactly today's target, so first-step metrics stay
directly comparable with every earlier run.

Nothing else changes: same memory, same head, and the same loss scales for every step, since each
chunk element is a one-step action and "do nothing" costs about 1 at each of them.
"""
from __future__ import annotations

import numpy as np
import torch

from .action_decoder import rotation_6d_to_matrix
from .policy_data import PolicyWindows, TARGET_DIM
from .policy_loss import PolicyLoss
from .policy_train import Metrics, to_device

K = 16
CHUNK_RECIPE = 'k16_readout_queries_future_one_step_actions_shared_scales_v1'
REPORT_HORIZONS = (1, 4, 8, 16)


class ChunkPolicyWindows(PolicyWindows):
    """PolicyWindows plus, for each supervised frame t, the actions at frames t .. t+K-1.

    A step counts only if frame t+j is inside the stored clip and its own label is valid.
    """

    def __init__(self, *args, horizon: int = K, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon

    def sample(self, i: int, start: int | None = None) -> dict:
        s = super().sample(i, start=start)
        obj, _rob, tgt, valid = self._load(s['video_id'])
        n = min(len(obj), len(tgt))
        t = s['start'] + np.arange(self.T_win)
        tj = t[:, None] + np.arange(self.horizon)[None, :]                    # (Tw, K)
        inside = tj < n
        idx = np.clip(tj, 0, n - 1)
        chunk = np.zeros((self.T_win, self.horizon, TARGET_DIM), np.float32)
        chunk[inside] = tgt[idx][inside]
        s.update(chunk_target=chunk, chunk_valid=inside & valid[idx])
        return s


class ChunkPolicyLoss(PolicyLoss):
    """PolicyLoss for chunks. `chunk` scores all K steps; called like PolicyLoss with single-step
    targets it scores the first step only, which keeps every existing evaluation working and
    comparable with one-step runs."""

    def forward(self, predictions, target, target_valid, weight):
        if target.ndim == 3 and predictions['delta_pos_body'].shape[2] > 1:
            predictions = {k: v[:, :, :1] for k, v in predictions.items()}
        return super().forward(predictions, target, target_valid, weight)

    def chunk(self, pred, batch):
        return PolicyLoss.forward(self, pred, batch['chunk_target'], batch['chunk_valid'], batch['weight'])


def chunk_train_step(model, criterion, batch, optimizer, *, amp=False, clip_grad=1.):
    """One update on the mean loss over all valid (frame, step) targets."""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch['robot'].device.type, dtype=torch.bfloat16, enabled=amp):
        pred = model(batch)
        terms = criterion.chunk(pred, batch)
    if terms['weight_sum'].item() == 0:
        return terms, None
    if not torch.isfinite(terms['loss']):
        raise FloatingPointError('nonfinite training loss')
    terms['loss'].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
    optimizer.step()
    return terms, float(norm)


def compose(dp, dR):
    """Chain body-frame steps: (..., K, 3), (..., K, 3, 3) -> the pose after 1..K steps, in the
    frame at the start: positions (..., K, 3) and rotations (..., K, 3, 3)."""
    p = torch.zeros_like(dp[..., 0, :])
    R = torch.eye(3, dtype=dp.dtype, device=dp.device).expand(dR.shape[:-3] + (3, 3))
    ps, Rs = [], []
    for j in range(dp.shape[-2]):
        p = p + (R @ dp[..., j, :, None])[..., 0]
        R = R @ dR[..., j, :, :]
        ps.append(p)
        Rs.append(R)
    return torch.stack(ps, -2), torch.stack(Rs, -3)


@torch.no_grad()
def evaluate_chunk(model, criterion, loader, device, *, amp=False):
    """Chunk metrics on the loader's windows.

    chunk_loss and its three parts: the training objective, over all K steps.
    Per horizon h in REPORT_HORIZONS:
      step_mm       error of the single action at step h (the action taken at frame t+h-1)
      endpoint_mm   where the gripper is after h steps, chaining the h predicted actions, against
                    chaining the h true ones. Counted only if all h steps have labels.
      endpoint baselines on the same targets: "zero" (no motion) and, when the loader provides the
      previous action, "repeat_last" (the previous action repeated h times).
    """
    was_training = model.training
    model.eval()
    mass = 0.
    losses = dict.fromkeys(('loss', 'position', 'rotation', 'gripper'), 0.)
    Kq = None
    acc = {}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(batch)
            terms = criterion.chunk(pred, batch)
        m = float(terms['weight_sum'])
        mass += m
        for k in losses:
            losses[k] += float(terms[k]) * m
        y = batch['chunk_target'].float()
        valid = batch['chunk_valid'] & (batch['weight'][:, None, None] > 0)              # (B,T,K)
        Kq = y.shape[2]
        dp, dR = pred['delta_pos_body'].float(), pred['delta_rotmat_body'].float()
        tR = rotation_6d_to_matrix(y[..., 3:9])
        step_mm = torch.linalg.vector_norm(dp - y[..., :3], dim=-1) * 1000
        step_deg = Metrics.angle(dR.transpose(-1, -2) @ tR)
        through = valid.cumprod(-1).bool()                                               # steps 1..h all valid
        tp, tRc = compose(y[..., :3], tR)
        candidates = {'model': compose(dp, dR), 'zero': (torch.zeros_like(tp), torch.eye(3, device=device).expand_as(tRc))}
        if 'previous_action' in batch:
            prev = batch['previous_action'].float()
            pdp = prev[..., None, :3].expand(-1, -1, Kq, -1)
            pdR = rotation_6d_to_matrix(prev[..., 3:9])[..., None, :, :].expand(-1, -1, Kq, -1, -1)
            candidates['repeat_last'] = compose(pdp, pdR)
            through = through & batch['previous_valid'][..., None]      # every method on the same targets
        ok = dict.fromkeys(candidates, through)
        a = acc.setdefault('step', {'mm': np.zeros(Kq), 'deg': np.zeros(Kq), 'n': np.zeros(Kq)})
        z = torch.zeros_like(step_mm)
        a['mm'] += torch.where(valid, step_mm, z).sum((0, 1)).double().cpu().numpy()
        a['deg'] += torch.where(valid, step_deg, z).sum((0, 1)).double().cpu().numpy()
        a['n'] += valid.sum((0, 1)).double().cpu().numpy()
        for name, (cp, cR) in candidates.items():
            e_mm = torch.linalg.vector_norm(cp - tp, dim=-1) * 1000
            e_deg = Metrics.angle(cR.transpose(-1, -2) @ tRc)
            a = acc.setdefault(name, {'mm': np.zeros(Kq), 'deg': np.zeros(Kq), 'n': np.zeros(Kq)})
            a['mm'] += torch.where(ok[name], e_mm, z).sum((0, 1)).double().cpu().numpy()
            a['deg'] += torch.where(ok[name], e_deg, z).sum((0, 1)).double().cpu().numpy()
            a['n'] += ok[name].sum((0, 1)).double().cpu().numpy()
    model.train(was_training)
    if mass <= 0:
        raise ValueError('chunk evaluation has no active targets')
    out = {'chunk_loss': losses['loss'] / mass, 'chunk_position': losses['position'] / mass,
           'chunk_rotation': losses['rotation'] / mass, 'chunk_gripper': losses['gripper'] / mass, 'chunk': {}}
    for h in REPORT_HORIZONS:
        if h > Kq:
            continue
        i = h - 1
        row = {'step_mm': float(acc['step']['mm'][i] / max(acc['step']['n'][i], 1)),
               'step_deg': float(acc['step']['deg'][i] / max(acc['step']['n'][i], 1)),
               'n_step': int(acc['step']['n'][i])}
        for name in ('model', 'zero', 'repeat_last'):
            if name in acc:
                n = max(acc[name]['n'][i], 1)
                row[f'endpoint_mm_{name}'] = float(acc[name]['mm'][i] / n)
                row[f'endpoint_deg_{name}'] = float(acc[name]['deg'][i] / n)
                row[f'n_{name}'] = int(acc[name]['n'][i])
        out['chunk'][str(h)] = row
    return out
