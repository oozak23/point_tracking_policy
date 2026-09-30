"""Point-prediction policy: docs/POINT_POLICY_PLAN.md.

Two additions to the action policy, each switchable so the plan's control runs share one code path:

  coords   each robot point's Track-On TRACKED image position and visibility flag, encoded and added
           to its token at every stream frame. Tokens alone hold a point's position only to ~25 px
           (out/probe_position.json).
  future   part 1: a per-point head predicting the point's displacement over the next H = 16 frames,
           from its token and its current coordinates. Part 2: that prediction, embedded, is added to
           the point's token before the (unchanged) readout and action head.

Targets for part 1 are FK displacements fk[t+j] - fk[t]: exact, unaffected by tracking error. A target
counts only if the point is inside the image at t and at t+j, and t+j is a stored frame of the clip.

Every new output layer is zero-initialised inside fork_rng, so at initialisation the model computes
exactly what ActionPolicy computes with the same seed, and part 1 predicts no motion.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .policy_data import N_ROB_MAX, PolicyWindows
from .policy_loss import PolicyLoss
from .policy_train import ActionPolicy, Metrics, to_device

H = 16                       # frames of future predicted per point
IMG_W, IMG_H = 854, 480      # FK and track pixel space
N_FREQ = 8
COORD_FEATS = 2 + 1 + 2 * 2 * N_FREQ
COORD_RECIPE = 'tracked_uv_vis_fourier8_input_add_zero_init_v1'
FUTURE_RECIPE = 'fk_disp16_scaled_per_horizon_head_token_plus_coords_plan_add_zero_init_v1'
REPORT_HORIZONS = (1, 4, 8, 16)


def normalise_uv(uv):
    """Pixels -> [-1, 1] in both axes."""
    return np.stack([uv[..., 0] / IMG_W * 2 - 1, uv[..., 1] / IMG_H * 2 - 1], -1)


class PointPolicyWindows(PolicyWindows):
    """PolicyWindows plus, per robot point: tracked coordinates for every stream frame, and FK
    displacement targets over the next H frames for every supervised frame.

    Robot token columns, track columns and FK ids are the same points in the same order (the merge
    writes robot rows in point_index order); this is asserted per clip.
    """

    def __init__(self, *args, horizon: int = H, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self._points: dict[str, tuple] = {}
        self._points_order: list[str] = []

    def _load_points(self, vid):
        if vid in self._points:
            return self._points[vid]
        base = self.gen / self.shard[vid]
        with np.load(base / 'trackon' / 'tracks' / f'{vid}.npz') as t:
            cols = np.flatnonzero(t['role'] == 1)
            trk = t['tracks'][:, cols].astype(np.float32)
            vis = t['visibility'][:, cols].astype(bool)
            birth = t['queries'][cols, 0].astype(np.int64)
        (merged,) = sorted((base / 'merged' / 'query_points').glob(f'{vid}_f*.npz'))
        with np.load(merged) as m:
            ids = m['point_index']
            fk = m['fk_uv'][:, ids].astype(np.float32)
            ok = m['fk_ok'][:, ids].astype(bool)
        if len(ids) != len(cols):
            raise ValueError(f'{vid}: {len(cols)} tracked robot columns but {len(ids)} FK ids')
        item = (trk, vis, birth, fk, ok)
        self._points[vid] = item
        self._points_order.append(vid)
        while len(self._points_order) > self._cache_max:
            self._points.pop(self._points_order.pop(0), None)
        return item

    def sample(self, i: int, start: int | None = None) -> dict:
        s = super().sample(i, start=start)
        vid, st, n_rob = s['video_id'], s['start'], s['n_rob']
        obj, _rob, tgt, _valid = self._load(vid)
        trk, vis, birth, fk, ok = self._load_points(vid)
        if trk.shape[1] != n_rob:
            raise ValueError(f'{vid}: token and track robot counts differ')
        n_clip = min(len(obj), len(tgt))                  # frames the base sample may use
        n_store = min(n_clip, len(fk), len(trk))          # frames with FK and tracks too
        M, Tw, Hz = self.M, self.T_win, self.horizon
        L = (M - 1) + Tw

        # ---- inputs: tracked coordinates at every stream frame (same rows as the tokens) ----
        frame = st - (M - 1) + np.arange(L)
        row_live = s['hist_valid'] & (frame < n_store)
        f = np.clip(frame, 0, n_store - 1)
        coords = np.zeros((L, N_ROB_MAX, 2), np.float32)
        coord_vis = np.zeros((L, N_ROB_MAX), np.float32)
        coord_live = np.zeros((L, N_ROB_MAX), bool)
        live = row_live[:, None] & (frame[:, None] >= birth[None, :])     # tracked from birth on
        coords[:, :n_rob] = np.where(live[..., None], normalise_uv(trk[f]), 0)
        coord_vis[:, :n_rob] = live & vis[f]
        coord_live[:, :n_rob] = live

        # ---- targets: FK displacement from t to t+j for each supervised frame ------------------
        t = st + np.arange(Tw)
        tj = t[:, None] + np.arange(1, Hz + 1)[None, :]                    # (Tw, H)
        in_range = (t[:, None] < n_store) & (tj < n_store)
        tc, tjc = np.clip(t, 0, n_store - 1), np.clip(tj, 0, n_store - 1)
        disp = fk[tjc] - fk[tc][:, None]                                   # (Tw, H, n_rob, 2)
        valid = in_range[..., None] & ok[tc][:, None] & ok[tjc]            # (Tw, H, n_rob)
        fut_disp = np.zeros((Tw, N_ROB_MAX, Hz, 2), np.float32)
        fut_valid = np.zeros((Tw, N_ROB_MAX, Hz), bool)
        fut_disp[:, :n_rob] = np.where(valid[..., None], disp, 0).transpose(0, 2, 1, 3)
        fut_valid[:, :n_rob] = valid.transpose(0, 2, 1)
        s.update(coords=coords, coord_vis=coord_vis, coord_live=coord_live,
                 fut_disp=fut_disp, fut_valid=fut_valid)
        return s


def coord_features(uv, vis, live):
    """(..., 2) normalised position, (...) visibility, (...) live -> (..., COORD_FEATS).

    Fourier features let a small MLP resolve fine position differences. Entries without a
    track (padding, before the point's birth frame) are all zeros.
    """
    freqs = (2.0 ** torch.arange(N_FREQ, device=uv.device, dtype=uv.dtype)) * math.pi
    ang = uv[..., None] * freqs                                         # (..., 2, F)
    feats = torch.cat([uv, vis[..., None], ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
    return torch.where(live[..., None], feats, torch.zeros_like(feats))


class PointPolicy(ActionPolicy):
    """ActionPolicy + current coordinates (coords) + future-point head and plan (future)."""

    def __init__(self, D=256, *, coords=True, future=True, horizon=H, **kwargs):
        super().__init__(D=D, **kwargs)
        self.use_coords, self.use_future, self.horizon = coords, future, horizon
        self.plan_grad = True        # False: the action loss never reaches part 1 through the plan
        # New modules are created inside fork_rng so the shared weights, and the dropout sequence
        # afterwards, are exactly those of ActionPolicy with the same seed.
        with torch.random.fork_rng(devices=[]):
            if coords:
                self.coord_embed = nn.Sequential(nn.Linear(COORD_FEATS, D), nn.GELU(), nn.Linear(D, D))
                nn.init.zeros_(self.coord_embed[-1].weight)
                nn.init.zeros_(self.coord_embed[-1].bias)
            if future:
                self.future_head = nn.Sequential(nn.LayerNorm(D + COORD_FEATS), nn.Linear(D + COORD_FEATS, D),
                                                 nn.GELU(), nn.Linear(D, horizon * 2))
                self.plan_embed = nn.Sequential(nn.Linear(horizon * 2, D), nn.GELU(), nn.Linear(D, D))
                for last in (self.future_head[-1], self.plan_embed[-1]):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

    def forward(self, batch):
        pad = ~batch['robot_pad']
        T = batch['target'].shape[1]
        robot = batch['robot'].float()
        feats = None
        if self.use_coords or self.use_future:
            feats = coord_features(batch['coords'].float(), batch['coord_vis'].float(), batch['coord_live'])
        if self.use_coords:
            robot = robot + self.coord_embed(feats)
        tokens = self.memory(robot, batch['scene'].float(), batch['hist_valid'], pad, T)
        future = None
        if self.use_future:
            now = feats[:, -T:].to(tokens.dtype)
            future = self.future_head(torch.cat([tokens, now], -1))          # (B, T, N, 2H), scaled units
            plan = future if self.plan_grad else future.detach()
            if 'oracle_plan' in batch:
                # Evaluation only (evaluation/57_...): the RECORDED future replaces the prediction
                # wherever a target exists, giving the "perfect plan" upper bound. Never in training.
                plan = torch.where(batch['oracle_mask'], batch['oracle_plan'].to(plan.dtype), plan)
            tokens = tokens + self.plan_embed(plan)
            future = future.unflatten(-1, (self.horizon, 2))
        pred = self.head(self.readout(tokens, pad))
        if future is not None:
            pred['future_disp'] = future
        return pred


class PointPolicyLoss(PolicyLoss):
    """The unchanged action loss, plus `points`: the future-point loss.

    points = sum(w * ||pred - disp / scale_j||^2) / sum(w) over valid (frame, point, horizon)
    targets, w the clip weight. scale_j is the training-set RMS displacement at horizon j
    (pipeline/85_point_loss_scales.py), so predicting no motion costs ~1 at every horizon.
    Called like PolicyLoss for actions, so every existing evaluation works unchanged.
    """

    def __init__(self, position_scale_m, rotation_scale, point_scales, point_weight=1.0):
        super().__init__(position_scale_m, rotation_scale)
        scales = torch.tensor(point_scales, dtype=torch.float32)
        if scales.ndim != 1 or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('point scales must be finite and positive')
        self.register_buffer('point_scales', scales)
        self.point_weight = float(point_weight)

    @classmethod
    def from_files(cls, action_scales: Path, point_scales: Path, *, manifest: Path, point_weight=1.0):
        base = PolicyLoss.from_json(action_scales, manifest=manifest)
        info = json.loads(Path(point_scales).read_text())
        if info['split'] != 'train' or info['manifest_sha256'] != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
            raise ValueError('point scales are not from the training split of this manifest')
        return cls(float(base.position_scale_m), float(base.rotation_scale), info['scale_px'], point_weight)

    def points(self, pred, batch):
        f = pred['future_disp'].float()                                     # (B, T, N, H, 2)
        Hz = f.shape[-2]
        y = batch['fut_disp'].float() / self.point_scales[:Hz, None]
        valid = (batch['fut_valid'] & batch['robot_pad'][:, None, :, None]
                 & (batch['weight'][:, None, None, None] > 0))
        w = batch['weight'].float()[:, None, None, None].expand(valid.shape)[valid]
        mass = w.sum()
        if not valid.any():
            return {'points': f.sum() * 0, 'point_weight_sum': mass.detach(), 'n_point_targets': 0}
        e = ((f[valid] - y[valid]) ** 2).sum(-1)
        return {'points': (e * w).sum() / mass, 'point_weight_sum': mass.detach(),
                'n_point_targets': int(valid.sum())}


def point_train_step(model, criterion, batch, optimizer, *, stage='joint', amp=False, clip_grad=1.):
    """stage 'points': part 1 pre-training, the point loss only. stage 'joint': action loss +
    point_weight * point loss. Returns (terms, grad norm); empty batches never step AdamW."""
    if stage not in ('points', 'joint'):
        raise ValueError(stage)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch['robot'].device.type, dtype=torch.bfloat16, enabled=amp):
        pred = model(batch)
        terms = criterion(pred, batch['target'], batch['target_valid'], batch['weight'])
        pterms = criterion.points(pred, batch)
    use_action = stage == 'joint' and terms['weight_sum'].item() > 0
    use_points = pterms['point_weight_sum'].item() > 0
    if not (use_action or use_points):
        return {**terms, **pterms}, None
    total = criterion.point_weight * pterms['points'] if use_points else 0.
    if use_action:
        total = total + terms['loss']
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite training loss')
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, error_if_nonfinite=True)
    optimizer.step()
    return {**terms, **pterms, 'total': total.detach()}, float(norm)


@torch.no_grad()
def evaluate_points(model, criterion, loader, device, *, amp=False):
    """Action metrics (as policy_train.evaluate) plus future-point metrics on the same windows.

    Point metrics per horizon: mean pixel error of the model, of "no motion", and of "repeat the
    last tracked motion" (the same information the model gets; no motion if either track is
    hidden), all on exactly the same targets.
    """
    was_training = model.training
    model.eval()
    metrics = Metrics()
    mass = loss_sum = 0.
    Hz = model.horizon
    sums = {k: np.zeros(Hz) for k in ('model', 'zero', 'last_tracked', 'n')}
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(batch)
            terms = criterion(pred, batch['target'], batch['target_valid'], batch['weight'])
            pterms = criterion.points(pred, batch)
        metrics.add(pred, batch, terms)
        m = float(pterms['point_weight_sum'])
        mass += m
        loss_sum += float(pterms['points']) * m
        T = batch['target'].shape[1]
        valid = batch['fut_valid'] & batch['robot_pad'][:, None, :, None]              # (B,T,N,H)
        scale = criterion.point_scales[:Hz]
        pred_px = pred['future_disp'].float() * scale[:, None]
        y = batch['fut_disp'].float()
        # last tracked motion: pixels between the previous and the current stream frame
        uv = batch['coords'].float()
        px = torch.stack([(uv[..., 0] + 1) / 2 * IMG_W, (uv[..., 1] + 1) / 2 * IMG_H], -1)
        now, prev = px[:, -T:], px[:, -T - 1:-1]
        seen = (batch['coord_vis'][:, -T:] > 0) & (batch['coord_vis'][:, -T - 1:-1] > 0)
        vel = torch.where(seen[..., None], now - prev, torch.zeros_like(now))            # (B,T,N,2)
        steps = torch.arange(1, Hz + 1, device=device, dtype=vel.dtype)
        last = vel[:, :, :, None, :] * steps[:, None]                                    # (B,T,N,H,2)
        for key, p in (('model', pred_px), ('zero', torch.zeros_like(y)), ('last_tracked', last)):
            err = torch.linalg.vector_norm(p - y, dim=-1)
            sums[key] += torch.where(valid, err, torch.zeros_like(err)).sum((0, 1, 2)).double().cpu().numpy()
        sums['n'] += valid.sum((0, 1, 2)).double().cpu().numpy()
    model.train(was_training)
    out = metrics.result()
    points = loss_sum / mass if mass else None
    out['points'] = points
    out['total'] = out['loss'] + criterion.point_weight * points if points is not None else None
    n = np.maximum(sums['n'], 1)
    out['point_px'] = {str(j): {k: float(sums[k][j - 1] / n[j - 1]) for k in ('model', 'zero', 'last_tracked')}
                       | {'n': int(sums['n'][j - 1])} for j in range(1, Hz + 1) if j in REPORT_HORIZONS}
    return out
