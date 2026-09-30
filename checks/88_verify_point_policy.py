#!/usr/bin/env python3
"""Checks for the point-prediction policy (droid_qp/point_policy.py, docs/POINT_POLICY_PLAN.md).

Data: targets are exactly the FK displacements, masked where the plan says; inputs are the tracked
positions, only from frames the tokens also come from. Model: at initialisation it computes exactly
what ActionPolicy computes; it never reads the targets; it is causal; every new layer learns; the
pre-training stage leaves the action head untouched. Loss: matches hand-computed values.

    $PY_TRACKON -B checks/88_verify_point_policy.py
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402
from droid_qp.point_policy import (H, IMG_H, IMG_W, PointPolicy, PointPolicyLoss,  # noqa: E402
                                   PointPolicyWindows, point_train_step)
from droid_qp.policy_train import ActionPolicy, make_optimizer  # noqa: E402

T_WIN, M = 4, 4
SMALL = dict(layers=1, M=M, scene_slots=M, checkpoint=False, dropout=0.)


def batch_of(data, index, starts):
    return default_collate([dict(data.sample(index, start=s), gripper_current=np.zeros(T_WIN, np.float32))
                            for s in starts])


def tensors(b):
    return {k: v for k, v in b.items() if isinstance(v, torch.Tensor)}


class PointPolicyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = PointPolicyWindows('train', T_win=T_WIN, M=M, cache_clips=4)
        cls.index = 0
        vid = cls.data.clips[cls.index]['video_id']
        base = cls.data.gen / cls.data.shard[vid]
        (mg,) = sorted((base / 'merged/query_points').glob(f'{vid}_f*.npz'))
        with np.load(mg) as m:
            ids = m['point_index']
            cls.fk, cls.ok = m['fk_uv'][:, ids].astype(np.float64), m['fk_ok'][:, ids]
        with np.load(base / 'trackon/tracks' / f'{vid}.npz') as t:
            cols = np.flatnonzero(t['role'] == 1)
            cls.trk, cls.vis = t['tracks'][:, cols], t['visibility'][:, cols]
            cls.birth = t['queries'][cols, 0].astype(int)
        obj, _, tgt, _ = cls.data._load(vid)
        cls.n = min(len(obj), len(tgt), len(cls.fk), len(cls.trk))
        cls.scales = json.loads((DATA / 'point_loss_scales.json').read_text())['scale_px']

    # ---- data ----------------------------------------------------------------------------
    def test_targets_are_fk_displacements_where_valid(self):
        s = self.data.sample(self.index, start=40)
        checked = 0
        for k in range(T_WIN):
            t = 40 + k
            for j in range(1, H + 1):
                for p in range(s['n_rob']):
                    expect = (t + j < self.n) and self.ok[t, p] and self.ok[t + j, p]
                    self.assertEqual(bool(s['fut_valid'][k, p, j - 1]), bool(expect))
                    if expect:
                        np.testing.assert_allclose(s['fut_disp'][k, p, j - 1],
                                                   self.fk[t + j, p] - self.fk[t, p], atol=1e-4)
                        checked += 1
        self.assertGreater(checked, 100)

    def test_targets_past_the_clip_end_are_masked(self):
        hi = self.n - T_WIN
        s = self.data.sample(self.index, start=hi)
        last = hi + T_WIN - 1                               # the clip's last frame
        self.assertFalse(s['fut_valid'][T_WIN - 1].any())  # nothing after it exists
        for j in range(1, H + 1):
            if last - 1 + j >= self.n:
                self.assertFalse(s['fut_valid'][T_WIN - 2, :, j - 1].any())

    def test_padded_points_have_no_targets_and_no_coordinates(self):
        s = self.data.sample(self.index, start=40)
        n_rob = s['n_rob']
        self.assertFalse(s['fut_valid'][:, n_rob:].any())
        self.assertFalse(s['coord_live'][:, n_rob:].any())
        self.assertFalse(s['coords'][:, n_rob:].any())

    def test_coordinates_are_the_tracked_positions_of_the_same_frames(self):
        st = 40
        s = self.data.sample(self.index, start=st)
        for r in range(M - 1 + T_WIN):
            f = st - (M - 1) + r
            for p in range(s['n_rob']):
                live = f >= self.birth[p]
                self.assertEqual(bool(s['coord_live'][r, p]), live)
                self.assertEqual(bool(s['coord_vis'][r, p]), bool(live and self.vis[f, p]))
                if live:
                    u, v = self.trk[f, p]
                    np.testing.assert_allclose(s['coords'][r, p], [u / IMG_W * 2 - 1, v / IMG_H * 2 - 1], atol=1e-5)

    def test_front_padded_rows_have_no_coordinates(self):
        s = self.data.sample(self.index, start=0)          # M-1 rows of front padding
        self.assertFalse(s['coord_live'][:M - 1].any())

    # ---- model ---------------------------------------------------------------------------
    def models(self, **kw):
        torch.manual_seed(7)
        base = ActionPolicy(**SMALL).eval()
        torch.manual_seed(7)
        new = PointPolicy(**SMALL, **kw).eval()
        return base, new

    def test_initialisation_equals_action_policy(self):
        b = tensors(batch_of(self.data, self.index, [30, 60]))
        base, new = self.models()
        with torch.no_grad():
            a, c = base(b), new(b)
        for key in ('delta_pos_body', 'delta_rotmat_body', 'gripper_logit'):
            self.assertTrue(torch.equal(a[key], c[key]), key)
        self.assertFalse(c['future_disp'].any())          # part 1 starts at "no motion"
        shared = {k: v for k, v in new.state_dict().items() if k in base.state_dict()}
        for k, v in base.state_dict().items():
            self.assertTrue(torch.equal(v, shared[k]), k)

    def test_model_never_reads_the_targets(self):
        b = tensors(batch_of(self.data, self.index, [30, 60]))
        _, new = self.models()
        self.perturb_new_layers(new)
        with torch.no_grad():
            out = new(b)
            b2 = dict(b, fut_disp=torch.randn_like(b['fut_disp']) * 1e3,
                      fut_valid=~b['fut_valid'])
            out2 = new(b2)
        for key in out:
            self.assertTrue(torch.equal(out[key], out2[key]), key)

    def test_causal_in_time(self):
        b = tensors(batch_of(self.data, self.index, [30]))
        _, new = self.models()
        self.perturb_new_layers(new)
        with torch.no_grad():
            out = new(b)
            b2 = dict(b)
            for key in ('robot', 'scene', 'coords'):
                x = b[key].clone()
                x[:, -1] = torch.randn_like(x[:, -1])       # scramble the LAST stream frame only
                b2[key] = x
            out2 = new(b2)
        for key in ('delta_pos_body', 'future_disp'):
            self.assertTrue(torch.equal(out[key][:, :-1], out2[key][:, :-1]), key)
            self.assertFalse(torch.equal(out[key][:, -1], out2[key][:, -1]), key)

    @staticmethod
    def perturb_new_layers(model):
        """Zero-initialised layers hide wiring; give them weights so their inputs matter."""
        g = torch.Generator().manual_seed(3)
        for mod in (model.coord_embed[-1], model.future_head[-1], model.plan_embed[-1]):
            with torch.no_grad():
                mod.weight.copy_(torch.randn(mod.weight.shape, generator=g) * 0.05)

    def test_every_new_layer_learns_and_pretraining_skips_the_action_head(self):
        b = tensors(batch_of(self.data, self.index, [30, 60]))
        b['weight'] = b['weight'].float()
        _, model = self.models()
        model.train()
        crit = PointPolicyLoss(0.0167, 0.0457, self.scales)
        opt = make_optimizer(model, crit, lr=1e-3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        _, norm = point_train_step(model, crit, b, opt, stage='points')
        self.assertIsNotNone(norm)
        after = model.state_dict()
        for k in before:
            moved = not torch.equal(before[k], after[k])
            if k.startswith(('head.', 'readout.', 'plan_embed.')):
                self.assertFalse(moved, f'{k} changed during point pre-training')
        self.assertFalse(torch.equal(before['future_head.3.weight'], after['future_head.3.weight']))
        for _ in range(3):
            point_train_step(model, crit, b, opt, stage='joint')
        final = model.state_dict()
        for k in before:
            if k.startswith(('coord_embed.', 'future_head.', 'plan_embed.')) and 'running' not in k:
                self.assertFalse(torch.equal(before[k], final[k]), f'{k} never changed')

    # ---- loss ----------------------------------------------------------------------------
    def test_point_loss_matches_hand_computed_values(self):
        crit = PointPolicyLoss(0.0167, 0.0457, [2.0] * H)
        B, T, N = 2, 3, 4
        disp = torch.zeros(B, T, N, H, 2)
        disp[..., 0] = 2.0                                  # every target: one scale unit along x
        valid = torch.ones(B, T, N, H, dtype=torch.bool)
        valid[1, :, :, :] = False
        disp[1] = float('nan')                              # masked targets must not matter
        batch = dict(fut_disp=disp, fut_valid=valid, robot_pad=torch.ones(B, N, dtype=torch.bool),
                     weight=torch.tensor([0.5, 1.0]))
        zero = {'future_disp': torch.zeros(B, T, N, H, 2)}
        self.assertAlmostEqual(float(crit.points(zero, batch)['points']), 1.0, places=6)
        exact = {'future_disp': torch.nan_to_num(disp) / 2.0}
        self.assertAlmostEqual(float(crit.points(exact, batch)['points']), 0.0, places=6)
        batch['weight'] = torch.tensor([5.0, 10.0])                     # rescaling weights changes nothing
        self.assertAlmostEqual(float(crit.points(zero, batch)['points']), 1.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
