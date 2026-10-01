#!/usr/bin/env python3
"""Checks for action chunking (droid_qp/chunk_policy.py, run C of docs/POINT_POLICY_PLAN.md).

Targets: step j of frame t is the recorded action at frame t+j; step 0 is today's one-step target;
steps past the clip's end are masked. Model: never reads the targets, is causal, and every one of
the K queries learns. Loss: the first-step view equals the one-step loss; perfect predictions cost
zero. Evaluation: chaining actions is right (analytically and against recorded poses), and a
"model" that returns the labels scores zero error at every horizon.

    $PY_TRACKON -B checks/94_verify_chunk_policy.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.action_decoder import rotation_6d_to_matrix  # noqa: E402
from droid_qp.chunk_policy import (K, ChunkPolicyLoss, ChunkPolicyWindows, chunk_train_step,  # noqa: E402
                                   compose, evaluate_chunk)
from droid_qp.policy_loss import PolicyLoss  # noqa: E402
from droid_qp.policy_train import ActionPolicy, make_optimizer  # noqa: E402

T_WIN, M = 4, 4
SMALL = dict(layers=1, M=M, scene_slots=M, checkpoint=False, dropout=0.)


def batch_of(data, index, starts):
    b = default_collate([data.sample(index, start=s) for s in starts])
    return {k: v for k, v in b.items() if isinstance(v, torch.Tensor)}


class LabelModel(torch.nn.Module):
    """Returns the recorded chunk as its prediction: the evaluation must then report zero error."""

    def forward(self, batch):
        y = batch['chunk_target'].float()
        d6 = y[..., 3:9]
        return {'delta_pos_body': y[..., :3], 'delta_rot6d_body': d6,
                'delta_rotmat_body': rotation_6d_to_matrix(d6),
                'gripper_logit': (y[..., 9:] * 2 - 1) * 20}


class ChunkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ChunkPolicyWindows('train', T_win=T_WIN, M=M, cache_clips=4)
        cls.index = 0
        vid = cls.data.clips[0]['video_id']
        with np.load(cls.data.gen / cls.data.shard[vid] / 'actions/labels' / f'{vid}.npz') as a:
            cls.lab = {k: a[k] for k in ('delta_pos_body', 'delta_rot6d_body', 'gripper_open_next', 'valid',
                                         'position', 'rot_6d')}
        obj, _, tgt, _ = cls.data._load(vid)
        cls.n = min(len(obj), len(tgt))
        cls.crit = ChunkPolicyLoss(0.0167, 0.0457)

    # ---- targets ---------------------------------------------------------------------------
    def test_step_j_is_the_action_at_frame_t_plus_j(self):
        st = 40
        s = self.data.sample(self.index, start=st)
        full = np.concatenate([self.lab['delta_pos_body'], self.lab['delta_rot6d_body'],
                               self.lab['gripper_open_next']], 1)
        for k in range(T_WIN):
            for j in range(K):
                f = st + k + j
                expect = f < self.n and bool(self.lab['valid'][f])
                self.assertEqual(bool(s['chunk_valid'][k, j]), expect)
                if f < self.n:
                    np.testing.assert_allclose(s['chunk_target'][k, j], full[f], atol=1e-7)
        np.testing.assert_array_equal(s['chunk_target'][:, 0], s['target'])      # step 0 = today's target
        np.testing.assert_array_equal(s['chunk_valid'][:, 0], s['target_valid'])

    def test_steps_past_the_clip_end_are_masked(self):
        hi = self.n - T_WIN
        s = self.data.sample(self.index, start=hi)
        for k in range(T_WIN):
            for j in range(K):
                if hi + k + j >= self.n:
                    self.assertFalse(s['chunk_valid'][k, j])
                    self.assertFalse(s['chunk_target'][k, j].any())

    # ---- model -----------------------------------------------------------------------------
    def model(self):
        torch.manual_seed(5)
        return ActionPolicy(**SMALL, k=K).eval()

    def test_output_has_one_action_per_query(self):
        out = self.model()(batch_of(self.data, self.index, [30, 60]))
        self.assertEqual(tuple(out['delta_pos_body'].shape), (2, T_WIN, K, 3))
        self.assertEqual(tuple(out['delta_rotmat_body'].shape), (2, T_WIN, K, 3, 3))

    def test_model_never_reads_the_targets_and_is_causal(self):
        b = batch_of(self.data, self.index, [30])
        m = self.model()
        with torch.no_grad():
            out = m(b)
            out_t = m(dict(b, chunk_target=torch.randn_like(b['chunk_target']), chunk_valid=~b['chunk_valid'],
                           target=torch.randn_like(b['target'])))
            b2 = dict(b)
            for key in ('robot', 'scene'):
                x = b[key].clone()
                x[:, -1] = torch.randn_like(x[:, -1])           # scramble the LAST stream frame only
                b2[key] = x
            out_c = m(b2)
        for key in out:
            self.assertTrue(torch.equal(out[key], out_t[key]), key)
            self.assertTrue(torch.equal(out[key][:, :-1], out_c[key][:, :-1]), key)
        self.assertFalse(torch.equal(out['delta_pos_body'][:, -1], out_c['delta_pos_body'][:, -1]))

    def test_every_query_learns(self):
        b = batch_of(self.data, self.index, [30, 60])
        b['weight'] = b['weight'].float()
        m = self.model().train()
        opt = make_optimizer(m, self.crit, lr=1e-3)
        before = m.readout.action_queries.weight.detach().clone()
        _, norm = chunk_train_step(m, self.crit, b, opt)
        self.assertIsNotNone(norm)
        moved = (m.readout.action_queries.weight.detach() != before).any(1)
        self.assertTrue(bool(moved.all()), f'queries that did not move: {(~moved).nonzero().ravel().tolist()}')

    # ---- loss ------------------------------------------------------------------------------
    def test_first_step_view_equals_the_one_step_loss(self):
        b = batch_of(self.data, self.index, [30, 60])
        with torch.no_grad():
            pred = self.model()(b)
        w = b['weight'].float()
        a = self.crit(pred, b['target'], b['target_valid'], w)
        one = PolicyLoss(0.0167, 0.0457)({k: v[:, :, :1] for k, v in pred.items()}, b['target'], b['target_valid'], w)
        for k in ('loss', 'position', 'rotation', 'gripper'):
            self.assertEqual(float(a[k]), float(one[k]))

    def test_perfect_chunk_costs_nothing_in_position_and_rotation(self):
        b = batch_of(self.data, self.index, [30, 60])
        b['weight'] = b['weight'].float()
        terms = self.crit.chunk(LabelModel()(b), b)
        self.assertLess(float(terms['position']), 1e-10)
        self.assertLess(float(terms['rotation']), 1e-8)
        self.assertLess(float(terms['gripper']), 1e-6)

    # ---- evaluation ------------------------------------------------------------------------
    def test_compose_of_known_steps(self):
        dp = torch.tensor([0.01, 0.0, 0.0]).expand(4, 3)
        eye = torch.eye(3).expand(4, 3, 3)
        p, R = compose(dp, eye)
        self.assertTrue(torch.allclose(p[:, 0], torch.tensor([0.01, 0.02, 0.03, 0.04])))
        rz = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]).expand(4, 3, 3)      # +90 deg about z per step
        p, R = compose(dp, rz)
        self.assertTrue(torch.allclose(p[1], torch.tensor([0.01, 0.01, 0.0]), atol=1e-7))   # second step heads +y
        self.assertTrue(torch.allclose(R[3], torch.eye(3), atol=1e-6))                       # four quarter turns

    def test_compose_of_labels_matches_recorded_poses(self):
        pos = torch.tensor(self.lab['position'], dtype=torch.float64)
        R = rotation_6d_to_matrix(torch.tensor(self.lab['rot_6d'], dtype=torch.float64))
        dp = torch.tensor(self.lab['delta_pos_body'], dtype=torch.float64)
        dR = rotation_6d_to_matrix(torch.tensor(self.lab['delta_rot6d_body'], dtype=torch.float64))
        checked = 0
        for t in range(20, 60):
            if t + K >= len(pos) or not self.lab['valid'][t:t + K].all():
                continue
            p, _ = compose(dp[t:t + K], dR[t:t + K])
            for h in (1, 4, 8, 16):
                want = R[t].T @ (pos[t + h] - pos[t])
                self.assertLess(float((p[h - 1] - want).abs().max()), 2e-4)
            checked += 1
        self.assertGreater(checked, 10)

    def test_a_model_that_returns_the_labels_scores_zero(self):
        b = batch_of(self.data, self.index, [30, 60])
        b['weight'] = b['weight'].float()
        out = evaluate_chunk(LabelModel(), self.crit, [b], torch.device('cpu'))
        for h, row in out['chunk'].items():
            self.assertLess(row['step_mm'], 1e-4, h)
            self.assertLess(row['endpoint_mm_model'], 1e-3, h)
            self.assertGreater(row['endpoint_mm_zero'], 0.1, h)      # the gripper really moves here
        self.assertLess(out['chunk_position'], 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
