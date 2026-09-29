#!/usr/bin/env python3
"""Step 4: check action-head conventions, rotation geometry, and gradient wiring.

Run with /scratch/oozak23/envs/track_on_r/bin/python scripts/49_verify_head.py.
Uses CPU and one existing cached clip; no dataset/checkpoint writes or training run.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.action_decoder import ActionHead, ActionMemory, ActionReadout, rotation_6d_to_matrix
from droid_qp.policy_data import PolicyWindows
from droid_qp.geom import quat_to_R


class HeadChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(49)

    def proper_rotation(self, R, atol=2e-6):
        self.assertTrue(torch.isfinite(R).all())
        eye = torch.eye(3, dtype=R.dtype).expand_as(R)
        torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, atol=atol, rtol=0)
        torch.testing.assert_close(torch.linalg.det(R), torch.ones_like(R[..., 0, 0]),
                                   atol=atol, rtol=0)

    def test_shapes_and_initial_motion(self):
        head = ActionHead()
        for k in (1, 4):
            out = head(torch.randn(2, 3, k, 256))
            for key, last in [('delta_pos_body', (3,)), ('delta_rot6d_body', (6,)),
                              ('delta_rotmat_body', (3, 3)), ('gripper_logit', (1,))]:
                self.assertEqual(out[key].shape, (2, 3, k, *last))
                self.assertTrue(torch.isfinite(out[key]).all())
            self.proper_rotation(out['delta_rotmat_body'])
            self.assertLess(out['delta_pos_body'].abs().max().item(), 0.01)
            self.assertLess((out['delta_rotmat_body'] - torch.eye(3)).abs().max().item(), 0.01)
            self.assertLess(out['gripper_logit'].abs().max().item(), 0.1)

    def test_rotation_column_order_against_known_turn(self):
        # Active +90 degrees about z maps +x to +y. A row/column mixup reverses it.
        d6 = torch.tensor([0., 1., 0., -1., 0., 0.])
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        torch.testing.assert_close(rotation_6d_to_matrix(d6), expected)

    def test_roundtrip_against_independent_rotations(self):
        # Independent quaternion formula already used by the label pipeline, not
        # this decoder's Gram-Schmidt construction. Requires no kinematics/SciPy env.
        q = np.random.default_rng(49).normal(size=(128, 4))
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        R = quat_to_R(q)
        d6 = np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
        recovered = rotation_6d_to_matrix(torch.from_numpy(d6))
        torch.testing.assert_close(recovered, torch.from_numpy(R), atol=1e-12, rtol=1e-12)
        self.proper_rotation(recovered, atol=1e-12)

    def test_rotation_degeneracies_and_low_precision(self):
        d6 = torch.tensor([[0., 0., 0., 0., 0., 0.],
                           [1., 0., 0., 2., 0., 0.],
                           [0., 1., 0., 0., -3., 0.],
                           [1., 1., 1., 2., 2., 2.00001],
                           [1e-9, 0., 0., 0., 1e-9, 0.]], requires_grad=True)
        R = rotation_6d_to_matrix(d6)
        self.proper_rotation(R)
        (R * torch.randn_like(R)).sum().backward()
        self.assertTrue(torch.isfinite(d6.grad).all())
        for dtype in (torch.float16, torch.bfloat16):
            converted = rotation_6d_to_matrix(d6.detach().to(dtype))
            self.assertEqual(converted.dtype, torch.float32)
            self.proper_rotation(converted)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = ActionHead(D=32)(torch.randn(2, 3, 1, 32))
        self.assertEqual(out['delta_rotmat_body'].dtype, torch.float32)
        self.proper_rotation(out['delta_rotmat_body'])

    def test_rotation_gradient_matches_finite_differences(self):
        d6 = (torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float64)
              + 0.1 * torch.randn(2, 6, dtype=torch.float64)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(rotation_6d_to_matrix, (d6,)))

    def test_head_outputs_are_unbounded_and_in_physical_units(self):
        head = ActionHead(D=32)
        with torch.no_grad():
            for p in head.parameters():
                p.zero_()
            head.position.bias.copy_(torch.tensor([0.02, -0.03, 0.04]))
            head.rotation.bias.copy_(torch.tensor([0., 1., 0., -1., 0., 0.]))
            head.gripper.bias.fill_(-5)
        out = head(torch.zeros(1, 1, 1, 32))
        torch.testing.assert_close(out['delta_pos_body'].flatten(), torch.tensor([0.02, -0.03, 0.04]))
        self.assertEqual(out['gripper_logit'].item(), -5)
        self.proper_rotation(out['delta_rotmat_body'])

    def test_each_branch_reaches_input_features(self):
        head = ActionHead(D=32)
        for key in ('delta_pos_body', 'delta_rotmat_body', 'gripper_logit'):
            x = torch.randn(2, 3, 1, 32, requires_grad=True)
            value = head(x)[key]
            (value - torch.randn_like(value)).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all(), key)
            self.assertGreater(x.grad.abs().max().item(), 0, key)

    def test_frames_and_query_slots_are_independent(self):
        head = ActionHead(D=32)
        x = torch.randn(2, 3, 4, 32)
        out = head(x)
        changed = x.clone()
        changed[0, 1, 2] += torch.randn(32)
        other = head(changed)
        untouched = torch.ones(2, 3, 4, dtype=torch.bool)
        untouched[0, 1, 2] = False
        for key in out:
            self.assertTrue(torch.equal(out[key][untouched], other[key][untouched]), key)
            self.assertGreater((out[key][0, 1, 2] - other[key][0, 1, 2]).abs().max().item(), 0, key)

    def test_invalid_shapes(self):
        head = ActionHead(D=32)
        for x in (torch.randn(2, 32), torch.randn(2, 3, 1, 16)):
            with self.assertRaises(ValueError):
                head(x)
        for x in (torch.ones(6, dtype=torch.int64), torch.randn(3), torch.tensor(1.)):
            with self.assertRaises(ValueError):
                rotation_6d_to_matrix(x)

    def test_real_window_forward_and_backward(self):
        data = PolicyWindows(T_win=3, M=4)
        sample = data.sample(0, start=0)
        robot = torch.from_numpy(sample['robot']).float().unsqueeze(0)
        scene = torch.from_numpy(sample['scene']).float().unsqueeze(0)
        hist = torch.from_numpy(sample['hist_valid']).unsqueeze(0)
        padding = ~torch.from_numpy(sample['robot_pad']).unsqueeze(0)
        target = torch.from_numpy(sample['target']).unsqueeze(0).unsqueeze(2)
        valid = torch.from_numpy(sample['target_valid']).unsqueeze(0).unsqueeze(2)
        self.assertTrue(valid.any())
        memory = ActionMemory(layers=1, M=4, scene_slots=4, checkpoint=True).train()
        readout, head = ActionReadout().train(), ActionHead().train()
        out = head(readout(memory(robot, scene, hist, padding, 3), padding))
        self.assertEqual(out['delta_pos_body'].shape, (1, 3, 1, 3))
        self.proper_rotation(out['delta_rotmat_body'])
        # Diagnostic objective only, to test wiring against actual labels. Step 5 owns
        # train-set group scales, reduction, validity rules, and clip weighting.
        target_R = rotation_6d_to_matrix(target[..., 3:9])
        diagnostic = ((out['delta_pos_body'][valid] - target[..., :3][valid]) / 0.01).square().mean()
        diagnostic += (out['delta_rotmat_body'][valid] - target_R[valid]).square().mean()
        diagnostic += F.binary_cross_entropy_with_logits(out['gripper_logit'][valid], target[..., 9:10][valid])
        diagnostic.backward()
        for module in (head, readout, memory):
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, f'{type(module).__name__}.{name}')
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().max().item(), 0, name)
        print(f"\nReal clip: {sample['video_id']}; finite gradients through head/readout/memory")


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
