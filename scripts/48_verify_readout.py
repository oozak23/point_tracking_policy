#!/usr/bin/env python3
"""Verify step 3 by perturbation and by composing it with ActionMemory.

Run: /scratch/oozak23/envs/track_on_r/bin/python scripts/48_verify_readout.py
CPU only; no checkpoints, dataset writes, or policy training required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droid_qp.action_decoder import ActionMemory, ActionReadout
from droid_qp.policy_data import PolicyWindows


class ReadoutChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(48)
        self.x = torch.randn(2, 4, 7, 32)
        # Different valid counts per example; padding is not confined to the end.
        self.mask = torch.tensor([[False, True, False, False, True, False, True],
                                  [True, False, True, True, False, True, True]])
        self.model = ActionReadout(D=32, heads=4, k=3).eval()

    def close(self, a, b):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    @torch.no_grad()
    def test_shapes_and_distinct_queries(self):
        for k in (1, 3):
            model = ActionReadout(D=32, k=k).eval()
            out = model(self.x, self.mask)
            self.assertEqual(out.shape, (2, 4, k, 32))
            self.assertTrue(torch.isfinite(out).all())
            for i in range(k):
                for j in range(i):
                    self.assertGreater((out[:, :, i] - out[:, :, j]).abs().max().item(), 1e-4)

    @torch.no_grad()
    def test_point_permutation_including_padding(self):
        out = self.model(self.x, self.mask)
        # Independently permute each example, moving padding along with its points.
        for b in range(2):
            perm = torch.randperm(7)
            other = self.model(self.x[b:b+1, :, perm], self.mask[b:b+1, perm])
            self.close(out[b:b+1], other)

    @torch.no_grad()
    def test_padding_values_and_padding_count_do_not_matter(self):
        out = self.model(self.x, self.mask)
        padded = self.mask[:, None, :, None].expand_as(self.x)
        for value in (1e6, float('nan')):
            self.close(out, self.model(self.x.masked_fill(padded, value), self.mask))
        # Compare against genuinely unpadded attention, separately for each example.
        for b in range(2):
            real = self.x[b:b+1, :, ~self.mask[b]]
            mask = torch.zeros(1, real.shape[2], dtype=torch.bool)
            self.close(out[b:b+1], self.model(real, mask))

    @torch.no_grad()
    def test_frames_are_independent_and_repeatable(self):
        out = self.model(self.x, self.mask)
        self.assertTrue(torch.equal(out, self.model(self.x, self.mask)))
        for t in range(4):
            self.close(out[:, t:t+1], self.model(self.x[:, t:t+1], self.mask))
        changed = self.x.clone()
        changed[:, 2] = torch.randn_like(changed[:, 2]) * 10
        other = self.model(changed, self.mask)
        self.assertTrue(torch.equal(out[:, [0, 1, 3]], other[:, [0, 1, 3]]))
        self.assertGreater((out[:, 2] - other[:, 2]).abs().max().item(), 1e-4)

    def test_backward_reaches_queries_and_real_points_only(self):
        self.model.train()
        x = self.x.clone().requires_grad_()
        out = self.model(x, self.mask)
        # A nonconstant target avoids the near-constant squared norm after LayerNorm.
        (out - torch.randn_like(out)).square().mean().backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().max().item(), 0, name)
        for query_grad in self.model.action_queries.weight.grad:
            self.assertGreater(query_grad.abs().max().item(), 0)
        padded = self.mask[:, None, :, None].expand_as(x)
        self.assertEqual(x.grad[padded].abs().max().item(), 0)
        self.assertGreater(x.grad[~padded].abs().max().item(), 0)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            ActionReadout(k=0)
        for x, mask in ((self.x[:, 0], self.mask),
                        (self.x[..., :16], self.mask),
                        (self.x, self.mask.float()),
                        (self.x, self.mask[:, :3]),
                        (self.x, torch.ones_like(self.mask)),
                        (self.x[:, :, :0], self.mask[:, :0])):
            with self.assertRaises(ValueError):
                self.model(x, mask)

    def test_memory_composition_and_scene_gradient(self):
        M, Tw, N, D = 3, 3, 7, 32
        L = M - 1 + Tw
        memory = ActionMemory(D=D, layers=1, M=M, scene_slots=M,
                              checkpoint=False).eval()
        robot = torch.randn(2, L, N, D, requires_grad=True)
        scene = torch.randn(2, L, 5, D, requires_grad=True)
        hist = torch.ones(2, L, dtype=torch.bool)
        hist[:, 0] = False  # entirely padded frame, then a mixed batch
        hist[0, 1] = False
        tokens = memory(robot, scene, hist, self.mask, Tw)
        out = self.model(tokens, self.mask)
        self.assertEqual(out.shape, (2, Tw, 3, D))
        (out - torch.randn_like(out)).square().mean().backward()
        for name, p in memory.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().max().item(), 0, name)
        self.assertTrue(torch.isfinite(scene.grad).all())
        self.assertGreater(scene.grad.abs().max().item(), 0)
        with torch.no_grad():
            # Perturb the final input frame; all earlier action tokens must agree.
            # Use the same grad mode for both passes: MHA can choose a different
            # inference kernel when gradients are disabled.
            reference = self.model(memory(robot, scene, hist, self.mask, Tw), self.mask)
            future = scene.detach().clone()
            future[:, -1] = torch.randn_like(future[:, -1]) * 10
            other = self.model(memory(robot, future, hist, self.mask, Tw), self.mask)
            self.assertTrue(torch.equal(reference[:, :-1], other[:, :-1]))
            self.assertGreater((reference[:, -1] - other[:, -1]).abs().max().item(), 1e-4)

        # Exercise the actual training branch with checkpoint recomputation as well.
        memory.train()
        memory.checkpoint = True
        memory.zero_grad(set_to_none=True)
        self.model.train()
        trained = self.model(memory(robot, scene, hist, self.mask, Tw), self.mask)
        (trained - torch.randn_like(trained)).square().mean().backward()
        for name, p in memory.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    @torch.no_grad()
    def test_real_cached_window(self):
        # Read one real cached clip at the production feature width, using a short
        # window/one replica layer to keep this integration check inexpensive on CPU.
        data = PolicyWindows(T_win=3, M=4)
        sample = data.sample(0, start=0)
        robot = torch.from_numpy(sample['robot']).float().unsqueeze(0)
        scene = torch.from_numpy(sample['scene']).float().unsqueeze(0)
        hist = torch.from_numpy(sample['hist_valid']).unsqueeze(0)
        padding = ~torch.from_numpy(sample['robot_pad']).unsqueeze(0)
        self.assertEqual((~padding).sum().item(), sample['n_rob'])
        memory = ActionMemory(layers=1, M=4, scene_slots=4, checkpoint=False).eval()
        model = ActionReadout().eval()
        out = model(memory(robot, scene, hist, padding, 3), padding)
        self.assertEqual(out.shape, (1, 3, 1, 256))
        self.assertTrue(torch.isfinite(out).all())
        print(f"\nReal clip: {sample['video_id']}, {sample['n_rob']} robot points; "
              f"output {tuple(out.shape)}")


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
