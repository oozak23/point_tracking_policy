#!/usr/bin/env python3
"""Small CPU checks of actual optimizer updates, metrics, scheduling and checkpointing."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_loss import PolicyLoss
from droid_qp.policy_train import ActionPolicy, Metrics, lr_factor, make_optimizer, save_checkpoint, train_step


def batch():
    target = torch.zeros(2, 3, 10)
    target[..., :3] = 0.01 * torch.randn(2, 3, 3)
    target[..., 3] = target[..., 7] = 1
    target[..., 9] = torch.tensor([[0., 1., 1.], [1., 0., 0.]])
    return dict(robot=torch.randn(2, 5, 4, 32), scene=torch.randn(2, 5, 3, 32),
                hist_valid=torch.tensor([[False, False, True, True, True],
                                         [False, True, True, True, True]]),
                robot_pad=torch.tensor([[True, True, False, False], [True, True, True, False]]),
                target=target, target_valid=torch.ones(2, 3, dtype=torch.bool),
                weight=torch.tensor([1., .5]), gripper_current=1-target[..., 9])


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(54)
        self.model = ActionPolicy(D=32, layers=1, M=3, scene_slots=3, checkpoint=False, dropout=0.)
        self.loss = PolicyLoss(.016, .045)
        self.optimizer = make_optimizer(self.model, self.loss, lr=1e-4, weight_decay=0.01)

    def test_physical_projection_learning_rates(self):
        groups = {g['name']: g['lr'] for g in self.optimizer.param_groups}
        self.assertAlmostEqual(groups['memory_readout_gripper'], 1e-4)
        self.assertAlmostEqual(groups['position'], 1.6e-6)
        self.assertAlmostEqual(groups['rotation'], 4.5e-6)

    def test_real_optimizer_changes_all_modules(self):
        b = batch()
        before = {k: v.detach().clone() for k, v in self.model.named_parameters()}
        terms, norm = train_step(self.model, self.loss, b, self.optimizer)
        self.assertGreater(norm, 0)
        self.assertTrue(torch.isfinite(terms['loss']))
        for prefix in ('memory.', 'readout.', 'head.'):
            self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.model.named_parameters()
                                if k.startswith(prefix)), prefix)

    def test_empty_batch_skips_weight_decay_and_optimizer(self):
        b = batch()
        b['target_valid'].fill_(False)
        before = {k: v.detach().clone() for k, v in self.model.named_parameters()}
        _, norm = train_step(self.model, self.loss, b, self.optimizer)
        self.assertIsNone(norm)
        self.assertEqual(len(self.optimizer.state), 0)
        for k, v in self.model.named_parameters():
            self.assertTrue(torch.equal(before[k], v), k)

    def test_scheduler_warmup_and_cosine(self):
        self.assertEqual(lr_factor(0, 100, 10), .1)
        self.assertEqual(lr_factor(9, 100, 10), 1)
        self.assertEqual(lr_factor(10, 100, 10), 1)
        self.assertAlmostEqual(lr_factor(100, 100, 10), .05)
        self.assertGreater(lr_factor(20, 100, 10), lr_factor(50, 100, 10))

    def test_checkpoint_restores_next_update(self):
        b = batch()
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: lr_factor(s, 100, 0))
        train_step(self.model, self.loss, b, self.optimizer)
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            save_checkpoint(path, dict(model=self.model.state_dict(), criterion=self.loss.state_dict(),
                                      optimizer=self.optimizer.state_dict(), scheduler=scheduler.state_dict()))
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
            other = ActionPolicy(D=32, layers=1, M=3, scene_slots=3, checkpoint=False, dropout=0.)
            opt = make_optimizer(other, self.loss, lr=1e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 100, 0))
            state = torch.load(path, weights_only=True)
            other.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
            train_step(self.model, self.loss, b, self.optimizer)
            train_step(other, self.loss, b, opt)
            for p, q in zip(self.model.parameters(), other.parameters()):
                self.assertTrue(torch.equal(p, q))

    def test_rotation_metric_at_identity_and_half_turn(self):
        R = torch.stack([torch.eye(3), torch.diag(torch.tensor([-1., -1., 1.]))])
        torch.testing.assert_close(Metrics.angle(R), torch.tensor([0., 180.]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
