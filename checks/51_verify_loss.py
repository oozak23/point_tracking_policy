#!/usr/bin/env python3
"""Step 5: analytical loss checks, split isolation, and real-decoder backward.

Run 50_compute_loss_scales.py first, then this script in the track_on_r environment.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.action_decoder import ActionHead, ActionMemory, ActionReadout
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_loss import PolicyLoss, compute_loss_scales


def example(B=2, T=2, k=1):
    pos = torch.zeros(B, T, k, 3, requires_grad=True)
    rot = torch.eye(3).expand(B, T, k, 3, 3).clone().requires_grad_()
    grip = torch.zeros(B, T, k, 1, requires_grad=True)
    target = torch.zeros(B, T, k, 10)
    target[..., 3] = target[..., 7] = 1  # identity 6D starts at target coordinate 3
    valid = torch.ones(B, T, k, dtype=torch.bool)
    return {'delta_pos_body': pos, 'delta_rotmat_body': rot, 'gripper_logit': grip}, target, valid


from droid_qp.paths import DATA  # noqa: E402


class LossChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)

    def test_analytical_weighting_rotation_and_gradients(self):
        pred, y, valid = example(T=1)
        with torch.no_grad():
            pred['delta_pos_body'][0, 0, 0, 0] = 1
            pred['delta_pos_body'][1, 0, 0, 0] = 2
            pred['delta_rotmat_body'][1, 0, 0] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        result = PolicyLoss(1, 1)(pred, y, valid, torch.tensor([1., 3.]))
        self.assertAlmostEqual(result['position'].item(), 3.25)
        self.assertAlmostEqual(result['rotation'].item(), 1.5)
        self.assertAlmostEqual(result['gripper'].item(), math.log(2), places=6)
        self.assertAlmostEqual(result['loss'].item(), 4.75 + math.log(2), places=6)
        self.assertEqual(result['weight_sum'].item(), 4)
        result['loss'].backward()
        torch.testing.assert_close(pred['delta_pos_body'].grad[:, 0, 0, 0], torch.tensor([0.5, 3.]))

    def test_perfect_predictions_and_scale_units(self):
        pred, y, valid = example()
        with torch.no_grad():
            pred['gripper_logit'].fill_(-100)
        result = PolicyLoss(0.02, 0.1)(pred, y, valid, torch.ones(2))
        self.assertLess(result['loss'].item(), 1e-6)
        with torch.no_grad():
            pred['delta_pos_body'][..., 0] = 0.02
        result = PolicyLoss(0.02, 0.1)(pred, y, valid, torch.ones(2))
        self.assertAlmostEqual(result['position'].item(), 1)

    def test_masks_exclude_nan_labels_and_predictions(self):
        pred, y, valid = example()
        valid[:, 1] = False
        with torch.no_grad():
            for p in pred.values():
                p[:, 1] = float('nan')
        y[:, 1] = float('nan')
        result = PolicyLoss(1, 1)(pred, y, valid, torch.tensor([1., 3.]))
        self.assertAlmostEqual(result['loss'].item(), math.log(2), places=6)
        self.assertEqual(result['weight_sum'].item(), 4)
        result['loss'].backward()
        for p in pred.values():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertEqual(p.grad[:, 1].abs().sum().item(), 0)

    def test_zero_weight_and_empty_batches(self):
        for empty_mask in (True, False):
            pred, y, valid = example()
            with torch.no_grad():
                for p in pred.values():
                    p.fill_(float('nan'))
            y.fill_(float('nan'))
            if empty_mask:
                valid.fill_(False)
            weights = torch.ones(2) if empty_mask else torch.zeros(2)
            result = PolicyLoss(1, 1)(pred, y, valid, weights)
            self.assertEqual(result['loss'].item(), 0)
            self.assertEqual(result['weight_sum'].item(), 0)
            self.assertEqual(result['n_valid'].item(), 0)
            result['loss'].backward()
            for p in pred.values():
                self.assertEqual(p.grad.abs().sum().item(), 0)

    def test_weight_rescaling_and_zero_weight_exclusion(self):
        pred, y, valid = example()
        y[0, ..., :3] = 2
        criterion = PolicyLoss(1, 1)
        a = criterion(pred, y, valid, torch.tensor([0.2, 0.7]))
        b = criterion(pred, y, valid, torch.tensor([2., 7.]))
        torch.testing.assert_close(a['loss'], b['loss'])
        y[1] = float('nan')
        a = criterion(pred, y, valid, torch.tensor([1., 0.]))
        self.assertAlmostEqual(a['position'].item(), 12)
        self.assertEqual(a['n_valid'].item(), 2)

    def test_single_step_and_explicit_chunk_alignment(self):
        criterion = PolicyLoss(1, 1)
        pred, y, valid = example()
        a = criterion(pred, y, valid, torch.ones(2))
        b = criterion(pred, y.squeeze(2), valid.squeeze(2), torch.ones(2))
        torch.testing.assert_close(a['loss'], b['loss'])
        pred, y, valid = example(k=2)
        y[:, :, 0, 0], y[:, :, 1, 0] = 1, 2
        result = criterion(pred, y, valid, torch.ones(2))
        self.assertAlmostEqual(result['position'].item(), 2.5)
        with self.assertRaises(ValueError):
            criterion(pred, y[:, :, 0], valid[:, :, 0], torch.ones(2))

    def test_invalid_inputs_and_active_nan_fail(self):
        for scale in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PolicyLoss(scale, 1)
        criterion = PolicyLoss(1, 1)
        pred, y, valid = example()
        for weight in (torch.ones(2, 1), torch.tensor([-1., 1.]), torch.tensor([float('nan'), 1.])):
            with self.assertRaises(ValueError):
                criterion(pred, y, valid, weight)
        with self.assertRaises(ValueError):
            criterion(pred, y, valid.float(), torch.ones(2))
        y[..., 9] = 0.4
        with self.assertRaises(ValueError):
            criterion(pred, y, valid, torch.ones(2))
        y[..., 9] = float('nan')
        with self.assertRaises(ValueError):
            criterion(pred, y, valid, torch.ones(2))

    def test_scales_are_checkpoint_buffers(self):
        criterion = PolicyLoss(0.02, 0.1)
        other = PolicyLoss(1, 1)
        other.load_state_dict(criterion.state_dict())
        self.assertEqual(len(list(other.parameters())), 0)
        torch.testing.assert_close(other.position_scale_m, criterion.position_scale_m)
        torch.testing.assert_close(other.rotation_scale, criterion.rotation_scale)

    def test_train_only_scale_statistics_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'gen/0000/actions/labels'
            labels.mkdir(parents=True)
            rows = [dict(video_id='a', episode='ep_a', split='train', weight=1),
                    dict(video_id='b', episode='ep_b', split='train', weight=3),
                    dict(video_id='val', episode='ep_val', split='val', weight=999)]
            manifest, index = root / 'manifest.json', root / 'index.json'
            manifest.write_text(json.dumps({'clips': rows}))
            # No validation label file or index entry exists: reading val would fail.
            index.write_text(json.dumps({'a': '0000', 'b': '0000'}))
            for vid, p, r in [('a', 1., [1, 0, 0, 0, 1, 0]), ('b', 2., [0, 1, 0, -1, 0, 0])]:
                np.savez(labels / f'{vid}.npz', valid=np.array([True, False]),
                         delta_pos_body=np.array([[p, 0, 0], [np.nan]*3]),
                         delta_rot6d_body=np.array([r, [np.nan]*6]))
            report = compute_loss_scales(manifest, root / 'gen', index)
            self.assertEqual(report['n_clips'], 2)
            self.assertEqual(report['n_valid_frames'], 2)
            self.assertAlmostEqual(report['position_scale_m'], math.sqrt(3.25))
            self.assertAlmostEqual(report['rotation_scale'], math.sqrt(1.5))
            artifact = root / 'scales.json'
            artifact.write_text(json.dumps(report))
            criterion = PolicyLoss.from_json(artifact, manifest=manifest)
            # Independent no-motion baseline must give each calibrated component 1.
            pred, y, valid = example(T=1)
            y[0, 0, 0, 0], y[1, 0, 0, 0] = 1, 2
            y[1, 0, 0, 3:9] = torch.tensor([0., 1., 0., -1., 0., 0.])
            result = criterion(pred, y, valid, torch.tensor([1., 3.]))
            self.assertAlmostEqual(result['position'].item(), 1, places=6)
            self.assertAlmostEqual(result['rotation'].item(), 1, places=6)
            rows[2]['weight'] = 5
            manifest.write_text(json.dumps({'clips': rows}))
            again = compute_loss_scales(manifest, root / 'gen', index)
            self.assertEqual(report['position_scale_m'], again['position_scale_m'])
            self.assertEqual(report['rotation_scale'], again['rotation_scale'])
            with self.assertRaises(ValueError):
                PolicyLoss.from_json(artifact, manifest=manifest)
            rows[2]['episode'] = 'ep_a'
            manifest.write_text(json.dumps({'clips': rows}))
            with self.assertRaises(ValueError):
                compute_loss_scales(manifest, root / 'gen', index)

    def test_actual_loss_through_real_decoder_with_autocast(self):
        criterion = PolicyLoss.from_json(DATA / 'policy_loss_scales.json',
                                        manifest=DATA / 'dataset_manifest.json')
        sample = PolicyWindows(T_win=3, M=4).sample(0, start=0)
        robot = torch.from_numpy(sample['robot']).float().unsqueeze(0)
        scene = torch.from_numpy(sample['scene']).float().unsqueeze(0)
        hist = torch.from_numpy(sample['hist_valid']).unsqueeze(0)
        padding = ~torch.from_numpy(sample['robot_pad']).unsqueeze(0)
        target = torch.from_numpy(sample['target']).unsqueeze(0)
        valid = torch.from_numpy(sample['target_valid']).unsqueeze(0)
        weight = torch.tensor([sample['weight']])
        memory = ActionMemory(layers=1, M=4, scene_slots=4, checkpoint=True).train()
        readout, head = ActionReadout().train(), ActionHead().train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pred = head(readout(memory(robot, scene, hist, padding, 3), padding))
            result = criterion(pred, target, valid, weight)
        self.assertEqual(result['loss'].dtype, torch.float32)
        result['loss'].backward()
        for module in (head, readout, memory):
            for name, p in module.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(p.grad.abs().max().item(), 0, name)
        print('\nReal-window loss:', {k: float(v.detach()) for k, v in result.items()})


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
