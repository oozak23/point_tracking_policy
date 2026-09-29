#!/usr/bin/env python3
"""Analytical validation baselines/events, intervention alignment and real-data checks."""
import json
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_data import PolicyWindows
from droid_qp.policy_eval import (ValidationMetrics, ValidationWindows, evaluate_validation,
                                 events, event_report, match_events, replace_robot_history)
from droid_qp.policy_loss import PolicyLoss
from droid_qp.policy_train import ActionPolicy


from droid_qp.paths import DATA  # noqa: E402


class ValidationChecks(unittest.TestCase):
    def test_event_matching_and_penalties(self):
        perfect = event_report(match_events([3, 8], [3, 8]))
        self.assertEqual(perfect['timing_cost_frames'], 0)
        delayed = event_report(match_events([3, 8], [4, 9]))
        self.assertEqual(delayed['timing_cost_frames'], 1)
        self.assertEqual(delayed['f1'], 1)
        missed = event_report(match_events([3, 8], []))
        self.assertEqual(missed['timing_cost_frames'], 3)
        extra = event_report(match_events([3], [3, 4]))
        self.assertEqual(extra['timing_cost_frames'], 3)
        # Maximum-cardinality assignment, rather than nearest-edge greedy matching.
        self.assertEqual(match_events([2, 4], [0, 3])['matched'], 2)

    def test_copy_baseline_common_mask_and_transition_lag(self):
        y = torch.zeros(1, 3, 10)
        y[..., 3] = y[..., 7] = 1
        y[0, :, 0] = torch.tensor([1., 2., 3.])
        y[0, :, 9] = torch.tensor([0., 1., 1.])
        previous = y[..., :9].clone()
        previous[0, :, 0] -= 1
        batch = dict(target=y, target_valid=torch.ones(1, 3, dtype=torch.bool),
                     weight=torch.ones(1), previous_action=previous,
                     previous_valid=torch.tensor([[False, True, True]]),
                     gripper_current=torch.tensor([[0., 0., 1.]]))
        pred = dict(delta_pos_body=previous[..., :3].unsqueeze(2),
                    delta_rotmat_body=torch.eye(3).expand(1, 3, 1, 3, 3),
                    gripper_logit=(y[..., 9]*40-20)[:, :, None, None])
        metric = ValidationMetrics()
        metric.add(pred, batch, None)
        r = metric.result()
        self.assertEqual(r['n_copy_comparable'], 2)
        self.assertEqual(r['copy_position_mm'], 1000)
        self.assertEqual(r['common_position_mm'], 1000)
        self.assertEqual(r['common_zero_mm'], 2500)
        self.assertEqual(r['events']['policy']['all']['timing_cost_frames'], 0)
        self.assertEqual(r['events']['persistence']['all']['timing_cost_frames'], 1)

    def test_history_shuffle_preserves_current_frame_and_point_ids(self):
        source = torch.randn(2, 5, 4, 3)
        donor = torch.arange(2*5*4*3).reshape(2, 5, 4, 3).float()
        maps = [torch.tensor([3, 1]), torch.tensor([2, 0, 1])]
        changed = replace_robot_history(source, donor, maps)
        self.assertTrue(torch.equal(source[:, -1], changed[:, -1]))
        torch.testing.assert_close(changed[0, :-1, :2], donor[0, :-1, maps[0]])
        self.assertTrue(torch.equal(source[0, :, 2:], changed[0, :, 2:]))

    def test_real_validation_and_short_clip(self):
        plan = json.loads((DATA / 'validation_plan_v1.json').read_text())
        short = next(r for r in plan['records'] if r['n_frames'] < 48)
        records = [plan['records'][0], short]
        dataset = ValidationWindows(PolicyWindows('val', T_win=48, M=24), records)
        sample = dataset[1]
        self.assertEqual(sample['target'].shape, (48, 10))
        self.assertEqual(sample['gripper_current'].shape, (48,))
        self.assertFalse(sample['previous_valid'][short['n_frames']:].any())
        model = ActionPolicy(layers=1, dropout=0., checkpoint=False)
        criterion = PolicyLoss.from_json(DATA / 'policy_loss_scales.json')
        report = evaluate_validation(model, criterion, DataLoader(dataset, batch_size=2), torch.device('cpu'))
        self.assertEqual(report['n_targets'], sum(int(dataset[i]['target_valid'].sum()) for i in range(2)))
        self.assertEqual(set(report['per_lab']), {r['lab'] for r in records})
        self.assertEqual(report['n_targets'], sum(v['n_targets'] for v in report['per_lab'].values()))
        self.assertTrue(torch.isfinite(torch.tensor(report['loss'])))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
