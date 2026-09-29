#!/usr/bin/env python3
"""Current-state alignment, causal late fusion, matched initialization and optimization."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.policy_proprio import (ProprioActionPolicy, ProprioPolicyWindows,
                                   current_state_window, STATE_SCALES)
from droid_qp.policy_train import ActionPolicy, make_optimizer, train_step
from droid_qp.policy_loss import PolicyLoss


def batch():
    y = torch.zeros(1, 3, 10)
    y[..., 0] = .01
    y[..., 3] = y[..., 7] = 1
    y[..., 9] = torch.tensor([0., 1., 0.])
    return dict(robot=torch.randn(1, 5, 3, 32), scene=torch.randn(1, 5, 2, 32),
                robot_pad=torch.ones(1, 3, dtype=torch.bool), hist_valid=torch.ones(1, 5,dtype=torch.bool),
                target=y, target_valid=torch.ones(1,3,dtype=torch.bool), weight=torch.ones(1),
                proprio=torch.randn(1,3,15))


class Checks(unittest.TestCase):
    def test_current_frame_alignment_and_no_target_access(self):
        labels = dict(joint_positions=np.repeat(np.arange(10)[:,None],7,axis=1),
                      joint_velocities=np.repeat((100+np.arange(10))[:,None],7,axis=1),
                      gripper_positions=np.arange(10)[:,None]/10)
        a = current_state_window(labels, 3, 4, np.ones(4,bool))
        np.testing.assert_equal(a[:,0],[3,4,5,6])
        np.testing.assert_equal(a[:,7],[103,104,105,106])
        np.testing.assert_allclose(a[:,-1],[.3,.4,.5,.6])
        labels['joint_positions'][6:] = 9999
        labels['joint_velocities'][6:] = 9999
        labels['gripper_positions'][6:] = 9999
        labels['delta_pos_body'] = np.full((10,3),np.nan)
        labels['gripper_open_next'] = np.full((10,1),np.nan)
        b = current_state_window(labels,3,4,np.ones(4,bool))
        np.testing.assert_equal(a[:3],b[:3])
        self.assertFalse(np.array_equal(a[3],b[3]))

    def test_padding_and_missing_state_fail_closed(self):
        labels = dict(joint_positions=np.zeros((2,7)),joint_velocities=np.zeros((2,7)),
                      gripper_positions=np.zeros((2,1)))
        self.assertEqual(current_state_window(labels,1,3,np.array([True,False,False])).shape,(3,15))
        with self.assertRaises(ValueError):
            current_state_window(labels,1,3,np.ones(3,bool))
        labels['joint_velocities'][1,0] = np.nan
        with self.assertRaises(ValueError):
            current_state_window(labels,1,1,np.ones(1,bool))

    def test_shared_initialization_and_rng_match_visual_baseline(self):
        torch.manual_seed(2026)
        visual = ActionPolicy(D=32,layers=1,M=3,scene_slots=3,checkpoint=False,dropout=.1)
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(2026)
        model = ProprioActionPolicy(D=32,layers=1,M=3,scene_slots=3,checkpoint=False,dropout=.1)
        self.assertTrue(torch.equal(baseline_rng,torch.get_rng_state()))
        for k,v in visual.state_dict().items():
            self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        b = batch()
        visual.eval(); model.eval()
        with torch.no_grad():
            a,c = visual(b),model(b)
        for k in a:
            torch.testing.assert_close(a[k],c[k],rtol=0,atol=0)
        torch.testing.assert_close(model.proprio_scales,torch.tensor(STATE_SCALES))

    def test_future_state_cannot_change_earlier_predictions(self):
        torch.manual_seed(70)
        model = ProprioActionPolicy(D=32,layers=1,M=3,scene_slots=3,checkpoint=False,dropout=0).eval()
        torch.nn.init.normal_(model.proprio_encoder[-1].weight,std=.1)
        b = batch(); changed = dict(b,proprio=b['proprio'].clone())
        changed['proprio'][:,2] += 10
        with torch.no_grad():
            a,c = model(b),model(changed)
        for k in a:
            torch.testing.assert_close(a[k][:,:2],c[k][:,:2],rtol=0,atol=0)
        self.assertFalse(torch.equal(a['delta_pos_body'][:,2],c['delta_pos_body'][:,2]))

    def test_encoder_learns_and_checkpoint_roundtrips(self):
        model = ProprioActionPolicy(D=32,layers=1,M=3,scene_slots=3,checkpoint=False,dropout=0)
        criterion = PolicyLoss(.016,.045)
        opt = make_optimizer(model,criterion,lr=3e-4)
        b = batch()
        for _ in range(2):
            terms,norm = train_step(model,criterion,b,opt)
            self.assertTrue(torch.isfinite(terms['loss']))
            self.assertGreater(norm,0)
        for index in (0,2):
            self.assertGreater(float(model.proprio_encoder[index].weight.grad.abs().sum()),0)
        clone = ProprioActionPolicy(D=32,layers=1,M=3,scene_slots=3,checkpoint=False,dropout=0)
        clone.load_state_dict(model.state_dict())
        for k,v in model(b).items():
            torch.testing.assert_close(v,clone(b)[k],rtol=0,atol=0)

    def test_real_window_and_short_clip_alignment(self):
        plan = json.loads((ROOT/'out/validation_plan_v1.json').read_text())
        data = ProprioPolicyWindows('val',T_win=48,M=24)
        for row in [plan['records'][0],next(r for r in plan['records'] if r['n_frames']<48)]:
            b = data.sample(row['index'],row['start']); live = b['hist_valid'][-48:]
            with np.load(data.gen/data.shard[row['video_id']]/'actions/labels'/f"{row['video_id']}.npz") as a:
                indices = b['start']+np.flatnonzero(live)
                expected = np.concatenate([a[k][indices] for k in ['joint_positions','joint_velocities','gripper_positions']],axis=1)
            np.testing.assert_allclose(b['proprio'][live],expected,atol=1e-6)
            np.testing.assert_equal(b['proprio'][~live],0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
