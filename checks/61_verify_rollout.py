#!/usr/bin/env python3
"""Analytical composition checks plus a real-clip oracle integration fixture."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from droid_qp.actions import sixd_to_rot
from droid_qp.policy_rollout import integrate, place_local_points, pose_error


from droid_qp.paths import DATA  # noqa: E402


class RolloutChecks(unittest.TestCase):
    def test_rotating_body_translation(self):
        quarter = np.array([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
        p, R = integrate(np.zeros(3), np.eye(3), np.array([[1., 0, 0], [1., 0, 0]]),
                         np.array([quarter, quarter]))
        np.testing.assert_allclose(p, [1., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(R, quarter @ quarter, atol=1e-12)
        np.testing.assert_allclose(place_local_points([[1., 0, 0]], p, R), [[0., 1, 0]])
        self.assertEqual(pose_error(p, R, p, R), (0., 0.))

    def test_real_oracle_predictions_and_projection(self):
        plan_path = DATA / 'validation_plan_v1.json'
        plan = json.loads(plan_path.read_text())
        row = plan['records'][0]
        vid, start = row['video_id'], row['start']
        shards = json.loads((DATA / 'shard_index.json').read_text())
        with np.load(DATA / 'gen'/shards[vid]/'actions/labels'/f'{vid}.npz') as f:
            a = {k: f[k] for k in f.files}
        with tempfile.TemporaryDirectory(prefix='rollout-oracle-', dir='/tmp') as temp:
            d = Path(temp)
            (d/'predictions').mkdir()
            (d/'report.json').write_text(json.dumps(dict(checkpoint='analytical-oracle-fixture',
                checkpoint_step=0, plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())))
            sl = slice(start, start+48)
            np.savez(d/'predictions'/f'{vid}.npz', start=start, delta_pos_body=a['delta_pos_body'][sl],
                     delta_rotmat_body=sixd_to_rot(a['delta_rot6d_body'][sl]),
                     gripper_logit=a['gripper_open_next'][sl].reshape(-1)*40-20, valid=a['valid'][sl])
            env = dict(os.environ, MPLCONFIGDIR='/tmp/oozak23-mpl', PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, str(ROOT/'evaluation/60_evaluate_rollout.py'), '--evaluation',
                            str(d), '--out', str(d/'rollout.json'), '--limit', '1'], env=env, check=True)
            report = json.loads((d/'rollout.json').read_text())
            self.assertLess(report['qa']['max_saved_fk_reprojection_px'], .001)
            one = report['horizons']['1']['aggregate']
            expected_copy_mm = np.linalg.norm(a['delta_pos_body'][start:start+48] -
                                               a['delta_pos_body'][start-1:start+47], axis=-1).mean()*1000
            self.assertAlmostEqual(one['recorded_previous/position_mm']['mean'], expected_copy_mm, places=4)
            self.assertEqual(one['recorded_previous/position_mm'], one['hold_previous/position_mm'])
            for h in report['horizons'].values():
                self.assertLess(h['aggregate']['policy/position_mm']['mean'], .01)
                self.assertLess(h['aggregate']['policy/oracle_width/pixel_error']['mean'], .001)
                self.assertGreater(h['aggregate']['policy/position_mm']['count'], 0)
                self.assertEqual(set(h['per_lab']), {row['lab']})


if __name__ == '__main__':
    unittest.main(verbosity=2)
