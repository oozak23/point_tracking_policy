#!/usr/bin/env python3
"""Test 0 of the points-to-actions experiment: the converter itself, before any real data.

Every check has an answer known in advance: exact recovery of known poses and motions, and
agreement with the action-label convention on a real clip. If these fail, the experiment's
numbers mean nothing.

    $PY_POINTWORLD -B checks/82_verify_pose_fit.py
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from droid_qp.paths import DATA  # noqa: E402
from droid_qp.actions import sixd_to_rot  # noqa: E402
from droid_qp.pose_fit import (gripper_model, kabsch, pnp, relative, repeat,  # noqa: E402
                               rotation_error_deg, scaled_intrinsics)
from droid_qp.robot_points import load_sampler, robot_points_3d  # noqa: E402

RNG = np.random.default_rng(0)


def random_rotation():
    q = RNG.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PoseFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = load_sampler(128, gripper_only=True, seed=0)
        cls.model, cls.rigid = gripper_model(cls.sampler)
        cls.palm = cls.model[cls.rigid]
        # A realistic 854x480 camera: PointWorld-style 320x180 intrinsics, scaled.
        cls.K = scaled_intrinsics(np.array([[230.0, 0, 160], [0, 230.0, 90], [0, 0, 1]]))

    def test_palm_points_are_rigid_and_fingers_move(self):
        self.assertEqual(int(self.rigid.sum()), 68)
        q = np.array([[0.3, -0.4, 0.1, -2.0, 0.2, 1.7, 0.5]] * 4)
        g = np.linspace(0, 1, 4)[:, None]
        P = robot_points_3d(self.sampler, q, g)
        spread = P.max(0) - P.min(0)                      # arm fixed, only the gripper moves
        self.assertLess(np.abs(spread[self.rigid]).max(), 1e-6)
        self.assertGreater(np.linalg.norm(spread[~self.rigid], axis=-1).max(), 0.010)

    def test_kabsch_recovers_known_pose(self):
        for _ in range(50):
            R, t = random_rotation(), RNG.normal(size=3)
            Rh, th = kabsch(self.palm, self.palm @ R.T + t)
            # arccos cannot resolve angles much below 1e-6 degrees, whatever the fit
            self.assertLess(rotation_error_deg(Rh, R), 1e-4)
            self.assertLess(np.abs(th - t).max(), 1e-9)

    def test_pnp_recovers_known_pose_and_motion(self):
        for _ in range(50):
            R = random_rotation()
            t = np.array([RNG.uniform(-0.2, 0.2), RNG.uniform(-0.1, 0.1), RNG.uniform(0.5, 1.5)])
            cam = self.palm @ R.T + t
            if (cam[:, 2] <= 0.05).any():
                continue
            uv = cam[:, :2] / cam[:, 2:] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
            Rh, th = pnp(self.palm, uv, self.K)
            self.assertLess(rotation_error_deg(Rh, R), 1e-4)
            self.assertLess(np.abs(th - t).max(), 1e-6)
            # a second pose 10 mm and 3 degrees away: the recovered motion must be exact
            dR = cv_rot(np.array([0.03, -0.04, 0.02]))
            R2, t2 = R @ dR, t + R @ np.array([0.006, -0.008, 0.0])
            cam2 = self.palm @ R2.T + t2
            uv2 = cam2[:, :2] / cam2[:, 2:] * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
            R2h, t2h = pnp(self.palm, uv2, self.K)
            dRh, dph = relative(Rh, th, R2h, t2h)
            dRt, dpt = relative(R, t, R2, t2)
            self.assertLess(np.abs(dph - dpt).max(), 1e-6)
            self.assertLess(rotation_error_deg(dRh, dRt), 1e-4)

    def test_pnp_refuses_too_few_points(self):
        self.assertIsNone(pnp(self.palm[:5], np.zeros((5, 2)), self.K))

    def test_pnp_refuses_collapsed_points(self):
        """A tracker can put every point on one pixel; that has no pose, not an exception."""
        self.assertIsNone(pnp(self.palm, np.full((len(self.palm), 2), 300.0), self.K))

    def test_identity_motion_is_exactly_zero(self):
        R, t = random_rotation(), RNG.normal(size=3)
        dR, dp = relative(R, t, R, t)
        self.assertLess(np.abs(dp).max(), 1e-12)
        self.assertLess(rotation_error_deg(dR, np.eye(3)), 1e-5)

    def test_rotation_error_of_a_known_rotation(self):
        Rz = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(rotation_error_deg(np.eye(3), Rz), 90.0, places=6)

    def test_repeat_is_constant_motion(self):
        dR, dp = cv_rot(np.array([0.0, 0.0, 0.1])), np.array([0.01, 0.0, 0.0])
        R1, p1 = repeat(dR, dp, 1)
        self.assertTrue(np.allclose(R1, dR) and np.allclose(p1, dp))
        R3, p3 = repeat(np.eye(3), dp, 3)
        self.assertTrue(np.allclose(p3, 3 * dp) and np.allclose(R3, np.eye(3)))

    def test_matches_the_label_convention_on_a_real_clip(self):
        """relative() on recorded absolute poses must reproduce the stored action labels,
        and composing k one-step labels must equal the k-step relative motion."""
        rec = json.loads((DATA / 'validation_plan_v1.json').read_text())['records'][0]
        shards = json.loads((DATA / 'shard_index.json').read_text())
        with np.load(DATA / 'gen' / shards[rec['video_id']] / 'actions/labels' / f"{rec['video_id']}.npz") as a:
            pos, R = a['position'].astype(np.float64), sixd_to_rot(a['rot_6d'].astype(np.float64))
            dp_lab, dR_lab = a['delta_pos_body'].astype(np.float64), sixd_to_rot(a['delta_rot6d_body'].astype(np.float64))
            valid = a['valid']
        # The last label can be valid while the pose after it is not stored: deltas come from
        # the full trajectory, the file is trimmed to the video's length.
        idx = np.flatnonzero(valid[:len(pos) - 1])
        self.assertGreater(len(idx), 100)
        for t in idx:
            dR, dp = relative(R[t], pos[t], R[t + 1], pos[t + 1])
            self.assertLess(np.abs(dp - dp_lab[t]).max(), 1e-5)
            self.assertLess(rotation_error_deg(dR, dR_lab[t]), 1e-2)
        k = 8
        for t in idx[:200]:
            if t + k >= len(pos) or not valid[t:t + k].all():
                continue
            Rc, pc = np.eye(3), np.zeros(3)
            for j in range(t, t + k):
                pc, Rc = pc + Rc @ dp_lab[j], Rc @ dR_lab[j]
            dR, dp = relative(R[t], pos[t], R[t + k], pos[t + k])
            self.assertLess(np.abs(pc - dp).max(), 1e-4)
            self.assertLess(rotation_error_deg(Rc, dR), 0.05)


def cv_rot(rvec):
    import cv2
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]


if __name__ == '__main__':
    unittest.main(verbosity=2)
