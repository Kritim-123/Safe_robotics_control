import unittest
import mujoco
import numpy as np
from controllers.walking import WalkingExperiment
from skills.pretrained_jump import gap_model, PretrainedGapJump, evaluate
from third_party.go2_atomic_skills.mujoco_helper import raycast_height_scan


class PretrainedJumpTests(unittest.TestCase):
    def test_scan_hits_gap_and_excludes_robot(self):
        m = gap_model(.3)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
        d.qpos[:3] = [1.2, 0, .33]
        mujoco.mj_forward(m, d)
        scan = raycast_height_scan(m, d, base_body='base', geomgroup=[0, 0, 0, 1, 0, 0])
        self.assertEqual(scan.shape, (187,))
        self.assertTrue(np.any(np.isclose(scan, 1.33)))
        self.assertTrue(np.any(np.isclose(scan, .33)))
        self.assertTrue(np.all(np.isclose(scan, .33) | np.isclose(scan, 1.33)))

    def test_handoff_and_reset_are_reproducible(self):
        robot = WalkingExperiment(gap_model(.3))
        m, d = robot.model, robot.data
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        mujoco.mj_forward(m, d)
        policy = PretrainedGapJump(m)
        first, _ = policy.targets(m, d, (1., 0, 0), False)
        second, scan = policy.targets(m, d, (1., 0, 0), True)
        self.assertEqual(policy.jump.history.flatten().shape, (1175,))
        self.assertTrue(np.all(np.isfinite(second)))
        np.testing.assert_array_equal(policy.last, policy.jump.last_action)
        self.assertEqual(policy.walk.step, 2)
        self.assertEqual(policy.jump.step, 2)
        policy.reset()
        repeated, _ = policy.targets(m, d, (1., 0, 0), False)
        np.testing.assert_array_equal(first, repeated)

    def test_actual_rollout_respects_limits_and_reports_failure(self):
        result = evaluate(width=.3, speed=1.)
        self.assertLessEqual(result['peak_torque'], 24.)
        self.assertFalse(result['success'])
        self.assertIn('guard', result['reason'])
        self.assertTrue(result['jump_activated'])

    def test_invalid_conditions(self):
        for width in [-.1, .5, float('nan')]:
            with self.assertRaises(ValueError):
                gap_model(width)


if __name__ == '__main__':
    unittest.main()
