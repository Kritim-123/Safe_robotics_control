"""Torque-loop behavior under saturation and invalid timing."""
import unittest
import numpy as np

from controllers.walking import JointPID


class PIDTests(unittest.TestCase):
    def test_saturation_does_not_leave_a_stale_integral_after_error_clears(self):
        pid = JointPID(kp=10, ki=2, kd=0, limits=np.array([1.]))
        zero, target = np.array([0.]), np.array([1.])
        for _ in range(200):
            torque = pid.update(target, zero, zero, zero, .01)
            self.assertLessEqual(abs(float(torque[0])), 1)
        np.testing.assert_allclose(pid.update(zero, zero, zero, zero, .01), 0, atol=1e-12)

    def test_velocity_feedback_brakes_motion_at_the_position_target(self):
        pid = JointPID(kp=10, ki=0, kd=.5, limits=np.array([2.]))
        zero = np.array([0.])
        self.assertLess(pid.update(zero, zero, zero, np.array([1.]), .01)[0], 0)
        self.assertGreater(pid.update(zero, zero, zero, np.array([-1.]), .01)[0], 0)

    def test_nonfinite_or_nonpositive_dt_is_rejected(self):
        pid = JointPID(kp=10, ki=2, kd=1, limits=np.array([1.]))
        zero = np.array([0.])
        for dt in (float('nan'), float('inf'), 0, -.01):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                pid.update(zero, zero, zero, zero, dt)


if __name__ == '__main__':
    unittest.main()
