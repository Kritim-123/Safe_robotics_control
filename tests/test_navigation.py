"""End-to-end checks exercise real MuJoCo dynamics, including handoff to stand."""
import unittest

import numpy as np

from skills.navigate import navigate


class NavigationTests(unittest.TestCase):
    def assert_arrived(self, history, result):
        self.assertTrue(result['success'], result)
        self.assertLessEqual(result['final_distance'], 0.12)
        self.assertLess(history[-1]['speed'], 0.04)
        self.assertGreater(min(row['height'] for row in history), 0.15)
        times = [row['time'] for row in history]
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_forward_turn_and_behind(self):
        for goal in [(1, 0), (0, 1), (-1, 0)]:
            with self.subTest(goal=goal):
                self.assert_arrived(*navigate(goal, viewer=False, timeout=60))

    def test_multiple_waypoints_without_reset(self):
        history, result = navigate((0, 1), waypoints=[(-0.6, 0), (-0.6, 1)],
                                   timeout=90, viewer=False)
        self.assert_arrived(history, result)
        self.assertEqual(result['waypoint_index'], 2)
        positions = np.array([[r['x'], r['y']] for r in history])
        self.assertLess(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)), 0.1)

    def test_nonzero_start_and_heading(self):
        self.assert_arrived(*navigate((-0.6, 0.5), start=(-0.6, -0.3),
                                      start_yaw=np.pi / 2, viewer=False, timeout=60))

    def test_already_at_goal(self):
        self.assert_arrived(*navigate((0, 0), viewer=False, timeout=10))

    def test_blocked_route_reports_failure(self):
        _, result = navigate((1, 0.6), viewer=False, timeout=60)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'obstacle contact')

    def test_timeout_is_not_success(self):
        _, result = navigate((1, 0), viewer=False, timeout=0.1)
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'timeout')

    def test_rejects_invalid_inputs(self):
        for args in [dict(goal=(np.nan, 0)), dict(timeout=-1), dict(start=(0,)),
                     dict(start_yaw=np.inf), dict(tolerance=0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                navigate(viewer=False, **args)


if __name__ == '__main__':
    unittest.main()
