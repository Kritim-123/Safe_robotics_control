"""Physical evidence for bounded fast trot and in-place jumping."""
import unittest
from unittest.mock import patch

import numpy as np

from controllers.navigation import NavigationExperiment
from controllers.walking import WalkingExperiment
from hybrid.arena import Scenario, Obstacle, build_model
from skills.dynamic import DynamicSkill, execute_dynamic_skill


class DynamicSkillTests(unittest.TestCase):
    def assert_physical_success(self, history, result):
        self.assertTrue(result['success'], result)
        self.assertFalse(result['collision'])
        self.assertFalse(result['fall'])
        self.assertLessEqual(result['max_torque_limit_ratio'], 1.)
        self.assertTrue(result['stable_landing'])
        self.assertEqual(history[-1]['time'], result['elapsed'])
        self.assertLess(history[-1]['speed'], .04)
        self.assertEqual(history[-1]['foot_contacts'], 4)
        self.assertLess(np.max(np.linalg.norm(np.diff([[r['x'], r['y']] for r in history], axis=0), axis=1)), .08)

    def test_jump_flight_and_landing_development_variations(self):
        for seed, yaw, friction in [(0, 0., .8), (1, .2, .7), (2, -.2, .9)]:
            with self.subTest(seed=seed):
                spec = Scenario('jump_development', bounds=(-3, 6, -3, 3), start_yaw=yaw, friction=friction)
                history, result = execute_dynamic_skill('jump', spec, seed=seed)
                self.assert_physical_success(history, result)
                self.assertGreaterEqual(result['longest_airtime'], .04)
                self.assertGreater(result['max_height']-result['decision_observation']['height'], .03)
                self.assertLess(result['distance'], .2)

    def test_fast_trot_faster_than_original_walk_same_active_duration(self):
        spec = Scenario('speed_comparison', bounds=(-3, 6, -3, 3))
        history, result = execute_dynamic_skill('run', spec, duration=4)
        self.assert_physical_success(history, result)
        original = WalkingExperiment(build_model(spec))
        walk_history, walk_result = original.run(duration=4)
        self.assertEqual(walk_result['reason'], 'duration reached')
        def active_progress(rows):
            start = min(rows, key=lambda r: abs(r['time']-2))
            end = min(rows, key=lambda r: abs(r['time']-6))
            return end['x']-start['x']
        self.assertGreater(active_progress(history), 1.5*active_progress(walk_history))
        self.assertFalse(result['all_feet_airborne'])  # Report fast trot, not aerial running.

    def test_ineligible_attempts_return_failure_and_wait_still_works(self):
        spec = Scenario('front_obstacle', obstacles=(Obstacle('blocker', (.95, 0)),))
        for name in ('run', 'jump'):
            history, result = execute_dynamic_skill(name, spec)
            self.assertEqual(result['reason'], 'failed initiation')
            self.assertFalse(result['success'])
            self.assertFalse(result['eligibility'][name])
            self.assertEqual(history[-1]['time'], result['elapsed'])
        self.assert_physical_success(*execute_dynamic_skill('wait', spec))

    def test_timeout_and_interruption_are_not_success(self):
        for name in ('run', 'jump'):
            _, timed = execute_dynamic_skill(name, timeout=2.2)
            self.assertEqual(timed['reason'], 'timeout')
            self.assertFalse(timed['success'])
            history, stopped = execute_dynamic_skill(name, interrupt_after=.3)
            self.assertEqual(stopped['reason'], 'interrupted')
            self.assertFalse(stopped['success'])
            self.assertTrue(stopped['stable_landing'])
            self.assertFalse(stopped['fall'])
        _, airborne_interruption = execute_dynamic_skill('jump', interrupt_after=.7)
        self.assertEqual(airborne_interruption['reason'], 'interrupted')
        self.assertTrue(airborne_interruption['all_feet_airborne'])
        self.assertTrue(airborne_interruption['stable_landing'])

    def test_selector_observes_settled_actual_obstacles_without_reset(self):
        spec = Scenario('moving', obstacles=(Obstacle('moving', (2., 1.), velocity=(0, .1)),))
        observations = []
        def select(observation, allowed):
            observations.append(observation)
            self.assertTrue(allowed['wait'])
            self.assertAlmostEqual(observation['time'], 2)
            self.assertAlmostEqual(observation['obstacles'][0]['center'][1], 1.2)
            return 'wait'
        history, result = execute_dynamic_skill(scenario=spec, selector=select)
        self.assert_physical_success(history, result)
        self.assertEqual(len(observations), 1)
        self.assertEqual(result['skill'], 'wait')

    def test_callback_error_releases_torques(self):
        robot = NavigationExperiment(build_model(Scenario('callback_error')))
        def fail(observation, allowed):
            raise RuntimeError('callback failure')
        with patch('skills.dynamic.NavigationExperiment', return_value=robot):
            with self.assertRaisesRegex(RuntimeError, 'callback failure'):
                execute_dynamic_skill(selector=fail)
        np.testing.assert_array_equal(robot.data.ctrl, 0)

    def test_selector_abstention_returns_failed_summary(self):
        history, result = execute_dynamic_skill(selector=lambda observation, allowed: None)
        self.assertEqual(result['reason'], 'no eligible skill')
        self.assertFalse(result['success'])
        self.assertEqual(history[-1]['time'], result['elapsed'])

    def test_invalid_time_parameters_rejected(self):
        robot = NavigationExperiment(build_model(Scenario('validation')))
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                DynamicSkill(robot, 'jump', bad)
        for kw in [dict(duration=.5), dict(run_speed=.6), dict(timeout=np.nan), dict(interrupt_after=-1)]:
            with self.assertRaises(ValueError):
                execute_dynamic_skill('jump', **kw)


if __name__ == '__main__':
    unittest.main()
