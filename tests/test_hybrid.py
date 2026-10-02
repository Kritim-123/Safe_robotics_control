"""Observation, selector and hybrid handoff regressions."""
import hashlib
import json
from dataclasses import replace
import tempfile
from contextlib import nullcontext
from unittest.mock import patch
import unittest

import mujoco
import numpy as np

from hybrid.arena import Arena, Obstacle, Scenario, build_model, scenario
from hybrid.perception import choose_masked, observe_encounter, threatening_obstacle
from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from hybrid.system import HybridSystem
from hybrid.sensing import ObstacleSensor
from skills.avoidance import path_clearance


class GeometryTests(unittest.TestCase):
    def test_friction_setting_reaches_actual_foot_contacts(self):
        spec = scenario('low_friction')
        m = build_model(spec)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
        d.qpos[2] -= .03
        mujoco.mj_forward(m, d)
        floor = m.geom('arena_floor').id
        contacts = [c for c in d.contact if floor in (c.geom1, c.geom2)]
        self.assertTrue(contacts)
        for contact in contacts:
            self.assertAlmostEqual(contact.friction[0], .25)

    def test_scenario_roundtrip_and_invalid_geometry(self):
        spec = scenario('crossing', 4)
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            path = Path(folder)/'scenario.json'
            path.write_text(json.dumps(spec.to_dict()))
            restored = Scenario.load(path)
            np.testing.assert_allclose(restored.obstacles[0].at(10), spec.obstacles[0].at(10))
        for invalid in [replace(spec, goal=(10, 0)), replace(spec, start=(0,)),
                        replace(spec, friction=float('nan')),
                        replace(spec, obstacles=(Obstacle('bad', (1, 0), velocity=(1,)),))]:
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_obstacle_prediction_detects_crossing_before_contact(self):
        obstacles = [(np.array([0.9, -0.9]), .23, np.array([0.0, .1]))]
        self.assertIsNone(threatening_obstacle(np.zeros(2), np.array([3., 0]), obstacles, horizon=0))
        self.assertEqual(threatening_obstacle(np.zeros(2), np.array([3., 0]), obstacles), 0)

    def test_side_obstacle_masks_only_blocked_bypass(self):
        obstacles = [(np.array([1.35, 0]), .23, np.zeros(2)),
                     (np.array([1.15, 1.1]), .37, np.zeros(2))]
        observation = observe_encounter(np.zeros(2), 0, np.array([3., 0]), obstacles, (-1.5, 4.5, -2, 2), 0)
        self.assertFalse(observation.allowed[0])
        self.assertTrue(observation.allowed[1])

    def test_thin_obstacle_between_clear_endpoints_is_blocked(self):
        obstacles = [(np.array([1., 0]), .1, np.zeros(2))]
        self.assertLess(path_clearance(np.zeros(2), [np.array([2., 0])], obstacles, (-2, 4, -2, 2)), 0)

    def test_mask_cannot_select_disallowed_or_nonfinite_action(self):
        self.assertEqual(choose_masked([100, 2, 1, 0], [False, True, True, True]), 1)
        self.assertIsNone(choose_masked([1, 2, 3, 4], [False]*4))
        with self.assertRaises(ValueError):
            choose_masked([np.nan, 0, 0, 0], [True]*4)

    def test_moving_obstacle_updates_without_moving_robot(self):
        spec = scenario('crossing')
        m = build_model(spec)
        d = mujoco.MjData(m)
        arena = Arena(spec, m)
        before = d.qpos.copy()
        d.time = 10
        arena.update(d)
        np.testing.assert_array_equal(d.qpos, before)
        np.testing.assert_allclose(arena.sensed_obstacles(d)[0][0], spec.obstacles[0].at(10))


class SpikingTests(unittest.TestCase):
    def test_model_checksum_and_real_binary_spikes(self):
        manifest = json.loads((DEFAULT_MODEL.parent/'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest(), manifest['sha256'])
        selector = SpikingSelector()
        observation = observe_encounter(np.zeros(2), 0, np.array([3., 0]),
            [(np.array([1.35, 0]), .23, np.zeros(2))], (-1.5, 4.5, -2, 2), 0)
        scores, diagnostics = selector.predict(observation.features, trace=True)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(diagnostics['raster'].shape, (16, 96))
        self.assertEqual(set(np.unique(diagnostics['raster'])), {0, 1})
        self.assertGreater(diagnostics['spikes'], 0)
        # Decision windows are independent, not dependent on earlier episodes.
        np.testing.assert_array_equal(scores, selector.predict(observation.features)[0])

    def test_invalid_observation_is_rejected(self):
        selector = SpikingSelector()
        with self.assertRaises(ValueError):
            selector.predict(np.zeros(11))
        with self.assertRaises(ValueError):
            selector.predict(np.full(12, np.inf))


class SensingTests(unittest.TestCase):
    def test_filter_reduces_noise_on_constant_velocity_motion(self):
        plain = ObstacleSensor(position_noise=.04, velocity_noise=.01, seed=3)
        smooth = ObstacleSensor(position_noise=.04, velocity_noise=.01, filter_tau=.2, seed=3)
        errors = [[], []]
        for step in range(200):
            now = step*.1
            truth = [(np.array([1., now*.1]), .2, np.array([0., .1]))]
            for i, sensor in enumerate((plain, smooth)):
                estimate = sensor.observe(now, truth)[0][0]
                if step > 20:
                    errors[i].append(np.linalg.norm(estimate-truth[0][0]))
        self.assertLess(np.mean(errors[1]), .65*np.mean(errors[0]))

    def test_delayed_snapshot_contains_only_past_observations(self):
        sensor = ObstacleSensor(delay=.3)
        outputs = []
        for step in range(8):
            now = step*.1
            outputs.append(sensor.observe(now, [(np.array([now, 0]), .2, np.array([1., 0]))])[0][0][0])
        np.testing.assert_allclose(outputs, [0, 0, 0, 0, .1, .2, .3, .4], atol=1e-8)
        self.assertLessEqual(len(sensor.history), 5)

    def test_noise_is_reproducible_without_mutating_truth(self):
        truth = [(np.array([1., 2]), .2, np.array([0., .1]))]
        first = ObstacleSensor(position_noise=.04, velocity_noise=.01, seed=4)
        second = ObstacleSensor(position_noise=.04, velocity_noise=.01, seed=4)
        a, b = first.observe(0, truth), second.observe(0, truth)
        np.testing.assert_array_equal(a[0][0], b[0][0])
        np.testing.assert_array_equal(a[0][2], b[0][2])
        self.assertFalse(np.array_equal(a[0][0], truth[0][0]))
        np.testing.assert_array_equal(truth[0][0], [1, 2])
        for kwargs in [dict(delay=-1), dict(position_noise=float('nan'))]:
            with self.assertRaises(ValueError):
                ObstacleSensor(**kwargs)


class HybridIntegrationTests(unittest.TestCase):
    def test_optional_replanning_handles_an_obstacle_entering_active_bypass(self):
        result = HybridSystem(scenario('late_crossing', 600), replanning=True).run()
        self.assertTrue(result.summary['success'], result.summary)
        self.assertFalse(result.summary['collision'])
        self.assertEqual(result.summary['replans'], 1)
        choices = [e for e in result.events if e['target'] == 'SKILL']
        self.assertEqual([e['skill'] for e in choices], ['bypass_left', 'wait', 'bypass_left'])
        self.assertEqual(choices[1]['source'], 'SKILL')
        self.assertEqual(choices[1]['interrupted_skill'], 'bypass_left')
        self.assertEqual(choices[1]['obstacle_index'], 1)
        positions = np.array([[r['x'], r['y']] for r in result.history])
        self.assertLess(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)), .08)

    def test_viewer_closed_before_first_step_returns_serializable_failure(self):
        class ClosedViewer:
            def __init__(self):
                self.cam = mujoco.MjvCamera()
            def is_running(self):
                return False
        with patch('mujoco.viewer.launch_passive', return_value=nullcontext(ClosedViewer())):
            result = HybridSystem(scenario('clear')).run(viewer=True)
        self.assertFalse(result.summary['success'])
        self.assertEqual(result.summary['reason'], 'viewer closed')
        self.assertEqual(result.events[-1]['target'], 'FAILED')
        json.dumps(result.summary, allow_nan=False)

    def test_snn_bypass_returns_to_baseline_and_reaches_goal(self):
        result = HybridSystem(scenario('static', 0), 'snn').run()
        self.assertTrue(result.summary['success'], result.summary)
        self.assertFalse(result.summary['collision'])
        self.assertGreater(result.summary['snn_spikes'], 0)
        self.assertEqual(result.summary['interventions'], 1)
        self.assertEqual(result.summary['returns_to_baseline'], 1)
        handoff = next(e for e in result.events if e['target'] == 'BASELINE')
        self.assertLess(handoff['speed'], .04)
        self.assertLess(handoff['angular_speed'], .2)
        self.assertEqual([event['target'] for event in result.events], ['SKILL', 'RECOVER', 'BASELINE', 'COMPLETE'])
        positions = np.array([[r['x'], r['y']] for r in result.history])
        self.assertLess(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)), .08)
        self.assertGreater(result.summary['min_footprint_clearance'], 0)

    def test_baseline_has_no_intervention_and_reports_collision(self):
        result = HybridSystem(scenario('static', 0), 'baseline').run()
        self.assertFalse(result.summary['success'])
        self.assertTrue(result.summary['collision'])
        self.assertEqual(result.summary['interventions'], 0)
        self.assertEqual(result.events[-1]['target'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
