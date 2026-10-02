"""Geometry, motion, faster sensing, and faithful neural evidence regressions."""
from dataclasses import replace
import unittest

import mujoco
import numpy as np

from hybrid.arena import Arena, Obstacle, Scenario, build_model, scenario
from hybrid.perception import observe_encounter, threatening_obstacle
from hybrid.system import HybridSystem
from hybrid.inspector import Inspector
from hybrid.snn import SpikingSelector
from skills.avoidance import clearance, path_clearance, segment_distance, ROBOT_RADIUS


class ComplexObstacleTests(unittest.TestCase):
    def test_rotated_box_geometry_is_conservatively_sensed(self):
        o = Obstacle('box', (1.5, .2), shape='box', half_extents=(.2, .4), yaw=.7, height=.6)
        spec = Scenario('test', obstacles=(o,))
        model = build_model(spec)
        data = mujoco.MjData(model)
        arena = Arena(spec, model)
        arena.update(data)
        self.assertEqual(model.geom('obstacle_geom_0').type[0], mujoco.mjtGeom.mjGEOM_BOX)
        np.testing.assert_allclose(model.geom('obstacle_geom_0').size, [.2, .4, .3])
        self.assertAlmostEqual(arena.sensed_obstacles(data)[0][1], np.hypot(.2, .4))
        for corner in ((x*.2, y*.4) for x in (-1, 1) for y in (-1, 1)):
            self.assertLessEqual(np.linalg.norm(corner), o.planning_radius+1e-12)

    def test_motion_velocity_matches_position_and_stop(self):
        for motion in ('linear', 'oscillate'):
            o = Obstacle('moving', (1, 0), velocity=(.1, .2), motion=motion,
                         start_after=5, stop_after=30, period=20, amplitude=(.2, .8))
            np.testing.assert_array_equal(o.at(2), o.center)
            np.testing.assert_array_equal(o.speed_at(2), [0, 0])
            for t in (6, 14, 22):
                numeric = (o.at(t+1e-5)-o.at(t-1e-5))/2e-5
                np.testing.assert_allclose(numeric, o.speed_at(t), atol=1e-8)
            np.testing.assert_array_equal(o.speed_at(35), [0, 0])
            np.testing.assert_allclose(o.at(35), o.at(100))

    def test_invalid_shape_and_motion_rejected(self):
        for o in (Obstacle('a', (1, 0), shape='unknown'), Obstacle('a', (1, 0), motion='unknown'),
                  Obstacle('a', (1, 0), period=0), Obstacle('a', (1, 0), half_extents=(-1, .2))):
            with self.assertRaises(ValueError):
                Scenario('bad', obstacles=(o,)).validate()

    def test_vectorized_geometry_matches_scalar_reference(self):
        rng = np.random.default_rng(40)
        bounds = (-4, 5, -4, 4)
        for _ in range(100):
            start = rng.normal(size=2)
            goal = start + rng.normal(size=2)
            obstacles = [(rng.normal(size=2), float(rng.uniform(.1, .6)), rng.normal(size=2)*.1) for _ in range(6)]
            points = rng.normal(size=(3, 2))
            expected, point = float('inf'), start
            for end in points:
                expected = min(expected, clearance(point, [], bounds), clearance(end, [], bounds),
                               *(segment_distance(c, point, end)-r-ROBOT_RADIUS for c, r, _ in obstacles))
                point = end
            self.assertAlmostEqual(path_clearance(start, points, obstacles, bounds), expected)
            distance = np.linalg.norm(goal-start)
            end = start+(goal-start)/distance*min(distance, 1.15)
            candidates = [(np.linalg.norm(c-start), i) for i, (c, r, v) in enumerate(obstacles)
                          if min(segment_distance(c+v*t, start, end) for t in np.linspace(0, 3, 7))-r-ROBOT_RADIUS-.1 < 0]
            expected_threat = min(candidates)[1] if candidates and distance >= .12 else None
            self.assertEqual(threatening_obstacle(start, goal, obstacles), expected_threat)
            encounter = observe_encounter(start, 0, goal, obstacles, bounds, 0)
            wait = min(clearance(start, [(c+v*t, r, v) for c, r, v in obstacles], bounds) for t in np.linspace(0, 3, 7))
            self.assertAlmostEqual(float(encounter.features[9]), wait, places=6)

    def test_fast_profile_changes_sampling_not_physics_or_model(self):
        system = HybridSystem(replace(scenario('clear'), timeout=.15), reaction='fast')
        result = system.run()
        self.assertEqual(result.summary['sensing_dt'], .02)
        self.assertEqual(result.summary['confirmation_dt'], .08)
        self.assertEqual(result.summary['control_dt'], .01)
        self.assertEqual(result.summary['physics_dt'], .002)
        with self.assertRaises(ValueError):
            HybridSystem(scenario('clear'), reaction='invalid')

    def test_neuron_evidence_sums_to_scores_and_margin(self):
        inspector = Inspector.__new__(Inspector)
        inspector.selector = SpikingSelector()
        f = [1.3, 0, 0, 0, .23, .65, .28, .28, .5, .65, 3, 0]
        scores, _ = inspector.selector.predict(f)
        inspector.events = [dict(features=f, scores=scores.tolist(), allowed=[True]*4)]
        a = inspector.evaluate({})
        e = a['evidence']
        self.assertAlmostEqual(sum(e['per_neuron'])+e['bias'], a['scores'][e['skill']], places=5)
        self.assertAlmostEqual(sum(e['comparison']['per_neuron'])+e['comparison']['bias'], e['comparison']['margin'], places=5)


if __name__ == '__main__':
    unittest.main()
