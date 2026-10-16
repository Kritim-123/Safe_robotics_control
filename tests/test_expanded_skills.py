"""Independent lifecycle and real MuJoCo tests for optional classical skills."""
import unittest

import mujoco
import numpy as np

from controllers.navigation import NavigationExperiment, WaypointController
from controllers.walking import JointPID
from hybrid.arena import Arena, Scenario, build_model
from skills.avoidance import SKILL_NAMES, clearance
from skills.temporal import build_temporal_skill, build_temporal_skills


class TemporalSkillTests(unittest.TestCase):
    def make(self, name='yield_hold', **kw):
        return build_temporal_skill(name, np.zeros(2), np.array([3., 0]), [], (-2, 5, -2, 2), 0, **kw)

    def test_original_selector_schema_unchanged(self):
        self.assertEqual(SKILL_NAMES, ('bypass_left', 'bypass_right', 'wait', 'back_off'))

    def test_new_threat_resets_entire_dwell(self):
        skill = self.make()
        for now, clear in [(0, True), (1.4, True), (1.5, False), (1.6, True), (3, True)]:
            skill.update([0, 0], now, clear)
            self.assertFalse(skill.complete([0, 0], now))
        skill.update([0, 0], 3.1, True)
        self.assertTrue(skill.complete([0, 0], 3.1))

    def test_retreat_arrival_begins_dwell_and_interrupt_is_terminal(self):
        skill = self.make('retreat_hold')
        skill.update([0, 0], 2, True)
        self.assertFalse(skill.holding)
        skill.update([-.55, 0], 3, True)
        self.assertTrue(skill.holding)
        self.assertIsNone(skill.target([-.55, 0]))
        self.assertFalse(skill.complete([-.55, 0], 3))
        skill.interrupt()
        skill.update([-.55, 0], 5, True)
        self.assertEqual(skill.status, 'interrupted')
        self.assertFalse(skill.complete([-.55, 0], 5))

    def test_timeout_has_priority_over_completion(self):
        skill = self.make(timeout=2)
        skill.update([0, 0], .5, True)
        skill.update([0, 0], 2, True)
        self.assertEqual(skill.status, 'timeout')
        self.assertFalse(skill.complete([0, 0], 2))

    def test_init_masks_wall_rear_obstacle_and_approaching_hazard(self):
        for obstacles, bounds, expected in [
            ([], (-.7, 5, -2, 2), (True, False)),
            ([(np.array([-1., 0]), .2, np.zeros(2))], (-2, 5, -2, 2), (True, False)),
            ([(np.array([1., 0]), .2, np.array([-.2, 0]))], (-2, 5, -2, 2), (False, True)),
        ]:
            result = build_temporal_skills([0, 0], [3, 0], None, 0, obstacles, bounds)
            self.assertEqual(result[2:], expected)
        with self.assertRaisesRegex(ValueError, 'initiation'):
            build_temporal_skill('retreat_hold', [0, 0], [3, 0], [], (-.7, 5, -2, 2), 0)

    def test_invalid_inputs_and_time_reversal(self):
        for kw in [dict(hold_duration=0), dict(timeout=1), dict(retreat_distance=np.nan)]:
            with self.assertRaises(ValueError):
                self.make(**kw)
        for goal in [[0, 0], [np.nan, 0], [1]]:
            with self.assertRaises(ValueError):
                build_temporal_skill('yield_hold', [0, 0], goal, [], (-2, 5, -2, 2), 0)
        skill = self.make()
        skill.update([0, 0], 1, False)
        for args in [([0, 0], .5, True), ([np.nan, 0], 2, True), ([0, 0], 2, 1)]:
            with self.assertRaises(ValueError):
                skill.update(*args)


class TemporalSkillPhysicsTests(unittest.TestCase):
    def exercise(self, name, outcome):
        spec = Scenario('independent_temporal_skill', bounds=(-2, 5, -2, 2))
        robot = NavigationExperiment(build_model(spec))
        m, d = robot.model, robot.data
        arena = Arena(spec, m)
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        mujoco.mj_forward(m, d)
        pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
        tracker = WaypointController()
        dt, control_dt = float(m.opt.timestep), .01
        target, target_velocity = robot.home.copy(), np.zeros(m.nu)
        next_control, next_sense = 0., 0.
        skill, recovery_start = None, None
        command = np.zeros(3)
        positions = []
        initial_position = None
        while d.time < 65:
            arena.update(d)
            position = d.body('base').xpos[:2].copy()
            rotation = d.body('base').xmat.reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            if d.time >= 2 and skill is None:
                initial_position = position.copy()
                skill = build_temporal_skill(name, position, [3, 0], [], spec.bounds,
                                             float(d.time), timeout=2 if outcome == 'timeout' else 60)
            if d.time >= next_sense-dt/2:
                next_sense += .1
                command[:] = 0
                if skill is not None:
                    if outcome == 'interrupted' and d.time-skill.started >= .5:
                        skill.interrupt()
                    skill.update(position, float(d.time), outcome != 'timeout')
                    point = skill.target(position)
                    if point is not None:
                        command = tracker.command(position, yaw, point)
                    if skill.status != 'running' and recovery_start is None:
                        recovery_start = float(d.time)
                positions.append(position)
            if d.time >= next_control-dt/2:
                previous = target.copy()
                target = robot.velocity_targets(command, control_dt)
                target_velocity = (target-previous)/control_dt
                next_control += control_dt
            d.ctrl[:] = pid.update(target, target_velocity, d.qpos[robot.qa], d.qvel[robot.va], dt)
            self.assertTrue(np.all(np.abs(d.ctrl) <= robot.limits+1e-9))
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            self.assertTrue(np.all(np.isfinite(d.qpos)))
            self.assertFalse(arena.collided(d))
            self.assertGreater(d.qpos[2], .15)
            self.assertGreater(d.body('base').xmat.reshape(3, 3)[2, 2], np.cos(np.deg2rad(35)))
            self.assertGreater(clearance(d.qpos[:2], [], spec.bounds), .035)
            if recovery_start is not None and d.time-recovery_start >= .8:
                stable = (robot.activity == 0 and np.linalg.norm(d.qvel[:2]) < .04
                          and np.linalg.norm(d.qvel[3:6]) < .2
                          and d.body('base').xmat.reshape(3, 3)[2, 2] > np.cos(np.deg2rad(12)))
                if stable:
                    break
                self.assertLess(d.time-recovery_start, 5, 'recovery timeout')
        self.assertIsNotNone(recovery_start)
        self.assertEqual(skill.status, outcome)
        self.assertLess(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)), .08)
        if name == 'retreat_hold' and outcome == 'complete':
            self.assertLess(d.qpos[0]-initial_position[0], -.35)
        return robot

    def test_both_skills_success_timeout_interruption_in_persistent_physics(self):
        for name in ('yield_hold', 'retreat_hold'):
            for outcome in ('complete', 'timeout', 'interrupted'):
                with self.subTest(name=name, outcome=outcome):
                    self.exercise(name, outcome)

    def test_failed_initiation_leaves_mujoco_state_unchanged(self):
        robot = NavigationExperiment(build_model(Scenario('failed_init')))
        m, d = robot.model, robot.data
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        mujoco.mj_forward(m, d)
        before = (d.qpos.copy(), d.qvel.copy(), d.time)
        for name in ('yield_hold', 'retreat_hold'):
            with self.assertRaises(ValueError):
                build_temporal_skill(name, d.qpos[:2], [3, 0], [], (-.45, 5, -2, 2), d.time)
            np.testing.assert_array_equal(d.qpos, before[0])
            np.testing.assert_array_equal(d.qvel, before[1])
            self.assertEqual(d.time, before[2])


if __name__ == '__main__':
    unittest.main()
