"""Physical geometry checks; route probes do not demonstrate walking ability."""
import unittest

import mujoco
import numpy as np

from indoor_demo import load_scene, hold_posture


class IndoorTests(unittest.TestCase):
    def setUp(self):
        self.model, self.data = load_scene()
        layout = self.model.body("indoor_layout").id
        self.obstacles = set(np.flatnonzero(self.model.geom_bodyid == layout))

    def obstacle_contacts(self, x, y, yaw=0):
        self.data.qpos[:2] = [x, y]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        return [c for c in self.data.contact
                if c.geom1 in self.obstacles or c.geom2 in self.obstacles]

    def test_spawn_and_lounge_route_clear(self):
        # Probe the complete robot at 10 cm intervals, including the turn.
        for x in np.linspace(0, 4.8, 49):
            self.assertFalse(self.obstacle_contacts(x, 0), f"Blocked at {(x, 0)}")
        for yaw in np.linspace(0, np.pi / 2, 20):
            self.assertFalse(self.obstacle_contacts(4.8, 0, yaw))
        for y in np.linspace(0, 2.6, 27):
            self.assertFalse(self.obstacle_contacts(4.8, y, np.pi / 2))
        for x in np.linspace(4.8, 5.4, 7):
            self.assertFalse(self.obstacle_contacts(x, 2.6, np.pi / 2))

    def test_walls_and_furniture_collide(self):
        for x, y in [(1, 2), (-2.9, 0), (2.05, 2.7), (2.35, -2.8)]:
            self.assertTrue(self.obstacle_contacts(x, y), f"Missing collision at {(x, y)}")

    def test_office_door_only_has_threshold_contacts(self):
        threshold = self.model.geom("office_threshold").id
        for y in np.linspace(0, -2.6, 27):
            for contact in self.obstacle_contacts(4.8, y, -np.pi / 2):
                self.assertIn(threshold, (contact.geom1, contact.geom2))
        self.assertAlmostEqual(self.model.geom_size[threshold, 2] * 2, 0.02)

    def test_posture_remains_stable(self):
        target = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]].copy()
        for _ in range(round(10 / self.model.opt.timestep)):
            hold_posture(self.model, self.data, target)
            mujoco.mj_step(self.model, self.data)
            self.assertTrue(np.isfinite(self.data.qpos).all())
            self.assertGreater(self.data.qpos[2], 0.18)
        self.assertLess(np.linalg.norm(self.data.qpos[:2]), 0.1)
        self.assertFalse(any(w.number for w in self.data.warning))


if __name__ == "__main__":
    unittest.main()
