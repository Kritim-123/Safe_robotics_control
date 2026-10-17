from dataclasses import replace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from skills.obstacle_jump import default_scene, execute_low_bar_jump, execute_obstacle_jump, target_contract
from skills.jump_profile import JumpConfig


class ObstacleJumpTests(unittest.TestCase):
    def test_actual_full_low_bar_crossing_and_actuator_forces(self):
        observations=[]
        def frame(m,d,row):
            np.testing.assert_array_equal(d.xfrc_applied,0.)
            np.testing.assert_array_equal(d.qfrc_applied,0.)
            if d.time>2.01:
                np.testing.assert_allclose(m.actuator_ctrlrange,np.tile([-72.,72.],(12,1)))
                np.testing.assert_allclose(m.actuator_forcerange,m.actuator_ctrlrange)
                np.testing.assert_allclose(d.actuator_force,d.ctrl,atol=1e-10)
            obstacle=m.geom('obstacle_geom_0').id
            self.assertFalse(any(obstacle in (c.geom1,c.geom2) for c in d.contact))
            observations.append(d.qpos[:3].copy())
        rows,result=execute_low_bar_jump(frame_callback=frame)
        self.assertTrue(result['success'],result)
        self.assertTrue(result['stable_landing'])
        self.assertFalse(result['collision'] or result['fall'])
        self.assertGreater(result['longest_airtime'],.4)
        self.assertTrue(all(result['foot_crossed']))
        self.assertTrue(all(result['final_feet_beyond']))
        self.assertGreater(result['final_collision_geometry_margin'],.003)
        # Independently verify foot sphere bottoms over the whole bar footprint.
        feet=np.asarray([r['feet'] for r in rows])
        for foot in range(4):
            overlap=(feet[:,foot,0]+.0175>=.23)&(feet[:,foot,0]-.0175<=.25)
            self.assertTrue(overlap.any())
            self.assertGreater(float(feet[overlap,foot,2].min())-.0175,.042)
        self.assertTrue(np.all(feet[-1,:,0]-.0175>.253))
        self.assertLess(np.linalg.norm(rows[-1]['qvel'][:3]),.04)
        self.assertLess(np.linalg.norm(rows[-1]['qvel'][3:6]),.2)
        self.assertLess(np.max(np.linalg.norm(np.diff(observations,axis=0),axis=1)),.03)

    def test_contract_refuses_unassessed_and_moving_targets(self):
        spec=default_scene();bar=spec.obstacles[0]
        for changed in (replace(bar,height=.2),replace(bar,velocity=(-.1,0.)),
                        replace(bar,center=(1.,0.)),replace(bar,shape='cylinder')):
            with self.assertRaises(ValueError):
                target_contract(replace(spec,obstacles=(changed,)),'bar')
        with self.assertRaises(ValueError):
            execute_low_bar_jump(scenario=replace(spec,friction=.4))
        from experiments.obstacle_jump import bar_scene
        with self.assertRaises(ValueError):
            execute_low_bar_jump(scenario=bar_scene(yaw=.4))

    def test_unrelated_obstacle_keeps_corridor_guard(self):
        from hybrid.arena import Obstacle
        spec=default_scene()
        spec=replace(spec,obstacles=(*spec.obstacles,Obstacle('blocker',(.9,0.),radius=.2)))
        _,result=execute_low_bar_jump(scenario=spec)
        self.assertEqual(result['reason'],'failed initiation')
        self.assertFalse(result['initiated'])
        self.assertEqual(result['actuator_limit'],24.)

    def test_contact_is_failure_and_airtime_does_not_imply_crossing(self):
        # The old short-hop configuration cannot satisfy the full crossing.
        _,result=execute_obstacle_jump(default_scene(),'bar',JumpConfig())
        self.assertFalse(result['success'])
        self.assertFalse(all(result['final_feet_beyond']))
        # A known longer-jump stress scene catches a thigh despite all feet crossing.
        import json
        from skills.obstacle_jump import PROFILE
        from experiments.obstacle_jump import bar_scene
        config=JumpConfig(**json.loads(PROFILE.read_text()))
        _,result=execute_obstacle_jump(bar_scene(distance=.30),'bar',config)
        self.assertTrue(result['collision'])
        self.assertEqual(result['reason'],'obstacle contact')
        self.assertFalse(result['success'])
        self.assertTrue(result['target_contacts'])

    def test_interruptions_and_callback_failure_release_torques(self):
        for when in (.3,.75):
            _,result=execute_low_bar_jump(interrupt_after=when)
            self.assertEqual(result['reason'],'interrupted')
            self.assertTrue(result['stable_landing'])
            self.assertFalse(result['success'] or result['collision'] or result['fall'])
            if when==.3:self.assertEqual(result['longest_airtime'],0.)
        held=[]
        def fail(m,d,row):
            if d.time>2.6:
                held.append(d)
                raise RuntimeError('observer failed')
        with self.assertRaisesRegex(RuntimeError,'observer failed'):
            execute_low_bar_jump(frame_callback=fail)
        np.testing.assert_array_equal(held[0].ctrl,0.)

    def test_registry_keeps_jump_separate_from_snn_classes(self):
        from skills.registry import get_skill
        from skills.dynamic import DYNAMIC_SKILL_NAMES
        info=get_skill('standalone/low_bar_jump')
        self.assertTrue(info['available'] and info['experimental'])
        self.assertFalse(info['snn_eligible'])
        self.assertIsNone(info['selector_class'])
        self.assertEqual(DYNAMIC_SKILL_NAMES,('wait','run','jump'))

    def test_viewer_closed_before_first_step_reports_finite_state(self):
        view=MagicMock();view.is_running.return_value=False
        context=MagicMock();context.__enter__.return_value=view
        with patch('skills.obstacle_jump.mujoco.viewer.launch_passive',return_value=context):
            rows,result=execute_low_bar_jump(viewer=True)
        self.assertEqual(rows,[])
        self.assertEqual(result['reason'],'viewer closed')
        self.assertFalse(result['success'] or result['initiated'])
        self.assertTrue(np.isfinite(result['final_collision_geometry_margin']))


if __name__=='__main__':unittest.main()
