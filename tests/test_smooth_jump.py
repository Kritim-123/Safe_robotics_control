from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from controllers.navigation import NavigationExperiment
from hybrid.arena import Scenario,Obstacle,build_model
from skills.dynamic import execute_dynamic_skill
from skills.jump_profile import JumpConfig,ForwardJumpTrajectory
from experiments.smooth_jump import run_trial


class SmoothJumpTests(unittest.TestCase):
    def test_trajectory_position_velocity_acceleration_continuity(self):
        robot=NavigationExperiment(build_model(Scenario('continuity')))
        curve=ForwardJumpTrajectory(robot,JumpConfig())
        for boundary in np.cumsum(curve.durations):
            before=curve.sample(float(boundary)-1e-8)
            after=curve.sample(float(boundary)+1e-8)
            np.testing.assert_allclose(before[0],after[0],atol=1e-6)
            np.testing.assert_allclose(before[1],after[1],atol=1e-5)
            np.testing.assert_allclose(before[2],after[2],atol=.001)
        for t in np.linspace(0.,2.,500):
            q,_,_,_=curve.sample(float(t))
            self.assertTrue(np.all(q>=robot.joint_limits[:,0]+.01))
            self.assertTrue(np.all(q<=robot.joint_limits[:,1]-.01))
        self.assertAlmostEqual(curve.gain_blend(0.),0.)
        self.assertAlmostEqual(curve.gain_blend(2.),0.)

    def test_short_forward_hop_and_measured_smoothness(self):
        with tempfile.TemporaryDirectory() as root:
            spec=Scenario('paired_jump',bounds=(-3.,6.,-3.,3.))
            old=run_trial(Path(root)/'old',spec,'legacy')
            new=run_trial(Path(root)/'new',spec)
            self.assertTrue(new['metrics']['qualifies'])
            self.assertGreater(new['metrics']['forward_progress'],.15)
            self.assertGreater(new['metrics']['flight_forward_progress'],.04)
            self.assertLess(new['metrics']['max_command_step'],old['metrics']['max_command_step']*.1)
            self.assertLess(new['metrics']['max_command_velocity_step'],old['metrics']['max_command_velocity_step']*.1)
            self.assertLess(new['metrics']['peak_base_acceleration'],old['metrics']['peak_base_acceleration']*.7)
            self.assertLess(new['metrics']['torque_step_rms'],old['metrics']['torque_step_rms']*.5)
            # Higher-flight forward hopping still has a larger landing force;
            # retain this measured tradeoff instead of claiming every metric improves.
            self.assertGreater(new['metrics']['peak_landing_normal_force'],old['metrics']['peak_landing_normal_force'])

    def test_explicit_torque_override_updates_model_and_pid(self):
        observed=[]
        def frame(m,d,name):
            if d.time>2.01:
                observed.append(float(np.max(np.abs(d.ctrl))))
                np.testing.assert_allclose(m.actuator_ctrlrange, np.tile([-36.,36.],(12,1)))
                np.testing.assert_allclose(m.actuator_forcerange,m.actuator_ctrlrange)
                np.testing.assert_allclose(d.actuator_force,d.ctrl,atol=1e-10)
        _,summary=execute_dynamic_skill('jump',jump_profile='smooth_forward',
            jump_config=replace(JumpConfig(),torque_limit=36.),frame_callback=frame)
        self.assertGreater(max(observed),24.)
        self.assertLessEqual(max(observed),36.)
        self.assertEqual(summary['actuator_limit'],36.)
        self.assertLessEqual(summary['max_torque_limit_ratio'],1.)

    def test_forward_landing_corridor_can_refuse_old_in_place_case(self):
        spec=Scenario('landing_blocked',obstacles=(Obstacle('blocker',(1.15,0),radius=.2),))
        _,old=execute_dynamic_skill('jump',spec)
        _,new=execute_dynamic_skill('jump',spec,jump_profile='smooth_forward')
        self.assertTrue(old['success'])
        self.assertFalse(new['eligibility']['jump'])
        self.assertEqual(new['reason'],'failed initiation')
        self.assertEqual(new['actuator_limit'],24.)

    def test_prepare_and_airborne_interruption_recover(self):
        for when in (.3,.8):
            history,result=execute_dynamic_skill('jump',jump_profile='smooth_forward',interrupt_after=when)
            self.assertEqual(result['reason'],'interrupted')
            self.assertFalse(result['success'])
            self.assertTrue(result['stable_landing'])
            self.assertFalse(result['fall'])
            self.assertLess(history[-1]['speed'],.04)
            if when==.3:self.assertFalse(result['all_feet_airborne'])
            else:self.assertTrue(result['all_feet_airborne'])

    def test_callback_error_releases_relaxed_torques(self):
        robot=NavigationExperiment(build_model(Scenario('cleanup')))
        def fail(m,d,name):
            if d.time>2.6:raise RuntimeError('observer failed')
        with patch('skills.dynamic.NavigationExperiment',return_value=robot):
            with self.assertRaisesRegex(RuntimeError,'observer failed'):
                execute_dynamic_skill('jump',jump_profile='smooth_forward',frame_callback=fail)
        np.testing.assert_array_equal(robot.data.ctrl,0.)

    def test_invalid_configuration_and_legacy_contract(self):
        for config in (replace(JumpConfig(),torque_limit=np.inf),replace(JumpConfig(),torque_limit=0),
                       replace(JumpConfig(),thrust=.01),replace(JumpConfig(),landing_kd=-1)):
            with self.assertRaises(ValueError):
                execute_dynamic_skill('jump',jump_profile='smooth_forward',jump_config=config)
        with self.assertRaises(ValueError):
            execute_dynamic_skill('jump',jump_config=JumpConfig())
        _,wait=execute_dynamic_skill('wait',jump_profile='smooth_forward')
        self.assertEqual(wait['actuator_limit'],24.)
        self.assertTrue(wait['success'])

    def test_learned_proposal_falls_back_when_forward_corridor_is_blocked(self):
        from experiments.spike_pattern_skills.associations import AssociationMemory
        from experiments.spike_pattern_skills.encoder import RandomLIFEncoder
        from experiments.spike_pattern_skills.execute_associations import execute_case
        from experiments.spike_pattern_skills.learn_associations import ENCODER_PATH
        from experiments.spike_pattern_skills.record import capture_scenario
        from experiments.spike_pattern_skills.situations import Situation
        spec=Scenario('learned_landing_blocked',obstacles=(Obstacle('blocker',(1.15,0.),radius=.2),))
        encoder=RandomLIFEncoder.load(ENCODER_PATH)
        observation=capture_scenario(spec,True,spec.name,0)
        counts=encoder.encode(observation['features'])['spike_counts']
        memory=AssociationMemory(encoder.checksum,[dict(context_id='fixture',skill='dynamic/jump',spike_counts=counts)])
        case=Situation('fixture',0,0,True,spec)
        with tempfile.TemporaryDirectory() as root:
            result=execute_case(case,memory,encoder,Path(root)/'episode',jump_profile='smooth_forward')
            self.assertEqual(result['decision']['proposal']['proposed_skill'],'dynamic/jump')
            self.assertTrue(result['decision']['supervision']['fallback'])
            self.assertEqual(result['summary']['skill'],'wait')
            self.assertEqual(result['summary']['actuator_limit'],24.)


if __name__=='__main__':unittest.main()
