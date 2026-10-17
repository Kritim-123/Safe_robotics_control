import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from controllers.walking import WalkingExperiment
from hybrid.arena import Scenario, Obstacle, build_model
from skills.pretrained_velocity import (POLICY_ROOT, PretrainedVelocityPolicy,
                                        execute_pretrained_skill, numeric_forward)


class PretrainedVelocityTests(unittest.TestCase):
    def robot(self):
        return WalkingExperiment(build_model(Scenario('policy_test', bounds=(-4., 6., -4., 4.))))

    def test_name_binding_angular_frame_and_causal_history(self):
        robot = self.robot()
        m, d = robot.model, robot.data
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        d.qpos[3:7] = np.array([.9, .1, .2, .3])/np.linalg.norm([.9, .1, .2, .3])
        d.qvel[3:6] = [1., 2., 3.]
        mujoco.mj_forward(m, d)
        for name in ('robot_lab', 'himloco'):
            policy = PretrainedVelocityPolicy(m, name)
            actual = [m.joint(int(m.actuator_trnid[i, 0])).name for i in policy.actuators]
            self.assertEqual(actual, policy.config['joint_names'])
            frame = policy.frame(d, (.1, 0, 0))
            offset = 0 if name == 'robot_lab' else 3
            np.testing.assert_allclose(frame[offset:offset+3], d.sensor('gyro').data*.25, atol=1e-7)
            initial_qpos, initial_qvel, initial_time = d.qpos.copy(), d.qvel.copy(), d.time
            first = policy.step(d, (.1, 0, 0))
            first_input = policy.last_input.copy()
            first_action = policy.last_action.copy()
            policy.step(d, (0, 0, 0))
            np.testing.assert_allclose(policy.history[0, -12:], first_action, atol=1e-7)
            if name == 'himloco':
                np.testing.assert_array_equal(policy.history[1], first_input[:45])
                self.assertFalse(policy.history[2:].any())
            np.testing.assert_array_equal(d.qpos, initial_qpos)
            np.testing.assert_array_equal(d.qvel, initial_qvel)
            self.assertEqual(d.time, initial_time)
            self.assertFalse(d.ctrl.any())
            policy.reset()
            np.testing.assert_array_equal(policy.step(d, (.1, 0, 0)), first)

    def test_export_matches_torch_on_nonzero_inputs(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(9001)
        for name in ('robot_lab', 'himloco'):
            folder = POLICY_ROOT/name
            model = torch.jit.load(str(folder/'source_policy.pt'), map_location='cpu').eval()
            with np.load(folder/'model.npz', allow_pickle=False) as data:
                weights = {k: data[k] for k in data.files}
            inputs = rng.normal(0, .5, (24, 45 if name == 'robot_lab' else 270)).astype(np.float32)
            with torch.inference_mode():
                reference = model(torch.from_numpy(inputs)).numpy()
            np.testing.assert_allclose(np.stack([numeric_forward(weights, name, x) for x in inputs]),
                                       reference, atol=2e-5, rtol=2e-5)

    def test_checksum_invalid_input_and_model_refusal(self):
        robot = self.robot()
        for name in ('robot_lab', 'himloco'):
            policy = PretrainedVelocityPolicy(robot.model, name)
            for command in [(float('nan'), 0, 0), (1., 0, 0), (0, 0)]:
                with self.assertRaises(ValueError):
                    policy.step(robot.data, command)
        with self.assertRaises(ValueError):
            numeric_forward({}, 'robot_lab', np.zeros(44))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'robot_lab'; folder.mkdir()
            manifest = json.loads((POLICY_ROOT/'robot_lab/manifest.json').read_text())
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'source_policy.pt').write_bytes(b'not the source policy')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                PretrainedVelocityPolicy(robot.model, root=Path(tmp))

    def test_physical_reverse_motion_recovery_and_limits(self):
        for name in ('robot_lab', 'himloco'):
            rows, result = execute_pretrained_skill(name, 'reverse')
            self.assertTrue(result['success'], result)
            self.assertTrue(result['recovered'])
            self.assertFalse(result['collision'] or result['fall'])
            self.assertLessEqual(result['peak_torque'], 24)
            self.assertLessEqual(result['max_torque_limit_ratio'], 1)
            self.assertGreater(result['signed_progress'], .25)
            xy = np.array([[r['x'], r['y']] for r in rows])
            self.assertLess(np.max(np.linalg.norm(np.diff(xy, axis=0), axis=1)), .1)
            self.assertLess(result['final_speed'], .04)

    def test_registry_exposes_validated_options_and_preserves_snn_schemas(self):
        from skills.registry import list_skills, get_skill, run_skill
        from skills.avoidance import SKILL_NAMES
        from hybrid.skill_lif import SKILLS
        available = list_skills()
        learned = [s for s in available if s['identifier'].startswith('pretrained/')]
        self.assertEqual(len(learned), 7)
        self.assertEqual({s['learned_policy'] for s in learned}, {'robot_lab', 'himloco'})
        self.assertTrue(all(s['selector_class'] is None and not s['snn_eligible'] for s in learned))
        self.assertEqual(SKILL_NAMES, ('bypass_left', 'bypass_right', 'wait', 'back_off'))
        self.assertEqual(SKILLS, ('wait', 'run', 'jump'))
        for identifier in ('pretrained/himloco/turn_left', 'pretrained/robot_lab/reverse'):
            self.assertFalse(get_skill(identifier)['available'])
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                run_skill(identifier)
        with self.assertRaises(ValueError):
            get_skill('missing')
        _, result = run_skill('pretrained/himloco/reverse')
        self.assertTrue(result['success'], result)
        # A replacement checkpoint must not inherit the old availability list.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'validation.json').write_bytes((POLICY_ROOT/'validation.json').read_bytes())
            for name in ('robot_lab', 'himloco'):
                (root/name).mkdir()
                manifest = json.loads((POLICY_ROOT/name/'manifest.json').read_text())
                manifest['artifacts']['source_policy.pt'] = 'changed-checkpoint'
                (root/name/'manifest.json').write_text(json.dumps(manifest))
            with patch('skills.registry.POLICY_ROOT', root):
                self.assertFalse(any(s['identifier'].startswith('pretrained/') for s in list_skills()))

    def test_failed_initiation_timeout_interruption_and_cleanup(self):
        blocked = Scenario('blocked', goal=(3, 0), bounds=(-4., 6., -4., 4.),
                           obstacles=(Obstacle('ahead', (.85, 0), .2),))
        _, refused = execute_pretrained_skill('himloco', 'forward', scenario=blocked)
        self.assertFalse(refused['success'])
        self.assertEqual(refused['reason'], 'failed initiation')
        self.assertEqual(refused['policy_ticks'], 0)
        _, timeout = execute_pretrained_skill('himloco', 'forward', timeout=2.3)
        self.assertFalse(timeout['success'])
        self.assertEqual(timeout['reason'], 'timeout')
        _, interrupted = execute_pretrained_skill('himloco', 'reverse', interrupt_after=.5)
        self.assertFalse(interrupted['success'])
        self.assertTrue(interrupted['interrupted'])
        self.assertTrue(interrupted['recovered'], interrupted)
        self.assertEqual(interrupted['reason'], 'interrupted')
        held = []
        def fail(model, data, skill):
            held.append(data)
            raise RuntimeError('callback error')
        with self.assertRaisesRegex(RuntimeError, 'callback error'):
            execute_pretrained_skill(frame_callback=fail)
        self.assertFalse(held[0].ctrl.any())


if __name__ == '__main__':
    unittest.main()
