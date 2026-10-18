import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np

from experiments.obstacle_v4.model import JumpHead, ROOT, V3
from experiments.obstacle_v5.run import cases, controls
from experiments.obstacle_v5_refinement.model import RefinedHead, NAMES, PROFILE
from experiments.obstacle_v5_refinement.runner import execute


class RefinedLowBarTests(unittest.TestCase):
    def folder(self):
        root = ROOT/'output/obstacle_v5_refinement/unit-tests'
        root.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(context.cleanup)
        return Path(context.name)/'episode'

    def encoder(self):
        w, b = RefinedHead.encoder_arrays()
        return RefinedHead(w, b, np.zeros((1, len(b))), np.zeros(1), dict(features=list(NAMES),
            v3_sha256=hashlib.sha256(V3.read_bytes()).hexdigest()))

    def features(self):
        return dict(present=1., front=.253, side=.001, height=.036, depth=.018, span=1.2,
                    target_speed=0., alignment=0., landing_clearance=2., stable=1.,
                    goal_distance=1.51, robot_speed=.001, friction=.82,
                    body_pitch=.001, body_height=.267, angular_speed=.004)

    def test_preserved_prefix_finer_resolution_and_packet_reset(self):
        head, old = self.encoder(), JumpHead.load()
        a = self.features(); first = head.encode(a)
        self.assertEqual(len(first), 264)
        np.testing.assert_array_equal(head.weights[:168, :13], old.weights)
        np.testing.assert_array_equal(head.biases[:168], old.biases)
        np.testing.assert_array_equal(first[:168], old.encode(a))
        other = dict(a, friction=.825, height=.037)
        changed = head.encode(other)
        self.assertGreater(np.abs(changed[168:]-first[168:]).sum(), 0)
        np.testing.assert_array_equal(head.encode(a), first)
        self.assertFalse(any('request' in k or 'time' in k or 'case' in k for k in NAMES))
        with self.assertRaises(ValueError):
            head.encode(dict(a, body_pitch=float('nan')))

    def test_model_roundtrip_and_normalization_tamper(self):
        head = self.encoder(); path = self.folder().with_suffix('.npz')
        head.save(path)
        loaded = RefinedHead.load(path)
        self.assertEqual(loaded.checksum, head.checksum)
        with np.load(path) as data:
            altered = {k:data[k].copy() for k in data.files}
        altered['scales'][3] *= 1.1
        np.savez_compressed(path, **altered)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            RefinedHead.load(path)

    def test_conflicting_count_outcomes_still_abstain(self):
        head = self.encoder(); features = self.features(); counts = head.encode(features)
        mixed = RefinedHead(head.weights, head.biases, np.array([counts, counts]),
                            np.array([1, 0]), head.metadata)
        self.assertEqual(mixed.predict(features)['proposed_skill'], 'wait')
        positive = RefinedHead(head.weights, head.biases, np.array([counts]),
                               np.array([1]), head.metadata)
        self.assertEqual(positive.predict(features)['proposed_skill'], 'low_bar_jump')

    def test_five_reported_failures_complete_autonomously(self):
        for i in (2, 4, 6, 8, 14):
            with self.subTest(case=i):
                result = execute(cases('test', 20)[i], self.folder())
                self.assertTrue(result['success'] and result['jump_completed'] and result['terminal_stable'])
                self.assertFalse(result['collision'] or result['fall'] or result['teacher_forced_collection'])
                self.assertEqual(result['explicit_jump_requests'], 0)
                self.assertGreater(result['jump_entry']['walked_distance'], .4)
                self.assertEqual(result['jump_entry']['foot_contacts'], 4)
                self.assertGreaterEqual(result['jump_entry']['stable_hold'], .4)
                self.assertLess(result['final_goal_distance'], .12)
                self.assertLessEqual(result['max_torque_limit_ratio'], 1.)

    def test_contact_case_has_one_plant_reset_and_continuous_motion(self):
        held = []; original = mujoco.mj_step; folder = self.folder()

        def step(m, d):
            held[:] = [d]; original(m, d)

        with patch('experiments.obstacle_v5_refinement.runner.mujoco.mj_resetDataKeyframe', wraps=mujoco.mj_resetDataKeyframe) as reset, \
                patch('experiments.obstacle_v5_refinement.runner.mujoco.mj_step', side_effect=step):
            result = execute(cases('test', 20)[8], folder)
        self.assertTrue(result['success'])
        self.assertEqual(sum(c.args[1] is held[0] for c in reset.call_args_list), 1)
        np.testing.assert_array_equal(held[0].ctrl, 0.)
        with np.load(folder/'trace.npz') as data:
            positions = data['values'][:, 1:4]
        self.assertLess(np.linalg.norm(np.diff(positions, axis=0), axis=1).max(), .02)

    def test_learned_proposal_is_required_and_guards_can_veto_it(self):
        head = RefinedHead.load(); original = head.predict
        with patch.object(head, 'predict', side_effect=lambda f:dict(original(f), proposed_skill='wait')):
            result = execute(cases('test', 20)[8], self.folder(), head=head)
        self.assertEqual(result['reason'], 'refused')
        self.assertFalse(result['jump_initiated'] or result['collision'])
        for spec in controls()[:3]:
            with self.subTest(scene=spec.name), patch.object(head, 'predict', side_effect=lambda f:dict(original(f), proposed_skill='low_bar_jump')):
                result = execute(spec, self.folder(), head=head)
            self.assertFalse(result['jump_initiated'] or result['collision'] or result['fall'])
            self.assertGreater(result['vetoed_jump_proposals'], 0)

    def test_preparation_interruption_and_viewer_close_release_torque(self):
        result = execute(cases('test', 20)[8], self.folder(), interrupt_after=.3)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertTrue(result['terminal_stable'] and result['torques_released'])
        self.assertFalse(result['jump_completed'] or result['collision'])
        view = MagicMock(); view.is_running.return_value = False
        context = MagicMock(); context.__enter__.return_value = view
        with patch('experiments.obstacle_v5_refinement.runner.mujoco.viewer.launch_passive', return_value=context):
            result = execute(cases('test', 20)[8], self.folder(), viewer=True)
        self.assertEqual(result['reason'], 'viewer closed')
        self.assertTrue(result['torques_released'])


if __name__ == '__main__':
    unittest.main()
