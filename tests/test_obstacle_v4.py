from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np

from experiments.obstacle_v4.model import JumpHead, NAMES, ROOT, observe
from experiments.obstacle_v4.runner import execute
from experiments.obstacle_v4.train import scene
from experiments.obstacle_v4.run import controls


class AutonomousLowBarTests(unittest.TestCase):
    def folder(self):
        root = ROOT/'output/obstacle_v4/unit-tests'
        root.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(context.cleanup)
        return Path(context.name)/'episode'

    def test_observation_contract_fresh_packets_and_model_integrity(self):
        self.assertFalse(any('request' in name or 'task' in name or 'time' in name for name in NAMES))
        model = JumpHead.load()
        a, _ = observe(scene(), (0., 0.), 0., 0., True, 2.)
        b, _ = observe(scene(), (0., 0.), 0., 0., True, 12.)
        self.assertEqual(a, b)
        first = model.encode(a)
        model.encode(dict(a, height=.3))
        np.testing.assert_array_equal(first, model.encode(a))
        self.assertTrue(((first >= 0) & (first <= 16)).all())
        self.assertEqual(model.predict(a)['proposed_skill'], 'low_bar_jump')
        path = self.folder().with_suffix('.npz'); model.save(path)
        self.assertEqual(JumpHead.load(path).checksum, model.checksum)
        with np.load(path) as data:
            changed = {key: data[key].copy() for key in data.files}
        changed['weights'][0, 0] += .01
        np.savez_compressed(path, **changed)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            JumpHead.load(path)
        with self.assertRaises(ValueError):
            model.encode(dict(a, height=float('nan')))

    def test_full_crossing_then_goal_with_one_physics_reset_and_no_jump_request(self):
        folder = self.folder(); held = []
        original = mujoco.mj_step

        def step(m, d):
            held[:] = [d]
            original(m, d)

        with patch('experiments.obstacle_v4.runner.mujoco.mj_resetDataKeyframe', wraps=mujoco.mj_resetDataKeyframe) as reset, \
                patch('experiments.obstacle_v4.runner.mujoco.mj_step', side_effect=step):
            result = execute(scene(), folder)
        # WalkingExperiment also initializes a separate IK data object. The
        # actual simulated plant must be reset exactly once, before stepping.
        self.assertEqual(sum(call.args[1] is held[0] for call in reset.call_args_list), 1)
        self.assertTrue(result['success'] and result['jump_completed'] and result['terminal_stable'])
        self.assertFalse(result['collision'] or result['fall'])
        self.assertLess(result['final_goal_distance'], .12)
        self.assertGreater(result['final_collision_geometry_margin'], .003)
        self.assertGreater(min(result['min_foot_bar_clearance']), .002)
        self.assertEqual(result['explicit_jump_requests'], 0)
        self.assertLessEqual(result['max_torque_limit_ratio'], 1.)
        np.testing.assert_array_equal(held[0].ctrl, np.zeros(12))
        packets = [json.loads(line) for line in (folder/'decisions.jsonl').read_text().splitlines()]
        self.assertTrue(all(p['mission'] == 'progress' for p in packets))
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        handoffs = [e['next'] for e in events if e['event'] == 'handoff']
        self.assertEqual(handoffs, ['low_bar_jump', 'run', 'wait'])
        with np.load(folder/'trace.npz') as data:
            positions = data['values'][:, 1:4]
        self.assertLess(np.linalg.norm(np.diff(positions, axis=0), axis=1).max(), .02)

    def test_neural_proposal_required_and_guard_can_veto_it(self):
        model = JumpHead.load(); original = model.predict
        with patch.object(model, 'predict', side_effect=lambda f: dict(original(f), proposed_skill='wait')):
            result = execute(scene(), self.folder(), head=model)
        self.assertFalse(result['jump_initiated'])
        self.assertEqual(result['reason'], 'refused')
        for spec in controls()[:3]:
            with self.subTest(scene=spec.name), patch.object(model, 'predict', side_effect=lambda f: dict(original(f), proposed_skill='low_bar_jump')):
                result = execute(spec, self.folder(), head=model)
            self.assertFalse(result['jump_initiated'] or result['collision'])
            self.assertGreater(result['vetoed_jump_proposals'], 0)

    def test_preparation_interruption_recovers_without_crossing_credit(self):
        result = execute(scene(), self.folder(), interrupt_after=.3)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertTrue(result['terminal_stable'] and result['torques_released'])
        self.assertFalse(result['success'] or result['jump_completed'] or result['collision'])
        self.assertEqual(result['longest_airtime'], 0.)

    def test_viewer_close_before_first_step_releases_torque(self):
        view = MagicMock(); view.is_running.return_value = False
        context = MagicMock(); context.__enter__.return_value = view
        with patch('experiments.obstacle_v4.runner.mujoco.viewer.launch_passive', return_value=context):
            result = execute(scene(), self.folder(), viewer=True)
        self.assertEqual(result['reason'], 'viewer closed')
        self.assertTrue(result['torques_released'])
        self.assertFalse(result['jump_initiated'] or result['success'])


if __name__ == '__main__':
    unittest.main()
