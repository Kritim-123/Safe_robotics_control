import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np

from experiments.obstacle_v4.model import JumpHead, ROOT
from experiments.obstacle_v5.model import WalkUpHead
from experiments.obstacle_v5.run import nominal, controls
from experiments.obstacle_v5.runner import execute


class WalkUpLowBarTests(unittest.TestCase):
    def folder(self):
        root = ROOT/'output/obstacle_v5/unit-tests'
        root.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(context.cleanup)
        return Path(context.name)/'episode'

    def test_walk_brake_observed_jump_and_goal_are_one_continuous_episode(self):
        folder = self.folder(); held = []
        original = mujoco.mj_step

        def step(m, d):
            held[:] = [d]
            original(m, d)

        with patch('experiments.obstacle_v5.runner.mujoco.mj_resetDataKeyframe', wraps=mujoco.mj_resetDataKeyframe) as reset, \
                patch('experiments.obstacle_v5.runner.mujoco.mj_step', side_effect=step):
            result = execute(nominal(), folder)
        self.assertTrue(result['success'] and result['jump_completed'] and result['terminal_stable'])
        self.assertFalse(result['collision'] or result['fall'])
        self.assertLess(result['final_goal_distance'], .12)
        self.assertEqual(result['explicit_jump_requests'], 0)
        self.assertGreaterEqual(result['jump_entry']['walked_distance'], .4)
        self.assertGreaterEqual(result['jump_entry']['stable_hold'], .4)
        self.assertFalse(result['teacher_forced_collection'])
        self.assertEqual(result['jump_entry']['foot_contacts'], 4)
        self.assertLess(result['jump_entry']['linear_speed'], .04)
        self.assertLess(result['jump_entry']['angular_speed'], .2)
        self.assertGreater(result['final_collision_geometry_margin'], .003)
        self.assertEqual(sum(c.args[1] is held[0] for c in reset.call_args_list), 1)
        self.assertTrue(result['torques_released'])
        np.testing.assert_array_equal(held[0].ctrl, 0.)
        self.assertLessEqual(result['max_torque_limit_ratio'], 1.)
        events = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        self.assertEqual([e['next'] for e in events if e['event'] == 'handoff'],
                         ['approach', 'wait', 'low_bar_jump', 'run', 'wait'])
        self.assertEqual(sum(e['event'] == 'approach_braking' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'precision_approach' for e in events), 1)
        with np.load(folder/'trace.npz') as data:
            trace = data['values']
        self.assertLess(np.linalg.norm(np.diff(trace[:, 1:4], axis=0), axis=1).max(), .02)
        packets = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
        self.assertTrue(all(p['mission'] == 'progress' for p in packets))

    def test_new_associations_preserve_the_old_encoder(self):
        old, new = JumpHead.load(), WalkUpHead.load()
        np.testing.assert_array_equal(old.weights, new.weights)
        np.testing.assert_array_equal(old.biases, new.biases)
        self.assertNotEqual(old.checksum, new.checksum)
        self.assertEqual(new.metadata['training_seed'], 17501)
        self.assertGreater(new.metadata['measured_association_rows'], 0)

    def test_learned_wait_prevents_takeoff_after_walking(self):
        model = JumpHead.load(); original = model.predict
        with patch.object(model, 'predict', side_effect=lambda f: dict(original(f), proposed_skill='wait')):
            result = execute(nominal(), self.folder(), head=model)
        self.assertFalse(result['jump_initiated'] or result['collision'])
        self.assertEqual(result['reason'], 'refused')
        self.assertGreater(result['final_position'][0], .4)
        self.assertTrue(result['terminal_stable'])

    def test_forced_jump_proposal_cannot_bypass_guard(self):
        model = JumpHead.load(); original = model.predict
        for spec in controls()[:3]:
            with self.subTest(scene=spec.name), patch.object(model, 'predict', side_effect=lambda f: dict(original(f), proposed_skill='low_bar_jump')):
                result = execute(spec, self.folder(), head=model)
            self.assertFalse(result['jump_initiated'] or result['collision'] or result['fall'])
            self.assertEqual(result['reason'], 'refused')
            self.assertGreater(result['vetoed_jump_proposals'], 0)

    def test_viewer_closed_during_approach_releases_torque(self):
        held = []; original = mujoco.mj_step

        def step(m, d):
            held[:] = [d]
            original(m, d)

        view = MagicMock()
        view.is_running.side_effect = lambda: not held or held[0].time < 4.
        context = MagicMock(); context.__enter__.return_value = view
        with patch('experiments.obstacle_v5.runner.mujoco.viewer.launch_passive', return_value=context), \
                patch('experiments.obstacle_v5.runner.mujoco.mj_step', side_effect=step), \
                patch('experiments.obstacle_v5.runner.time.sleep'):
            result = execute(nominal(), self.folder(), viewer=True)
        self.assertEqual(result['reason'], 'viewer closed')
        self.assertFalse(result['jump_initiated'] or result['success'])
        self.assertTrue(result['torques_released'])
        np.testing.assert_array_equal(held[0].ctrl, 0.)


if __name__ == '__main__':
    unittest.main()
