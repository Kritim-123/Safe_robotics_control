import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np

from experiments.obstacle_v5_multi.learn import MODEL
from experiments.obstacle_v5_multi.run import course
from experiments.obstacle_v5_multi.runner import execute
from experiments.obstacle_v5_refinement.model import RefinedHead, ROOT


class MultipleObstacleTests(unittest.TestCase):
    def test_encoder_and_existing_associations_preserved(self):
        old = RefinedHead.load(); new = RefinedHead.load(MODEL)
        for name in ('weights', 'biases', 'scales', 'centers'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
        np.testing.assert_array_equal(old.bank, new.bank[:len(old.bank)])
        np.testing.assert_array_equal(old.labels, new.labels[:len(old.labels)])
        self.assertEqual(new.metadata['parent_checksum'], old.checksum)

    def folder(self):
        root = ROOT/'output/obstacle_v5_multi/unit-tests'
        root.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(context.cleanup)
        return Path(context.name)/'episode'

    def test_three_jumps_one_plant_and_fresh_confirmations(self):
        folder = self.folder(); held = []; step = mujoco.mj_step

        def remember(m, d):
            held[:] = [d]; step(m, d)

        with patch('experiments.obstacle_v5_multi.runner.mujoco.mj_resetDataKeyframe', wraps=mujoco.mj_resetDataKeyframe) as reset, \
                patch('experiments.obstacle_v5_multi.runner.mujoco.mj_step', side_effect=remember):
            result = execute(course(), folder, head=RefinedHead.load(MODEL), timeout=240.)
        self.assertTrue(result['success'] and result['terminal_stable'])
        self.assertEqual(result['completed_jumps'], 3)
        self.assertFalse(result['collision'] or result['fall'] or result['teacher_forced_collection'])
        self.assertEqual(result['explicit_jump_requests'], 0)
        self.assertLessEqual(result['max_torque_limit_ratio'], 1.)
        self.assertEqual(sum(c.args[1] is held[0] for c in reset.call_args_list), 1)
        np.testing.assert_array_equal(held[0].ctrl, 0.)
        with np.load(folder/'trace.npz') as data:
            self.assertLess(np.linalg.norm(np.diff(data['values'][:, 1:4], axis=0), axis=1).max(), .02)
        packets = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
        self.assertEqual([e['target'] for e in result['jump_entries']], ['bar_1', 'bar_2', 'bar_3'])
        for entry in result['jump_entries']:
            rows = [r for r in packets if entry['time']-.101 <= r['time'] <= entry['time']+1e-8]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['eligible'] and r['jump_proposal']['proposed_skill']=='low_bar_jump' for r in rows))
            self.assertGreaterEqual(entry['stable_hold'], .4)
        for outcome in result['jump_outcomes']:
            self.assertGreater(outcome['rear_margin'], .003)
            self.assertGreater(min(outcome['min_foot_bar_clearance']), .002)

    def test_later_tall_bar_is_refused_after_first_verified_jump(self):
        spec = course(count=2, spacing=2.5)
        spec = replace(spec, obstacles=(spec.obstacles[0], replace(spec.obstacles[1], height=.08)))
        result = execute(spec, self.folder(), head=RefinedHead.load(MODEL), timeout=240.)
        self.assertFalse(result['success'] or result['collision'] or result['fall'])
        self.assertEqual(result['completed_jumps'], 1)
        self.assertEqual(result['reason'], 'refused')
        self.assertTrue(result['torques_released'])

    def test_required_learned_proposal_and_blocked_landing_veto(self):
        head = RefinedHead.load(MODEL); original = head.predict
        with patch.object(head, 'predict', side_effect=lambda f:dict(original(f), proposed_skill='wait')):
            result = execute(course(count=2), self.folder(), head=head, timeout=240.)
        self.assertEqual(result['completed_jumps'], 0)
        self.assertEqual(result['reason'], 'refused')
        result = execute(course(count=2, spacing=1.5), self.folder(), head=head, timeout=240.)
        self.assertEqual(result['completed_jumps'], 0)
        self.assertFalse(result['collision'] or result['fall'])

    def test_interruption_and_closed_viewer_release_torques(self):
        head = RefinedHead.load(MODEL)
        result = execute(course(count=2), self.folder(), head=head, timeout=240., interrupt_after=.3)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertEqual(result['completed_jumps'], 0)
        self.assertTrue(result['terminal_stable'] and result['torques_released'])
        view = MagicMock(); view.is_running.return_value = False
        context = MagicMock(); context.__enter__.return_value = view
        with patch('experiments.obstacle_v5_multi.runner.mujoco.viewer.launch_passive', return_value=context):
            result = execute(course(), self.folder(), head=head, viewer=True)
        self.assertEqual(result['reason'], 'viewer closed')
        self.assertTrue(result['torques_released'])


if __name__ == '__main__':
    unittest.main()
