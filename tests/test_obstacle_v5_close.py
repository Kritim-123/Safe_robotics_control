from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np

from hybrid.arena import Obstacle
from experiments.obstacle_v5_close.geometry import landing_clearance
from experiments.obstacle_v5_close.runner import execute
from experiments.obstacle_v5_multi.run import course
from experiments.obstacle_v5_multi.learn import MODEL
from experiments.obstacle_v5_refinement.model import ROOT, RefinedHead


class CloseSpacingTests(unittest.TestCase):
    def folder(self):
        root = ROOT/'output/obstacle_v5_close/unit-tests'
        root.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=root); self.addCleanup(temp.cleanup)
        return Path(temp.name)/'episode'

    def test_thin_bar_shape_with_unchanged_disc_and_sampling_margin(self):
        spec = course(spacing=1.8)
        result = landing_clearance(spec, 0, [.65, 0.], 0., 0.)
        expected = 2.7-.01-(.65+1.2)-.42-1.2/32/2
        self.assertAlmostEqual(result, expected)
        self.assertGreater(result, .15)
        self.assertLess(landing_clearance(course(spacing=1.5), 0, [.65, 0.], 0., 0.), .15)

    def test_other_cylinders_boxes_motion_and_bounds_are_checked(self):
        spec = course(count=2, spacing=1.8)
        for obstacle in (Obstacle('block', (1.6, 0.), radius=.18),
                         Obstacle('box', (1.6, .05), shape='box', half_extents=(.1, .3), yaw=.7),
                         Obstacle('moving', (1.6, .8), velocity=(0., -.4), stop_after=3.)):
            with self.subTest(obstacle=obstacle.name):
                blocked = replace(spec, obstacles=spec.obstacles+(obstacle,))
                self.assertLess(landing_clearance(blocked, 0, [.65, 0.], 0., 0.), .15)
        boundary = replace(spec, obstacles=(spec.obstacles[0],), bounds=(-4., 2., -4., 4.))
        self.assertLess(landing_clearance(boundary, 0, [.65, 0.], 0., 0.), .15)

    def test_both_former_refusals_complete_with_same_model(self):
        for spacing in (1.8, 2.):
            with self.subTest(spacing=spacing):
                folder = self.folder()
                result = execute(course(spacing=spacing, height=.025), folder, timeout=240.)
                self.assertTrue(result['success'] and result['terminal_stable'] and result['torques_released'])
                self.assertEqual(result['completed_jumps'], 3)
                self.assertEqual(result['head_checksum'], RefinedHead.load(MODEL).checksum)
                self.assertFalse(result['collision'] or result['fall'] or result['teacher_forced_collection'])
                self.assertEqual(result['explicit_jump_requests'], 0)
                self.assertLessEqual(result['max_torque_limit_ratio'], 1.)
                packets = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
                for entry in result['jump_entries']:
                    confirmations = [r for r in packets if entry['time']-.101 <= r['time'] <= entry['time']+1e-8]
                    self.assertEqual(len(confirmations), 2)
                    self.assertTrue(all(r['eligible'] and r['jump_proposal']['proposed_skill']=='low_bar_jump' for r in confirmations))
                with np.load(folder/'trace.npz') as trace:
                    self.assertLess(np.linalg.norm(np.diff(trace['values'][:, 1:4], axis=0), axis=1).max(), .02)

    def test_neural_refusal_and_real_landing_block_still_prevent_jump(self):
        head = RefinedHead.load(MODEL); original = head.predict
        with patch.object(head, 'predict', side_effect=lambda f:dict(original(f), proposed_skill='wait')):
            result = execute(course(count=2, spacing=1.8), self.folder(), head=head, timeout=240.)
        self.assertEqual(result['reason'], 'refused')
        self.assertEqual(result['completed_jumps'], 0)
        spec = course(count=2, spacing=1.8)
        spec = replace(spec, obstacles=spec.obstacles+(Obstacle('block', (1.7, 0.), radius=.18),))
        with patch.object(head, 'predict', side_effect=lambda f:dict(original(f), proposed_skill='low_bar_jump')):
            result = execute(spec, self.folder(), head=head, timeout=240.)
        self.assertFalse(result['jump_initiated'] or result['collision'] or result['fall'])
        self.assertGreater(result['vetoed_jump_proposals'], 0)

    def test_one_robot_reset_interruption_and_viewer_close(self):
        actual = mujoco.mj_step; held = []

        def step(m, d):
            held[:] = [d]; actual(m, d)

        with patch('experiments.obstacle_v5_close.runner.mujoco.mj_resetDataKeyframe', wraps=mujoco.mj_resetDataKeyframe) as reset, \
                patch('experiments.obstacle_v5_close.runner.mujoco.mj_step', side_effect=step):
            result = execute(course(count=2, spacing=1.8), self.folder(), timeout=240., interrupt_after=.3)
        self.assertEqual(result['reason'], 'interrupted')
        self.assertEqual(sum(c.args[1] is held[0] for c in reset.call_args_list), 1)
        self.assertTrue(result['terminal_stable'] and result['torques_released'])
        self.assertFalse(result['jump_completed'] or result['collision'])
        view = MagicMock(); view.is_running.return_value = False
        context = MagicMock(); context.__enter__.return_value = view
        with patch('experiments.obstacle_v5_close.runner.mujoco.viewer.launch_passive', return_value=context):
            result = execute(course(spacing=1.8), self.folder(), viewer=True)
        self.assertEqual(result['reason'], 'viewer closed')
        self.assertTrue(result['torques_released'])


if __name__ == '__main__':
    unittest.main()
