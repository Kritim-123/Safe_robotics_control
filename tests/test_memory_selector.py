import unittest
import numpy as np
from unittest.mock import patch
from hybrid.arena import scenario
from hybrid.system import HybridSystem

from hybrid.memory import TemporalSelector
from hybrid.snn import SpikingSelector
from hybrid.perception import choose_masked


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.x = SpikingSelector().weights['mean'].copy()

    def test_reset_matches_legacy(self):
        temporal = TemporalSelector(retain=False)
        legacy = SpikingSelector()
        for i in range(5):
            x = self.x + i*.03
            np.testing.assert_array_equal(temporal.observe(x, i*.1, 0)[0], legacy.predict(x)[0])

    def test_history_changes_same_final_input(self):
        a, b = TemporalSelector(), TemporalSelector()
        a.observe(self.x + .5, 0, 0)
        b.observe(self.x - .5, 0, 0)
        sa = a.observe(self.x, .1, 0)[0]
        sb = b.observe(self.x, .1, 0)[0]
        self.assertGreater(np.max(np.abs(sa-sb)), 0)

    def test_resets_identity_episode_and_gap(self):
        for cause in ('identity', 'episode', 'gap'):
            a, fresh = TemporalSelector(), TemporalSelector()
            a.observe(self.x+.5, 0, 0)
            now, identity = (.1, 1) if cause == 'identity' else (1., 0)
            if cause == 'episode':
                a.reset()
            np.testing.assert_array_equal(a.observe(self.x, now, identity)[0], fresh.observe(self.x, now, identity)[0])

    def test_duplicate_does_not_advance(self):
        a = TemporalSelector()
        first = a.observe(self.x, .1, 0)
        second = a.observe(self.x, .1, 0)
        self.assertIs(first, second)
        self.assertEqual(a.updates, 1)

    def test_invalid_clears_memory(self):
        a = TemporalSelector()
        for now in (float('nan'), -1, .05):
            a.observe(self.x, .1, 0)
            with self.assertRaises(ValueError):
                a.observe(self.x, now, 0)
            self.assertIsNone(a.last_time)
        with self.assertRaises(ValueError):
            a.observe(np.full(12, np.nan), 0, 0)

    def test_mask_and_no_action(self):
        self.assertEqual(choose_masked([99, 2, 1, 0], [False, True, True, True]), 1)
        self.assertIsNone(choose_masked([99, 2, 1, 0], [False]*4))
        with self.assertRaises(ValueError):
            choose_masked([np.nan]*4, [True]*4)

    def test_supervisor_reports_selector_failure(self):
        system = HybridSystem(scenario('static', 0), memory='retain', expanded_skills=True)
        with patch.object(system.selector, 'observe', side_effect=ValueError('invalid observation')):
            result = system.run()
        self.assertFalse(result.summary['success'])
        self.assertEqual(result.summary['reason'], 'selector failure')

    def test_no_eligible_action_fails_explicitly(self):
        system = HybridSystem(scenario('static', 0), memory='retain', expanded_skills=True)
        with patch('hybrid.system.choose_masked', return_value=None):
            result = system.run()
        self.assertEqual(result.summary['reason'], 'no applicable skill')
        self.assertFalse(result.summary['collision'])

    def test_expanded_hold_recovers_without_resetting_physics(self):
        result = HybridSystem(scenario('crossing', 0), memory='retain',
                              expanded_skills=True, replanning=True).run()
        self.assertTrue(result.summary['success'], result.summary)
        self.assertIn('yield_hold', [e.get('skill') for e in result.events])
        self.assertGreater(result.summary['returns_to_baseline'], 0)
        positions = np.array([[r['x'], r['y']] for r in result.history])
        self.assertLess(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)), .08)


if __name__ == '__main__':
    unittest.main()
