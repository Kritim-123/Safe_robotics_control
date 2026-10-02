"""Recorded neural signals must describe exactly the production inference."""
import unittest
import numpy as np
from hybrid.snn import SpikingSelector


class TraceTests(unittest.TestCase):
    def test_trace_matches_inference_and_firing_equations(self):
        selector = SpikingSelector()
        for features in np.random.default_rng(29).normal(size=(8, 12)):
            scores, plain = selector.predict(features)
            replay, trace = selector.predict(features, trace=True)
            np.testing.assert_array_equal(scores, replay)
            np.testing.assert_array_equal(trace['readouts'][-1], scores)
            np.testing.assert_array_equal(trace['raster'], trace['membrane_before_reset'] >= 1)
            np.testing.assert_allclose(trace['membrane_after_reset'],
                                       trace['membrane_before_reset'] - trace['raster'])
            self.assertEqual(int(trace['raster'].sum()), plain['spikes'])


if __name__ == '__main__':
    unittest.main()
