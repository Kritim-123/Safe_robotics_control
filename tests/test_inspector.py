"""Inspector uses production predictions and keeps mask effects separate."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np

from hybrid.inspector import Inspector, mask_for, handler_for
from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from hybrid.perception import choose_masked


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.features = [1.3, 0, 0, 0, .23, .65, .28, .28, .5, .65, 3, 0]
        scores, _ = SpikingSelector().predict(self.features)
        (root/'summary.json').write_text(json.dumps(dict(mode='snn', model_sha256=hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest())))
        (root/'events.json').write_text(json.dumps([dict(features=self.features, scores=scores.tolist(), allowed=[True]*4)]))
        self.inspector = Inspector(root)

    def test_replay_matches_production(self):
        result = self.inspector.evaluate({})
        expected, trace = SpikingSelector().predict(self.features, trace=True)
        np.testing.assert_array_equal(result['scores'], expected)
        np.testing.assert_array_equal(result['trace']['raster'], trace['raster'])
        self.assertEqual(result['selected'], choose_masked(expected, mask_for(self.features)))

    def test_sensitivity_matches_independent_intervention(self):
        result = self.inspector.evaluate({})
        for i, entry in enumerate(result['sensitivity']):
            for variant in (entry['low'], entry['high']):
                vector = np.array(self.features, dtype=np.float32)
                vector[i] = variant['value']
                scores, _ = self.inspector.selector.predict(vector)
                np.testing.assert_allclose(variant['delta_scores'], scores-np.array(result['scores'], dtype=np.float32))
                self.assertEqual(variant['selected'], choose_masked(scores, mask_for(vector)))

    def test_frozen_mask_and_no_available_skill(self):
        f = self.features.copy()
        f[6:10] = [-1]*4
        recomputed = self.inspector.evaluate(dict(features=f))
        frozen = self.inspector.evaluate(dict(features=f, mask_mode='recorded'))
        self.assertIsNone(recomputed['selected'])
        self.assertEqual(frozen['allowed'], [True]*4)
        np.testing.assert_array_equal(recomputed['scores'], frozen['scores'])

    def test_invalid_inputs_and_wrong_model(self):
        for payload in [dict(decision=-1), dict(features=[0]*12), dict(features=[float('nan')]*12), dict(mask_mode='invalid')]:
            with self.assertRaises(ValueError):
                self.inspector.evaluate(payload)
        root = Path(self.folder.name)
        (root/'summary.json').write_text(json.dumps(dict(mode='snn', model_sha256='wrong')))
        with self.assertRaises(ValueError):
            Inspector(root)

    def test_http_ui_and_evaluation(self):
        with ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.inspector)) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            root = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(root) as response:
                    self.assertIn('What makes the robot choose a skill?', response.read().decode())
                with urlopen(root+'/api/meta') as response:
                    self.assertEqual(len(json.load(response)['names']), 12)
                request = Request(root+'/api/evaluate', data=b'{}', headers={'Content-Type': 'application/json'})
                with urlopen(request) as response:
                    self.assertEqual(len(json.load(response)['sensitivity']), 12)
                request = Request(root+'/api/evaluate', data=b'{"decision":-1}', headers={'Content-Type': 'application/json'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
            finally:
                server.shutdown()
                worker.join()


if __name__ == '__main__':
    unittest.main()
