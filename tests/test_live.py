"""Live observers must see the same decisions and times as episode records."""
from dataclasses import replace
import unittest

from hybrid.arena import scenario
from hybrid.system import HybridSystem
from hybrid.perception import choose_masked
from skills.avoidance import SKILL_NAMES
from hybrid.live import LiveSession, handler_for
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import json
import tempfile
import threading


class LiveTelemetryTests(unittest.TestCase):
    def test_live_api_isolates_large_traces_and_validates_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            session = LiveSession(directory)
            session.snapshot['decision'] = dict(id='1:1', event=dict(time=2), analysis=dict(trace=[1, 0]))
            with ThreadingHTTPServer(('127.0.0.1', 0), handler_for(session)) as server:
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                url = f'http://127.0.0.1:{server.server_port}'
                try:
                    with urlopen(url+'/api/state') as response:
                        state = json.load(response)
                    self.assertNotIn('decision', state)
                    self.assertEqual(state['decision_id'], '1:1')
                    with urlopen(url+'/api/decision') as response:
                        self.assertEqual(json.load(response)['analysis']['trace'], [1, 0])
                    self.assertIn('decision', session.state())
                    with urlopen(url+'/live.js') as response:
                        self.assertIn('drawEvidence', response.read().decode())
                    req = Request(url+'/api/start', data=b'{"scenario":"invalid"}', headers={'Content-Type':'application/json'})
                    with self.assertRaises(HTTPError) as error:
                        urlopen(req)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                    self.assertIsNone(session.thread)
                    session.command('camera', {'camera':'overview'})
                    self.assertEqual(session.camera_mode, 'overview')
                finally:
                    server.shutdown()
                    worker.join()

    def test_event_and_frame_callbacks_match_saved_records(self):
        events, frames = [], []

        def frame(model, data, state, row):
            self.assertAlmostEqual(data.time, row['time'])
            self.assertEqual(state, row['state'])
            frames.append(row)

        system = HybridSystem(replace(scenario('static'), timeout=3))
        result = system.run(event_callback=events.append, record_callback=frame,
                            should_stop=lambda: len(frames) >= 24)
        self.assertEqual(result.summary['reason'], 'user stopped')
        self.assertEqual(events, result.events)
        self.assertEqual(frames, result.history)
        choices = [e for e in events if 'features' in e]
        self.assertEqual(len(choices), 1)
        event = choices[0]
        self.assertEqual(event['skill'], SKILL_NAMES[choose_masked(event['scores'], event['allowed'])])
        self.assertLessEqual(event['time'], frames[-1]['time'])


if __name__ == '__main__':
    unittest.main()
