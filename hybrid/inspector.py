"""Local interactive UI for exact SNN inference and controlled input experiments."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser

import numpy as np

from hybrid.perception import FEATURE_NAMES, choose_masked
from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from skills.avoidance import SKILL_NAMES

UNITS = ['m', 'm', 'm/s', 'm/s', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'rad']
DESCRIPTIONS = [
    'Obstacle distance along the direction toward the goal.',
    'Obstacle position to the left of the goal direction; negative means right.',
    'Obstacle velocity along the goal direction; negative means approaching along that axis.',
    'Obstacle velocity to the left of the goal direction; negative means rightward.',
    'Radius of the selected obstacle.',
    'Minimum current clearance from the robot footprint to obstacles or walls.',
    'Minimum clearance along the proposed left bypass. Must exceed 0.045 m.',
    'Minimum clearance along the proposed right bypass. Must exceed 0.045 m.',
    'Minimum clearance along the proposed back-off path. Must exceed 0.025 m.',
    'Minimum predicted clearance while standing over the next 3 s. Must exceed 0.070 m.',
    'Distance from the robot to its goal.',
    'Angle from the robot heading to the goal direction.',
]


def mask_for(features):
    return np.array([features[6] > .045, features[7] > .045,
                     features[9] > .07, features[8] > .025])


class Inspector:
    def __init__(self, directory, model=DEFAULT_MODEL):
        self.selector = SpikingSelector(model)
        directory = Path(directory)
        self.summary = json.loads((directory/'summary.json').read_text())
        if self.summary.get('mode') != 'snn':
            raise ValueError('Select a saved SNN run')
        if self.summary['model_sha256'] != hashlib.sha256(Path(model).read_bytes()).hexdigest():
            raise ValueError('Model hash must match the saved episode')
        self.events = [e for e in json.loads((directory/'events.json').read_text()) if 'features' in e]
        if not self.events:
            raise ValueError('This run has no SNN decisions to inspect')
        for event in self.events:
            scores, _ = self.selector.predict(event['features'])
            np.testing.assert_allclose(scores, event['scores'], atol=2e-5, rtol=2e-5)

    def metadata(self):
        w = self.selector.weights
        return dict(events=self.events, summary=self.summary, names=FEATURE_NAMES,
                    skills=SKILL_NAMES, units=UNITS, descriptions=DESCRIPTIONS,
                    mean=w['mean'].tolist(), std=w['std'].tolist(), steps=self.selector.steps)

    def evaluate(self, payload):
        index = payload.get('decision', 0)
        if type(index) is not int or not 0 <= index < len(self.events):
            raise ValueError('Invalid decision index')
        event = self.events[index]
        features = np.asarray(payload.get('features', event['features']), dtype=np.float32)
        scores, trace = self.selector.predict(features, trace=True)
        if features[4] <= 0 or features[10] < 0:
            raise ValueError('Radius must be positive and goal distance nonnegative')
        mode = payload.get('mask_mode', 'recompute')
        if mode not in ('recompute', 'recorded'):
            raise ValueError('Unknown mask mode')
        allowed = mask_for(features) if mode == 'recompute' else np.asarray(event['allowed'])
        selected = choose_masked(scores, allowed)
        base_scores, _ = self.selector.predict(event['features'])
        # Symmetric, finite interventions on one feature at a time. No gradients
        # or additive attribution claims: binary firing is discontinuous.
        sensitivity = []
        steps = self.selector.weights['std'] * .25
        for i, delta in enumerate(steps):
            low, high = features.copy(), features.copy()
            low[i] -= delta
            high[i] += delta
            if i in (4, 10):
                low[i] = max(low[i], .001 if i == 4 else 0)
            variants = []
            for vector in (low, high):
                changed_scores, _ = self.selector.predict(vector)
                changed_mask = mask_for(vector) if mode == 'recompute' else allowed
                variants.append(dict(value=float(vector[i]),
                                     delta_scores=(changed_scores-scores).tolist(),
                                     selected=choose_masked(changed_scores, changed_mask)))
            sensitivity.append(dict(low=variants[0], high=variants[1]))
        winner = selected if selected is not None else int(np.argmax(scores))
        alternatives = [i for i in range(4) if i != winner and allowed[i]]
        runner = max(alternatives, key=lambda i: scores[i]) if alternatives else None
        rates = trace['raster'][:, 64:].mean(axis=0)
        contributions = self.selector.weights['w3'][winner]*rates
        evidence = dict(skill=winner, per_neuron=contributions.tolist(),
                        bias=float(self.selector.weights['b3'][winner]), runner=runner,
                        comparison=None)
        if runner is not None:
            evidence['comparison'] = dict(per_neuron=((self.selector.weights['w3'][winner]-self.selector.weights['w3'][runner])*rates).tolist(),
                                         bias=float(self.selector.weights['b3'][winner]-self.selector.weights['b3'][runner]),
                                         margin=float(scores[winner]-scores[runner]))
        return dict(scores=scores.tolist(), baseline_scores=base_scores.tolist(), evidence=evidence,
                    allowed=allowed.tolist(), selected=selected,
                    preference=int(np.argmax(scores)), sensitivity=sensitivity,
                    trace={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in trace.items()})


def handler_for(inspector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, body, content_type='application/json'):
            data = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.send(200, Path(__file__).with_name('inspector.html').read_text(encoding='utf-8'), 'text/html')
            elif self.path == '/api/meta':
                self.send(200, json.dumps(inspector.metadata(), allow_nan=False))
            else:
                self.send(404, '{"error":"Not found"}')

        def do_POST(self):
            if self.path != '/api/evaluate':
                return self.send(404, '{"error":"Not found"}')
            # Local read-only model endpoint, no file paths or commands accepted.
            if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
                return self.send(403, '{"error":"Cross-origin request rejected"}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected an object')
                self.send(200, json.dumps(inspector.evaluate(payload), allow_nan=False))
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.send(400, json.dumps(dict(error=str(error))))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('output/hybrid/latest'))
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    inspector = Inspector(args.run, args.model)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(inspector)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'SNN inspector: {url} — Ctrl+C to stop', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
