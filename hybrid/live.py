"""Synchronized live MuJoCo video and SNN telemetry in one local browser UI."""
import argparse
import base64
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
import json
from pathlib import Path
import threading
import time
import webbrowser

import mujoco
from PIL import Image

from hybrid.arena import scenario
from hybrid.inspector import Inspector
from hybrid.run import save_result
from hybrid.snn import SpikingSelector
from hybrid.system import HybridSystem

DEMO_SCENARIOS = ('mixed_course', 'crate', 'moving_gate', 'oscillating_gate', 'late_crossing')


class LiveSession:
    def __init__(self, output, speed=2.0):
        if not 0 < speed <= 8:
            raise ValueError('Playback speed must be between 0 and 8')
        self.output, self.speed = Path(output), speed
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.play = threading.Event()
        self.play.set()
        self.thread = None
        self.run_number = 0
        self.options = dict(scenario='mixed_course', reaction='fast', seed=0)
        self.camera_mode = 'follow'
        self.snapshot = dict(status='ready', frame=None, decision=None, row=None)

    def state(self):
        with self.lock:
            return dict(self.snapshot)

    def command(self, action, options=None):
        with self.lock:
            if action == 'start':
                if self.thread is not None and self.thread.is_alive():
                    return
                supplied = options or {}
                name = supplied.get('scenario', 'mixed_course')
                reaction = supplied.get('reaction', 'fast')
                seed = supplied.get('seed', 0)
                if name not in DEMO_SCENARIOS or reaction not in ('fast', 'standard') or type(seed) is not int or not 0 <= seed <= 100000:
                    raise ValueError('Invalid scenario, reaction profile or seed')
                self.options = dict(scenario=name, reaction=reaction, seed=seed)
                self.run_number += 1
                self.stop.clear()
                self.play.set()
                self.snapshot = dict(status='starting', frame=None, decision=None, row=None)
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
            elif action == 'pause':
                self.play.clear()
            elif action == 'resume':
                self.play.set()
            elif action == 'stop':
                self.stop.set()
                self.play.set()
            elif action == 'camera':
                mode = (options or {}).get('camera')
                if mode not in ('follow', 'overview'):
                    raise ValueError('Invalid camera mode')
                self.camera_mode = mode
            else:
                raise ValueError('Unknown command')

    def run(self):
        renderer = None
        try:
            spec = scenario(self.options['scenario'], self.options['seed'])
            robot = HybridSystem(spec, replanning=True, max_speed=.24, reaction=self.options['reaction'])
            inspector = Inspector.__new__(Inspector)
            inspector.selector = SpikingSelector()
            inspector.events = []
            decision = None
            camera = mujoco.MjvCamera()
            x0, x1, y0, y1 = spec.bounds
            camera.lookat[:] = [(x0+x1)/2, (y0+y1)/2, .15]
            camera.distance, camera.azimuth, camera.elevation = 1.05*max(x1-x0, y1-y0), 90, -65
            last_wall = time.perf_counter()
            last_sim = 0.0
            timeline = []
            map_spec = dict(bounds=spec.bounds, start=spec.start, goal=spec.goal,
                            obstacles=[dict(name=o.name, shape=o.shape, radius=o.planning_radius,
                                            half_extents=o.half_extents, yaw=o.yaw,
                                            moving=o.motion == 'oscillate' or any(o.velocity)) for o in spec.obstacles])

            def on_event(event):
                nonlocal decision
                timeline.append({k: event[k] for k in ('time', 'target', 'reason', 'skill') if k in event})
                if 'features' in event:
                    inspector.events.append(event)
                    decision = dict(id=f'{self.run_number}:{len(inspector.events)}', event=event, analysis=inspector.evaluate(dict(
                        decision=len(inspector.events)-1, mask_mode='recorded')))

            def on_frame(model, data, state, row):
                nonlocal renderer, last_wall, last_sim
                if renderer is None:
                    renderer = mujoco.Renderer(model, height=480, width=720)
                if self.camera_mode == 'follow':
                    camera.lookat[:] = [row['x'], row['y'], .15]
                    camera.distance, camera.elevation = 3.4, -55
                else:
                    camera.lookat[:] = [(x0+x1)/2, (y0+y1)/2, .15]
                    camera.distance, camera.elevation = 1.05*max(x1-x0, y1-y0), -65
                renderer.update_scene(data, camera=camera)
                encoded = io.BytesIO()
                Image.fromarray(renderer.render()).save(encoded, format='JPEG', quality=80)
                # One atomic snapshot pairs a physics frame and its telemetry.
                with self.lock:
                    self.snapshot = dict(status='running', frame=base64.b64encode(encoded.getvalue()).decode(),
                                         row=row, decision=decision, playback_speed=self.speed,
                                         timeline=list(timeline), arena=map_spec, options=self.options,
                                         sensing_ms=robot.sensing_dt*1000, confirmation_ms=robot.confirmation_dt*1000)
                while not self.play.is_set() and not self.stop.is_set():
                    with self.lock:
                        self.snapshot = {**self.snapshot, 'status': 'paused'}
                    self.play.wait(.1)
                    last_wall = time.perf_counter()
                delay = max(0, (float(data.time)-last_sim)/self.speed-(time.perf_counter()-last_wall))
                self.stop.wait(delay)
                last_wall, last_sim = time.perf_counter(), float(data.time)

            result = robot.run(record_callback=on_frame, event_callback=on_event, should_stop=self.stop.is_set)
            result.summary['live_playback_speed'] = self.speed
            save_result(result, self.output)
            with self.lock:
                self.snapshot = {**self.snapshot, 'status': 'finished', 'summary': result.summary, 'timeline': list(timeline)}
        except Exception as error:
            with self.lock:
                self.snapshot = {**self.snapshot, 'status': 'error', 'error': str(error)}
        finally:
            if renderer is not None:
                renderer.close()


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, value, mime='application/json'):
            data = value.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime+'; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.reply(200, Path(__file__).with_name('live_dashboard.html').read_text(encoding='utf-8'), 'text/html')
            elif self.path == '/live.js':
                self.reply(200, Path(__file__).with_name('live_dashboard.js').read_text(encoding='utf-8'), 'text/javascript')
            elif self.path == '/api/state':
                state = session.state()
                decision = state.pop('decision')
                state['decision_id'] = decision['id'] if decision else None
                self.reply(200, json.dumps(state, allow_nan=False))
            elif self.path == '/api/decision':
                self.reply(200, json.dumps(session.state()['decision'], allow_nan=False))
            else:
                self.reply(404, '{}')

        def do_POST(self):
            if self.path not in ('/api/start', '/api/pause', '/api/resume', '/api/stop', '/api/camera'):
                return self.reply(404, '{}')
            if self.headers.get('Origin') not in (None, f'http://{self.headers.get("Host")}'):
                return self.reply(403, '{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 2048:
                    raise ValueError('Invalid request size')
                options = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(options, dict):
                    raise ValueError('Expected an object')
                session.command(self.path.rsplit('/', 1)[-1], options)
            except (ValueError, TypeError) as error:
                return self.reply(400, json.dumps(dict(error=str(error))))
            self.reply(200, '{}')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--speed', type=float, default=2)
    parser.add_argument('--output', type=Path, default=Path('output/hybrid/live'))
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    session = LiveSession(args.output, args.speed)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(session)) as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Go2 live dashboard: {url}', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            session.command('stop')
            if session.thread:
                session.thread.join(timeout=5)


if __name__ == '__main__':
    main()
