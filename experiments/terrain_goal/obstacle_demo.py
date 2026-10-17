"""Finite V3 obstacle course; task requests are explicit, physics never resets."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hybrid.arena import Scenario, Obstacle
from experiments.continual_snn.scenes import StreamCase, Request
from experiments.terrain_goal.assessment import primary_config
from experiments.terrain_goal.loading_v2 import load_any
from experiments.terrain_goal.runner import execute
from experiments.terrain_goal.demo import CountMovie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    scene = Scenario(
        'v3-live-multiple-obstacles-and-hop', goal=(3.0, 0.0),
        bounds=(-2., 6., -4., 4.), timeout=44., friction=.85,
        obstacles=(
            Obstacle('clearing_gate', (1.05, 0.), radius=.18,
                     velocity=(0., .45), start_after=5., stop_after=5.),
            Obstacle('crossing_gate', (1.65, -1.4), radius=.18,
                     velocity=(0., .42), start_after=9., stop_after=7.),
            Obstacle('side_box_left', (2.4, .95), shape='box',
                     half_extents=(.28, .2), height=.5),
            Obstacle('side_box_right', (2.4, -.95), shape='box',
                     half_extents=(.28, .2), height=.5),
        ))
    case = StreamCase(scene, 'multiple_obstacles', 171026, 'live_demo', (
        Request(2., 14., 'progress'), Request(14., 20., 'jump'),
        Request(20., 36., 'progress'), Request(36., 44., 'hold')),
        finish_after=40., timeout=44., required_progress=.4,
        required_jumps=1, maximum_jumps=1)
    model = load_any(ROOT / 'docs/agents/terrain_goal_v3/primary.npz')
    folder = ROOT / 'output/terrain_goal' / ('live-v3-obstacles-' + str(time.time_ns()))
    print('V3 output: ' + str(folder), flush=True)
    movie = CountMovie(folder, case, model) if args.video else None
    previous = None

    def frame(m, d, supervisor):
        nonlocal previous
        state = (supervisor.current, supervisor.mode)
        if state != previous:
            print(f't={d.time:.2f}s skill={state[0]} phase={state[1]}', flush=True)
            previous = state
        if movie:
            movie(m, d, supervisor)

    try:
        result = execute(case, folder, model=model, adaptation=primary_config(),
                         viewer=args.viewer, full_trace=True, frame_callback=frame)
    finally:
        if movie:
            movie.close()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
