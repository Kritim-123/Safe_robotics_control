"""Finite live V4 demo and disjoint whole-scene evaluations."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from hybrid.arena import Obstacle
from .model import ROOT, MODEL, V3
from .train import scene
from .runner import execute


def controls():
    base = scene()
    return [
        replace(scene('tall', height=.08), name='v4-control-tall'),
        replace(base, name='v4-control-moving', obstacles=(replace(base.obstacles[0], velocity=(0., .1)),)),
        replace(base, name='v4-control-blocked-landing', obstacles=base.obstacles+(
            Obstacle('landing_block', (1., 0.), radius=.18),)),
        scene('control-low-friction', friction=.5),
        scene('control-distant', distance=.55),
        replace(base, name='v4-control-misaligned', obstacles=(replace(base.obstacles[0], center=(.55, 0.), yaw=.2),)),
        replace(base, name='v4-control-no-obstacle', obstacles=()),
    ]


def cases(split, count):
    rng = np.random.default_rng(18401 if split == 'validation' else 19401)
    return [scene(f'{split}-{i:02d}', rng.uniform(.23, .24), rng.uniform(.02, .04),
                  rng.uniform(.012, .02), rng.uniform(.8, .9)) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('demo', 'evaluate'), default='demo')
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--viewer-speed', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError('Count must be 1 to 100')
    folder = (args.output or ROOT/'output/obstacle_v4'/('demo-'+str(time.time_ns()))).resolve()
    if args.phase == 'demo':
        print(json.dumps(execute(scene(), folder, viewer=args.viewer, video=args.video,
                                 viewer_speed=args.viewer_speed), indent=2))
        return
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for spec in cases(args.split, args.count)+controls():
        row = execute(spec, folder/spec.name)
        rows.append(row)
        (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(spec.name, row['reason'], row['success'], flush=True)
    positives = rows[:args.count]; negatives = rows[args.count:-1]; clear = rows[-1]
    result = dict(split=args.split, low_bar_cases=len(positives),
                  autonomous_crossings=sum(r['jump_completed'] for r in positives),
                  reached_goals=sum(r['success'] for r in positives),
                  contacts=sum(r['collision'] for r in positives), falls=sum(r['fall'] for r in positives),
                  refused_low_bars=sum(r['reason'] == 'refused' for r in positives),
                  negative_controls=len(negatives), negative_jumps=sum(r['jump_initiated'] for r in negatives),
                  negative_contacts=sum(r['collision'] for r in negatives),
                  clear_goal_success=clear['success'], clear_jump=clear['jump_initiated'],
                  all_explicit_jump_requests=sum(r['explicit_jump_requests'] for r in rows),
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  v3_sha256=hashlib.sha256(V3.read_bytes()).hexdigest())
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
