"""Finite V5 walk-up, observed jump and goal demo, with disjoint evaluations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from hybrid.arena import Obstacle
from experiments.obstacle_v4.model import ROOT
from experiments.obstacle_v4.train import scene
from .runner import execute


def nominal():
    return replace(scene('v5-walk-up', distance=.9), name='v5-nominal', goal=(2.16, 0.))


def cases(split, count):
    rng = np.random.default_rng(18501 if split == 'validation' else 19501)
    result = []
    for i in range(count):
        distance = float(rng.uniform(.75, 1.2))
        spec = scene('unused', distance, rng.uniform(.02, .04), rng.uniform(.012, .02), rng.uniform(.8, .9))
        result.append(replace(spec, name=f'v5-{split}-{i:02d}', goal=(distance+1.26, 0.)))
    return result


def controls():
    base = nominal(); bar = base.obstacles[0]
    return [
        replace(base, name='v5-control-tall', obstacles=(replace(bar, height=.08),)),
        replace(base, name='v5-control-moving', obstacles=(replace(bar, velocity=(0., .1)),)),
        replace(base, name='v5-control-blocked-landing', obstacles=base.obstacles+(
            Obstacle('landing_block', (1.9, 0.), radius=.18),)),
        replace(base, name='v5-control-low-friction', friction=.5),
        replace(base, name='v5-control-deep-bar', obstacles=(replace(bar, half_extents=(.06, .6)),)),
        replace(base, name='v5-control-misaligned', obstacles=(replace(bar, yaw=.2),)),
        replace(base, name='v5-control-clear', obstacles=()),
    ]


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
    folder = (args.output or ROOT/'output/obstacle_v5'/('demo-'+str(time.time_ns()))).resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep evaluation outputs inside output/')
    if args.phase == 'demo':
        print(json.dumps(execute(nominal(), folder, viewer=args.viewer, video=args.video,
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
                  overshoots=sum(r['reason'] == 'launch approach overshoot' for r in positives),
                  negative_controls=len(negatives), negative_jumps=sum(r['jump_initiated'] for r in negatives),
                  negative_contacts=sum(r['collision'] for r in negatives),
                  clear_goal_success=clear['success'], clear_jump=clear['jump_initiated'],
                  all_explicit_jump_requests=sum(r['explicit_jump_requests'] for r in rows),
                  launch_entries=[dict(case=r['case'], **r['jump_entry']) for r in positives if r['jump_entry']])
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
