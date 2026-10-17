"""Frozen forward-hop stress assessment; no fitting or profile changes."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np

from experiments.smooth_jump import ROOT, run_trial, save
from hybrid.arena import Scenario, Obstacle
from hybrid.terrain import TerrainPatch
from skills.jump_profile import JumpConfig

SEED = 20261026
CONFIG = ROOT/'docs/agents/smooth_forward_jump/config.json'


def cases():
    rng = np.random.default_rng(SEED)
    result = []
    def add(family, index, parameters, *, interrupt=None, **kwargs):
        spec = Scenario(f'{family}-{index:02d}', bounds=(-6., 6., -6., 6.), **kwargs)
        spec.validate()
        result.append(dict(family=family, scenario=spec, parameters=parameters, interrupt_after=interrupt))
    for i in range(10):
        yaw = float(rng.uniform(-np.pi, np.pi))
        start = tuple(float(v) for v in rng.uniform(-.3, .3, 2))
        add('flat', i, dict(yaw=yaw), start=start, start_yaw=yaw,
            goal=tuple(np.array(start)+3*np.array([np.cos(yaw), np.sin(yaw)])))
    for i, friction in enumerate((.10, .15, .20, .25, .30, .40, .55, .75, 1., 1.4)):
        add('friction', i, dict(friction=friction), friction=friction)
    for family, yaw in (('uphill', 0.), ('downhill', np.pi), ('cross_slope', np.pi/2)):
        for i, degrees in enumerate((1., 2., 3., 4., 5., 6., 8., 10., 12., 15.)):
            # Keep the initial support surface relatively low: start is .2 m
            # along the .75 m half-length wedge from its low edge.
            center = tuple(.55*np.array([np.cos(yaw), np.sin(yaw)]))
            patch = TerrainPatch('slope', center, (.75, 1.),
                                 float(1.5*np.tan(np.deg2rad(degrees))), kind='ramp', yaw=float(yaw))
            add(family, i, dict(slope_deg=degrees), terrain=(patch,))
    for i, height in enumerate(np.linspace(.005, .05, 10)):
        # One front foot starts on a small block; the other three use the plane.
        add('uneven_footing', i, dict(block_height=float(height)),
            terrain=(TerrainPatch('foot_block', (.20, .14), (.12, .10), float(height)),))
    for family in ('step_up', 'step_down'):
        for i, height in enumerate(np.linspace(.005, .05, 10)):
            edge = .30 if i % 2 == 0 else .40
            center = (edge+1., 0.) if family == 'step_up' else (edge-1., 0.)
            add(family, i, dict(step_height=float(height), edge_x=edge),
                terrain=(TerrainPatch('step', center, (1., 1.), float(height)),))
    for i, distance in enumerate(np.linspace(.8, 1.7, 10)):
        add('static_obstacle', i, dict(obstacle_x=float(distance)),
            obstacles=(Obstacle('blocker', (float(distance), 0.), radius=.20),))
    for i, speed in enumerate(np.linspace(.05, .50, 10)):
        # At selection (t=2) the obstacle is stationary 1.5 m ahead, then
        # approaches during preparation/flight/recovery; no force is injected.
        add('approaching_obstacle', i, dict(approach_speed=float(speed), start_after=2.),
            obstacles=(Obstacle('approach', (1.5, 0.), radius=.20,
                                velocity=(-float(speed), 0.), start_after=2.),))
    for family, times in (('prepare_interrupt', np.linspace(.0, .50, 10)),
                          ('launched_interrupt', np.linspace(.56, 1.50, 10))):
        for i, when in enumerate(times):
            add(family, i, dict(interrupt_after=float(when)), interrupt=float(when))
    return result


def outcome(result, interrupted):
    s, m = result['summary'], result['metrics']
    if s['reason'] == 'failed initiation':
        return 'refused'
    if s['reason'] == 'interrupted' and s['stable_landing'] and not s['collision'] and not s['fall']:
        return 'controlled_interruption'
    if interrupted:
        return 'interruption_failure'
    if m['qualifies']:
        return 'forward_qualified'
    if s['success']:
        return 'completed_short_or_backward'
    return 'execution_failure'


def aggregate(rows):
    groups = []
    for family in dict.fromkeys(row['family'] for row in rows):
        group = [row for row in rows if row['family'] == family]
        started = [row for row in group if row['summary']['actuator_limit'] == 28.]
        distances = [row['metrics']['forward_progress'] for row in started]
        forces = [row['metrics']['peak_landing_normal_force'] for row in started
                  if row['metrics']['peak_landing_normal_force'] is not None]
        groups.append(dict(family=family, trials=len(group), initiated=len(started),
            outcomes=dict(Counter(row['outcome'] for row in group)),
            reasons=dict(Counter(row['summary']['reason'] for row in group)),
            collisions=sum(row['summary']['collision'] for row in group),
            falls=sum(row['summary']['fall'] for row in group),
            forward_min=min(distances) if distances else None,
            forward_max=max(distances) if distances else None,
            forward_mean=float(np.mean(distances)) if distances else None,
            peak_landing_force=max(forces) if forces else None))
    return dict(trials=len(rows), outcomes=dict(Counter(r['outcome'] for r in rows)), groups=groups)


def run(output, *, replay=None, video=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):
        raise ValueError('Assessment output must stay inside this project output directory')
    output.mkdir(parents=True, exist_ok=False)
    config = JumpConfig(**json.loads(CONFIG.read_text(encoding='utf-8')))
    config.validate()
    declared = cases()
    save(output/'frozen_config.json', asdict(config))
    save(output/'cases.json', [dict(family=c['family'], scenario=c['scenario'].to_dict(),
                                  parameters=c['parameters'], interrupt_after=c['interrupt_after']) for c in declared])
    paths = ['experiments/jump_stress.py', 'experiments/smooth_jump.py', 'skills/dynamic.py',
             'skills/jump_profile.py', 'hybrid/arena.py', 'hybrid/terrain.py', str(CONFIG.relative_to(ROOT))]
    save(output/'manifest.json', dict(seed=SEED, python=sys.version, platform=platform.platform(),
        mujoco=mujoco.__version__, numpy=np.__version__, profile='smooth_forward',
        sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        frozen_before_execution=True, tuning=False, external_forces=False))
    selected = declared if replay is None else [c for c in declared if c['scenario'].name == replay]
    if not selected:
        raise ValueError('Unknown replay case')
    rows = []
    with (output/'results.jsonl').open('x', encoding='utf-8') as stream:
        for case in selected:
            result = run_trial(output/case['scenario'].name, case['scenario'], config=config,
                               interrupt_after=case['interrupt_after'], video=video)
            row = dict(family=case['family'], parameters=case['parameters'],
                       interrupt_after=case['interrupt_after'], **result)
            row['outcome'] = outcome(result, case['interrupt_after'] is not None)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            print(case['scenario'].name, row['outcome'], result['summary']['reason'],
                  round(result['metrics']['forward_progress'], 3), flush=True)
    save(output/'aggregate.json', aggregate(rows))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', help='Replay one declared scenario by exact name')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    run(args.output if args.output.is_absolute() else ROOT/args.output, replay=args.replay, video=args.video)
