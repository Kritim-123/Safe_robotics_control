"""Develop, assess and record an explicit low-bar crossing motor skill."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.smooth_jump import ROOT, save
from hybrid.arena import Scenario, Obstacle
from skills.jump_profile import JumpConfig
from skills.obstacle_jump import execute_obstacle_jump

PROFILE = ROOT/'docs/agents/obstacle_jump/config.json'


def bar_scene(name='low_bar_jump', *, distance=.24, height=.04, depth=.02,
              yaw=0., friction=.8, start=(0., 0.)):
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    return Scenario(name, start=start, start_yaw=yaw, goal=tuple(np.array(start)+forward*1.5),
        friction=friction, bounds=(-4., 5., -4., 4.), obstacles=(
            Obstacle('bar', tuple(np.array(start)+forward*distance), shape='box',
                     half_extents=(depth/2, .6), height=height, yaw=yaw),))


def trial(output, spec, config, *, video=False, viewer=False, interrupt_after=None):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):
        raise ValueError('Output must be inside project output directory')
    output.mkdir(parents=True, exist_ok=False)
    renderer = movie = None
    next_frame = 0.
    if video:
        import imageio.v2 as imageio
        movie = imageio.get_writer(str(output/'recording.mp4'), fps=60)
    def frame(model, data, row):
        nonlocal renderer, next_frame
        if movie is None or data.time < next_frame:
            return
        from PIL import Image, ImageDraw
        if renderer is None:
            renderer = mujoco.Renderer(model, height=480, width=800)
        camera = mujoco.MjvCamera()
        heading = np.array([np.cos(spec.start_yaw), np.sin(spec.start_yaw)])
        camera.lookat[:] = [*(np.array(spec.start)+heading*.30), .18]
        camera.distance, camera.azimuth, camera.elevation = 1.8, np.rad2deg(spec.start_yaw)+90., -16.
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 800, 62), fill='black')
        bar = spec.obstacles[0]
        draw.text((10, 6), f'{spec.name} | {row["phase"]} | t={data.time:.2f}s\n'
            f'Bar {bar.height*100:.1f} cm high / {bar.half_extents[0]*200:.1f} cm deep | '
            f'foot contacts {row["foot_contacts"]} | torque {max(abs(data.ctrl)):.1f} Nm\n'
            'Continuous MuJoCo physics; no external launch force', fill='white')
        movie.append_data(np.asarray(image)); next_frame += 1/60
    try:
        rows, summary = execute_obstacle_jump(spec, 'bar', config, frame_callback=frame,
            viewer=viewer, interrupt_after=interrupt_after)
    finally:
        if renderer is not None: renderer.close()
        if movie is not None: movie.close()
    fields = ('time', 'qpos', 'qvel', 'torque', 'feet', 'foot_contacts', 'body_rear', 'upright', 'normal_clearance')
    np.savez_compressed(output/'trace.npz', **{k: np.asarray([r[k] for r in rows]) for k in fields},
                        phase=np.asarray([r['phase'] for r in rows]))
    summary['interrupt_after'] = interrupt_after
    save(output/'summary.json', summary)
    save(output/'manifest.json', dict(mujoco=mujoco.__version__, numpy=np.__version__,
        sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'skills/obstacle_jump.py', 'skills/jump_profile.py', 'controllers/walking.py',
            'hybrid/arena.py', 'experiments/obstacle_jump.py')}))
    return summary


def develop(output):
    """Retain every trial of leading flat candidates against a real low bar."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'output/obstacle_jump/flat-development-v1/results.jsonl'
    candidates = [json.loads(line) for line in source.read_text().splitlines()]
    candidates = sorted([r for r in candidates if r['summary']['success']],
                        key=lambda r:r['metrics']['forward_progress'], reverse=True)[:12]
    results = []
    with (output/'results.jsonl').open('x') as stream:
        for i, row in enumerate(candidates):
            config = JumpConfig(**row['config'])
            for distance in (.26, .28, .30):
                name = f'candidate-{i:02d}-distance-{distance:.2f}'
                summary = trial(output/name, bar_scene(name, distance=distance,height=.01), config)
                results.append(summary); stream.write(json.dumps(summary)+'\n'); stream.flush()
                print(name, summary['reason'], np.round(summary['min_foot_bar_clearance'], 3)
                      if all(v is not None for v in summary['min_foot_bar_clearance']) else
                      summary['min_foot_bar_clearance'], summary['final_collision_geometry_margin'], flush=True)
    save(output/'successful.json', [r for r in results if r['success']])


def evaluate(output, config, *, axis=False):
    """Freeze exact cases before execution; no candidate changes on these results."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261030 if axis else 20261029)
    cases = []
    for i in range(16 if axis else 24):
        if axis:
            spec = bar_scene(f'axis-{i:02d}',distance=float(rng.uniform(.23,.24)),
                height=float(rng.uniform(.02,.04)),depth=float(rng.uniform(.012,.02)),
                friction=float(rng.uniform(.8,.9)),
                start=tuple(float(v) for v in rng.uniform(-.18,.18,2)))
            cases.append(('axis_held_out',spec,None))
            continue
        spec = bar_scene(f'held-{i:02d}', distance=float(rng.uniform(.23,.24)),
            height=float(rng.uniform(.02,.04)), depth=float(rng.uniform(.012,.02)),
            friction=float(rng.uniform(.8,.9)), yaw=float(rng.uniform(-np.pi,np.pi)),
            start=tuple(float(v) for v in rng.uniform(-.2,.2,2)))
        cases.append(('held_out', spec, None))
    for i, (distance, friction) in enumerate(() if axis else ((.23,.8),(.24,.8),(.23,.9),(.24,.9))):
        cases.append(('boundary',bar_scene(f'boundary-{i:02d}',distance=distance,friction=friction),None))
    for i, (distance, friction) in enumerate(() if axis else ((.25,.8),(.27,.8),(.30,.8),(.24,.5),(.24,.7),(.24,1.2))):
        cases.append(('stress',bar_scene(f'stress-{i:02d}',distance=distance,friction=friction),None))
    for i, when in enumerate(() if axis else (.30,.75)):
        cases.append(('interrupt',bar_scene(f'interrupt-{i:02d}'),when))
    save(output/'cases.json',[dict(split=split,scenario=spec.to_dict(),interrupt_after=when) for split,spec,when in cases])
    from dataclasses import asdict
    save(output/'frozen_config.json',asdict(config))
    results = []
    with (output/'results.jsonl').open('x') as stream:
        for split, spec, when in cases:
            summary = trial(output/spec.name,spec,config,interrupt_after=when)
            row = dict(split=split,summary=summary)
            results.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(split,spec.name,summary['reason'],round(summary['final_collision_geometry_margin'],3),flush=True)
    save(output/'aggregate.json',{split:dict(trials=sum(r['split']==split for r in results),
        successes=sum(r['summary']['success'] for r in results if r['split']==split),
        collisions=sum(r['summary']['collision'] for r in results if r['split']==split),
        falls=sum(r['summary']['fall'] for r in results if r['split']==split))
        for split in (('axis_held_out',) if axis else ('held_out','boundary','stress','interrupt'))})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('demo', 'develop', 'evaluate'), default='demo')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PROFILE)
    parser.add_argument('--height', type=float, default=.04)
    parser.add_argument('--distance', type=float, default=.24)
    parser.add_argument('--depth', type=float, default=.02)
    parser.add_argument('--axis', action='store_true',help='Evaluate the +X-only assessment split')
    parser.add_argument('--viewer', action='store_true'); parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT/args.output
    if args.phase == 'develop':
        develop(output)
    else:
        config = JumpConfig(**json.loads(args.config.read_text()))
        if args.phase == 'evaluate':
            evaluate(output,config,axis=args.axis)
        else:
            summary = trial(output, bar_scene(distance=args.distance, height=args.height, depth=args.depth),
                            config, video=args.video, viewer=args.viewer)
            print(json.dumps(summary, indent=2))
