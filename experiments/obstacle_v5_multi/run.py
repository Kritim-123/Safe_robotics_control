"""Finite repeated-jump course using the unchanged refined V5 models."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from experiments.obstacle_v4.train import scene
from experiments.obstacle_v5_refinement.model import ROOT
from .runner import execute


def course(count=3, spacing=2.2, height=.03, friction=.85, first=.9):
    base = scene(distance=first, height=height, friction=friction)
    bars = tuple(replace(base.obstacles[0], name=f'bar_{i+1}', center=(first+i*spacing, 0.))
                 for i in range(count))
    goal = bars[-1].center[0]+1.26
    return replace(base, name=f'v5-multi-{count}-{spacing:.2f}-{height:.3f}',
                   obstacles=bars, goal=(goal, 0.), bounds=(-4., goal+2., -4., 4.), timeout=240.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--spacing', type=float, default=2.2)
    parser.add_argument('--height', type=float, default=.03)
    parser.add_argument('--friction', type=float, default=.85)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--viewer-speed', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    if not 1.5 <= args.spacing <= 3. or not .02 <= args.height <= .04 or not .8 <= args.friction <= .9:
        parser.error('Use spacing 1.5–3 m, height .02–.04 m and friction .8–.9')
    output = args.output or ROOT/'output/obstacle_v5_multi'/str(time.time_ns())
    from .learn import MODEL
    from experiments.obstacle_v5_refinement.model import RefinedHead
    head = RefinedHead.load(args.model or MODEL)
    result = execute(course(args.count, args.spacing, args.height, args.friction), output, head=head,
                     timeout=240., viewer=args.viewer, video=args.video, viewer_speed=args.viewer_speed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
