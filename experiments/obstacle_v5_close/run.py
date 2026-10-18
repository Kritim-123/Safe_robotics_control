"""Run V5 with shape-aware clearance on the previously refused close bars."""
import argparse
import json
from pathlib import Path
import time

from experiments.obstacle_v5_multi.run import course
from experiments.obstacle_v5_multi.learn import MODEL
from experiments.obstacle_v5_refinement.model import RefinedHead, ROOT
from .runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing', type=float, default=1.8)
    parser.add_argument('--count', type=int, choices=(2, 3, 4), default=3)
    parser.add_argument('--height', type=float, default=.025)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--viewer-speed', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1.2 <= args.spacing <= 3. or not .02 <= args.height <= .04:
        parser.error('Use spacing 1.2–3 m and height .02–.04 m')
    folder = args.output or ROOT/'output/obstacle_v5_close'/str(time.time_ns())
    result = execute(course(args.count, args.spacing, args.height), folder,
                     timeout=240., head=RefinedHead.load(MODEL), viewer=args.viewer,
                     video=args.video, viewer_speed=args.viewer_speed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
