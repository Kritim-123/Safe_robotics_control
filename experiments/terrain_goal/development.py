"""Declared terrain gait/arrival development probes, never final holdouts."""
import argparse
from dataclasses import replace
from pathlib import Path
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.models import ROOT
from .batch import run
from .contract import AdaptationConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('none','lift','height','preview'),default='none')
    parser.add_argument('--lift',type=float,default=.08)
    parser.add_argument('--gain',type=float,default=1.)
    parser.add_argument('--arrival',action='store_true')
    parser.add_argument('--geometry',choices=('circles','oriented_boxes'),default='circles')
    parser.add_argument('--backend',choices=('classical','himloco'),default='classical')
    parser.add_argument('--brake-rate',type=float,default=.25)
    parser.add_argument('--body-raise',type=float,default=0.)
    parser.add_argument('--period',type=float,default=.6)
    parser.add_argument('--stance',type=float,default=.6)
    parser.add_argument('--speed',type=float,default=.35)
    parser.add_argument('--maximum-up-step',type=float,default=0.)
    parser.add_argument('--stop-radius',type=float,default=.08)
    parser.add_argument('--approach-speed',type=float,default=0.)
    parser.add_argument('--stages',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--variants',type=int,nargs='+',default=[0,3,5])
    parser.add_argument('--seed',type=int,default=20261317)
    args=parser.parse_args()
    scenes=[c for i,c in enumerate(terrain_cases(args.seed)) if i//6 in args.stages and i%6 in args.variants]
    if args.arrival:scenes=[replace(c,required_progress=0.) for c in scenes]
    cfg=AdaptationConfig(goal_arrival=args.arrival,terrain_mode=args.mode,terrain_lift=args.lift,terrain_gain=args.gain,
                         geometry_mode=args.geometry,motor_backend=args.backend,brake_rate=args.brake_rate,
                         body_raise=args.body_raise,gait_period=args.period,gait_stance=args.stance,terrain_speed=args.speed,
                         maximum_up_step=args.maximum_up_step,stop_radius=args.stop_radius,approach_speed=args.approach_speed)
    run(args.output,scenes,[('rule','rule',None)],adaptation=cfg)


if __name__=='__main__':main()
