"""Evaluate portable releases without requiring the original fitting folders."""
import argparse
from dataclasses import replace
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.observations import SensorConfig
from experiments.continual_snn.transfer import terrain_cases
from .loading_v2 import load_any
from .assessment import primary_config
from .goal_cases import goal_cases
from .batch import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('clean','noise','benchmark'),default='clean')
    p.add_argument('--model',type=Path,default=ROOT/'docs/agents/terrain_goal_v3/primary.npz')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int)
    p.add_argument('--seed',type=int);p.add_argument('--chunk',type=int,default=0)
    p.add_argument('--chunks',type=int,default=1);a=p.parse_args()
    if not 0<=a.chunk<a.chunks or (a.count is not None and a.count<1):raise ValueError('Invalid count/chunk')
    cfg=primary_config();sensor=None
    if a.phase=='benchmark':
        if a.count is not None or a.seed is not None:raise ValueError('Historical benchmark has fixed 60 cases and seed')
        cases=terrain_cases(20261117)
    elif a.phase=='clean':
        cases=goal_cases(15500 if a.seed is None else a.seed,100 if a.count is None else a.count,
                         'request_refinement_confirmation')
    else:
        cases=goal_cases(15501 if a.seed is None else a.seed,40 if a.count is None else a.count,
                         'request_refinement_noisy_confirmation')
        cfg=replace(cfg,selector_position_noise=.02,selector_yaw_noise=.02)
        sensor=SensorConfig(.03,.015,.1,9919)
    arms=[('v2','continuous',load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz')),
          ('candidate','continuous',load_any(a.model)),('rule','rule',None)]
    run(a.output,cases[a.chunk::a.chunks],arms,adaptation=cfg,sensor=sensor)


if __name__=='__main__':main()
