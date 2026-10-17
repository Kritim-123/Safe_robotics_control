"""V2 validation/test cohorts use new seeds; V1 noisy failures are development evidence."""
import argparse,time
from dataclasses import replace
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from experiments.continual_snn.observations import SensorConfig
from .assessment import primary_config
from .batch import run
from .goal_cases import goal_cases
from .hysteresis import HystereticGoalSelector
from .loading import load_selector_any
from .loading_v2 import load_any


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=30);p.add_argument('--flat-only',action='store_true')
    p.add_argument('--noise',choices=('none','mild','severe'),default='mild')
    p.add_argument('--phase',default='hysteresis_validation');p.add_argument('--deadline',type=float)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();v1path=ROOT/'docs/agents/terrain_goal_v1/primary.npz'
    v1=load_selector_any(v1path)
    v2=load_any(a.checkpoint) if a.checkpoint else HystereticGoalSelector.from_path(v1path)
    cfg=primary_config();sensor=None
    if a.noise=='mild':
        cfg=replace(cfg,selector_position_noise=.02,selector_yaw_noise=.02);sensor=SensorConfig(.03,.015,.1,9917)
    elif a.noise=='severe':
        cfg=replace(cfg,selector_position_noise=.05,selector_yaw_noise=.05);sensor=SensorConfig(.08,.05,.2,9918)
    scenes=goal_cases(a.seed,a.count,a.phase,flat_only=a.flat_only)
    run(a.output,scenes,[('v1','continuous',v1),('v2','continuous',v2),('rule','rule',None)],
        adaptation=cfg,sensor=sensor,deadline=a.deadline)


if __name__=='__main__':main()
