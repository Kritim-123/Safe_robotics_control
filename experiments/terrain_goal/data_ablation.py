"""Matched current-runtime comparison of representation and aggregation data."""
import argparse
from pathlib import Path
from .batch import run
from .contract import AdaptationConfig
from .goal_cases import goal_cases
from .loading import load_selector_any
from experiments.continual_snn.models import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=ROOT/'output/terrain_goal'
    paths={
        'gaussian_teacher':base/'fit-goals-teacher-v4/goal64_ridge_a0.001.npz',
        'gaussian_dagger':base/'fit-goals-dagger-v5/goal64_ridge_a0.001.npz',
        'monotonic_teacher':base/'factored-goal-teacher-v4/factored.npz',
        'monotonic_dagger':base/'factored-goal-dagger-v5/factored.npz'}
    arms=[(name,'continuous',load_selector_any(path)) for name,path in paths.items()]+[('rule','rule',None)]
    cfg=AdaptationConfig(goal_arrival=True,goal_decision_source='neural',terrain_mode='lift',terrain_lift=.12,
        gait_period=.8,gait_stance=.7,brake_rate=.5,geometry_mode='oriented_boxes',maximum_up_step=.05)
    run(a.output,goal_cases(13800,20,'goal_aggregation_validation',flat_only=True),arms,adaptation=cfg)


if __name__=='__main__':main()
