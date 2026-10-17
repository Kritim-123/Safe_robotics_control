"""Goal stopping evaluations with mission intent retained at the neural input."""
import argparse
from pathlib import Path
from .batch import run
from .contract import AdaptationConfig
from .goal_cases import goal_cases
from .validation import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['factor32_ridge_a0.001','task96_ridge_a0.001','goal64_ridge_a0.0001','goal64_ridge_a0.001','goal64_ridge_a0.01'])
    p.add_argument('--seed',type=int,default=13100);p.add_argument('--count',type=int,default=30)
    p.add_argument('--backend',choices=('classical','himloco'),default='classical')
    p.add_argument('--flat-only',action='store_true');p.add_argument('--maximum-up-step',type=float,default=0.)
    p.add_argument('--profile',choices=('low','high'),default='low')
    p.add_argument('--phase',default='goal_model_validation')
    p.add_argument('--stop-radius',type=float,default=.08)
    p.add_argument('--approach-speed',type=float,default=0.)
    a=p.parse_args();scenes=goal_cases(a.seed,a.count,a.phase,flat_only=a.flat_only)
    cfg=AdaptationConfig(goal_arrival=True,goal_decision_source='neural',terrain_mode='lift',terrain_lift=.12 if a.profile=='high' else .08,
        brake_rate=.8 if a.backend=='himloco' else .5,geometry_mode='oriented_boxes',motor_backend=a.backend,
        maximum_up_step=a.maximum_up_step,gait_period=.8 if a.profile=='high' else .6,gait_stance=.7 if a.profile=='high' else .6,
        stop_radius=a.stop_radius,approach_speed=a.approach_speed)
    arms=[(n,'continuous',load(a.fit/(n+'.npz'))) for n in a.models]+[('rule','rule',None)]
    run(a.output,scenes,arms,adaptation=cfg)


if __name__=='__main__':main()
