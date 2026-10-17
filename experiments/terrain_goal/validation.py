"""Predeclared physical model comparisons on validation scenes."""
import argparse
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.loading import load_selector
from experiments.continual_snn.scenes import make_case,FAMILIES
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.study import run_batch
from .batch import run
from .models import Selector
from .contract import AdaptationConfig


def load(path):
    from .loading import load_selector_any
    return load_selector_any(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['unchanged80_prototype','task96_ridge_a0.001','task96_prototype','factor32_ridge_a0.01','factor32_prototype'])
    p.add_argument('--terrain',action='store_true');p.add_argument('--stages',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--seed',type=int,default=20261417)
    args=p.parse_args()
    arms=[(name,'continuous',load(args.fit/(name+'.npz'))) for name in args.models]+[('rule','rule',None)]
    if args.terrain:
        scenes=[c for i,c in enumerate(terrain_cases(args.seed)) if i//6 in args.stages and i%6 in (0,3,5)]
    else:scenes=[make_case(f,s,'terrain_goal_validation') for s in range(12000,12003) for f in FAMILIES]
    run(args.output,scenes,arms)


if __name__=='__main__':main()
