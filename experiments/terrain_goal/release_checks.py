"""Matched final-release benchmark and physical run/jump request parity."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.scenes import make_case
from experiments.continual_snn.stress_scenes import make_stress_case
from experiments.continual_snn.study import save
from .assessment import primary_config
from .loading_v2 import load_any
from .batch import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('benchmark','parity'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    v1=load_any(ROOT/'docs/agents/terrain_goal_v1/primary.npz');v2=load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz')
    arms=[('v1','continuous',v1),('v2','continuous',v2)]
    if a.phase=='benchmark':
        run(a.output,terrain_cases(20261117),arms+[('rule','rule',None)],adaptation=primary_config())
    else:
        scenes=[make_case('run_jump_run',14700,'release_parity'),make_case('sustained_jump',14700,'release_parity'),
                make_stress_case('cancel_preparation',14700,'release_parity')]
        run(a.output,scenes,arms,adaptation=replace(primary_config(),goal_arrival=False),full_trace=True)
        results=[]
        for case in scenes:
            path=a.output/case.scenario.name
            with np.load(path/'v1/trace.npz') as x,np.load(path/'v2/trace.npz') as y:
                identical=np.array_equal(x['values'],y['values'])
            results.append(dict(case=case.scenario.name,bit_identical_physics=identical))
            if not identical:raise AssertionError('V2 changed non-goal skill execution')
        save(a.output/'physical_parity.json',dict(paired_episodes=len(results),all_physics_bit_identical=True,cases=results))


if __name__=='__main__':main()
