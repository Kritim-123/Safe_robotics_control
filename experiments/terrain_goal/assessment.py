"""Frozen-model final cohorts; no fitting or validation-based parameter edits."""
import argparse,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.loading import load_selector
from experiments.continual_snn.scenes import make_case,FAMILIES,Request
from experiments.continual_snn.stress_scenes import make_stress_case
from experiments.continual_snn.scenes import STRESS_FAMILIES
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.observations import SensorConfig
from .batch import run
from .contract import AdaptationConfig
from .goal_cases import goal_cases
from .loading import load_selector_any


def primary_config(**changes):
    return AdaptationConfig(goal_arrival=True,goal_decision_source='neural',terrain_mode='lift',terrain_lift=.12,
        gait_period=.8,gait_stance=.7,brake_rate=.5,geometry_mode='oriented_boxes',maximum_up_step=.05,**changes)


def ood_cases():
    rng=np.random.default_rng(1420001);result=[]
    for c in goal_cases(14200,100,'goal_ood'):
        yaw=float(rng.uniform(-.45,.45));distance=float(rng.uniform(2.7,4.2))
        start=np.asarray(c.scenario.start);goal=start+distance*np.array([np.cos(yaw),np.sin(yaw)])
        patches=tuple(replace(p,height=p.height*1.25,friction=p.friction*.8) for p in c.scenario.terrain)
        spec=replace(c.scenario,start_yaw=yaw,goal=tuple(goal),bounds=(-2.,8.,-4.,4.),
            terrain=patches,friction=float(rng.uniform(.4,.7)))
        end=distance/.08+18.
        result.append(replace(c,scenario=spec,requests=(Request(2.,end,'progress'),Request(end,end+10.,'hold')),
            finish_after=end+4.,timeout=end+10.))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('main','benchmark','neural_only','regression','gait_regression','noise','severe_noise','stale','ood','negative'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'docs/agents/terrain_goal_v1/primary.npz')
    p.add_argument('--chunk',type=int,default=0);p.add_argument('--chunks',type=int,default=1)
    p.add_argument('--count',type=int,default=1000);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if not 0<=a.chunk<a.chunks:raise ValueError('Invalid chunk')
    new=load_selector_any(a.model);old=load_selector(ROOT/'docs/agents/continual_snn_v2/primary.npz')
    cfg=primary_config();sensor=None
    arms=[('v2_same_motor','continuous',old),('new','continuous',new),('rule','rule',None)]
    if a.phase=='main':scenes=goal_cases(14000,a.count,'goal_test')
    elif a.phase=='benchmark':
        scenes=terrain_cases(20261117);arms=[('new','continuous',new),('rule','rule',None)]
    elif a.phase=='neural_only':
        scenes=terrain_cases(20261117);cfg=AdaptationConfig()
        arms=[('new_selector_old_motor','continuous',new),('v2_old_motor','continuous',old),('rule','rule',None)]
    elif a.phase in ('regression','gait_regression'):
        scenes=[make_case(f,s,'goal_regression') for s in range(14500,14510) for f in FAMILIES]
        if a.phase=='regression':
            scenes += [make_stress_case(f,s,'goal_stress_regression') for s in range(14600,14610) for f in STRESS_FAMILIES]
            cfg=AdaptationConfig()
        else:cfg=replace(cfg,goal_arrival=False)
        arms=[('v2','continuous',old),('new','continuous',new),('rule','rule',None)]
    elif a.phase in ('noise','severe_noise','stale'):
        scenes=goal_cases(14000,200,'goal_test')
        arms=[('new','continuous',new),('rule','rule',None)]
        if a.phase=='noise':
            scenes=scenes[:100];cfg=replace(cfg,selector_position_noise=.02,selector_yaw_noise=.02)
            sensor=SensorConfig(.03,.015,.1,8817)
        elif a.phase=='severe_noise':
            scenes=scenes[100:200];cfg=replace(cfg,selector_position_noise=.05,selector_yaw_noise=.05)
            sensor=SensorConfig(.08,.05,.2,8818)
        else:scenes=scenes[:20];sensor=SensorConfig(delay=.5,seed=8819)
    elif a.phase=='ood':scenes=ood_cases();arms=[('new','continuous',new),('rule','rule',None)]
    else:
        scenes=goal_cases(14000,30,'goal_test')
        arms=[('always_run','always_run',None),('always_wait','always_wait',None)]
    scenes=scenes[a.chunk::a.chunks]
    run(a.output,scenes,arms,adaptation=cfg,sensor=sensor,deadline=a.deadline)


if __name__=='__main__':main()
