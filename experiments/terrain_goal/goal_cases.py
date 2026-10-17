"""Fresh continuous goal/terrain perturbations, disjoint by entire episode."""
from dataclasses import replace
import numpy as np
from experiments.continual_snn.scenes import StreamCase,Request
from experiments.spike_pattern_skills.curriculum import curriculum


def goal_cases(seed,count,split,*,flat_only=False):
    rng=np.random.default_rng(seed);templates=curriculum(seed)
    result=[]
    for i in range(count):
        stage=0 if flat_only else i%10
        template=templates[stage*6+int(rng.integers(0,6))]
        yaw=float(rng.uniform(-.22,.22));distance=float(rng.uniform(.65,2.6))
        start=np.array([float(rng.uniform(-.08,.08)),float(rng.uniform(-.08,.08))])
        direction=np.array([np.cos(yaw),np.sin(yaw)])
        # Terrain courses need a goal beyond their end; flat goal tests cover
        # shorter distances independently. No failed benchmark is relabelled.
        if stage:distance=float(rng.uniform(1.7,2.6))
        goal=start+direction*distance
        patches=tuple(replace(p,height=p.height*float(rng.uniform(.85,1.15)),
            friction=p.friction*float(rng.uniform(.9,1.1))) for p in template.scenario.terrain)
        spec=replace(template.scenario,name=f'{split}-goal-{seed}-{i:03d}',start=tuple(start),
            goal=tuple(goal),start_yaw=yaw,friction=float(rng.uniform(.7,.95)),terrain=patches,
            timeout=65.)
        end=max(28.,distance/.09+15.)
        case=StreamCase(spec,'terrain_transfer',seed*1000+i,split,
            (Request(2.,end,'progress'),Request(end,end+10.,'hold')),finish_after=end+4.,timeout=end+10.,
            required_progress=0.,maximum_jumps=0)
        case.validate();result.append(case)
    return result
