"""Fresh procedural terrain transfer; no new motor skill or training labels."""
from dataclasses import replace
from experiments.spike_pattern_skills.curriculum import curriculum
from .scenes import StreamCase,Request,make_case,FAMILIES
import numpy as np


def terrain_cases(seed=20261117):
    result=[]
    for item in curriculum(seed):
        spec=replace(item.scenario,name='transfer-'+item.scenario.name,timeout=32.)
        case=StreamCase(spec,'terrain_transfer',item.scene_seed,'terrain_transfer',
            (Request(2.,22.,'progress'),Request(22.,35.,'hold')),finish_after=26.,timeout=32.,
            required_progress=item.traversal_x,maximum_jumps=0)
        case.validate();result.append(case)
    return result


def heading_cases(seeds=range(9200,9220)):
    result=[]
    for seed in seeds:
        yaw=float(np.random.default_rng(seed).uniform(-np.pi,np.pi))
        result.extend(make_case(family,seed,'heading_transfer',yaw=yaw) for family in FAMILIES)
    return result
