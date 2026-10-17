"""Predeclared stress diagnostics; do not fit neural parameters on their outcomes."""
from dataclasses import replace
import numpy as np

from hybrid.arena import Obstacle
from .scenes import StreamCase,Request,make_case,STRESS_FAMILIES


def make_stress_case(family,seed,split='stress'):
    if family not in STRESS_FAMILIES:raise ValueError('Unknown stress family')
    rng=np.random.default_rng(seed)
    base=make_case('clear_stop',seed,'development',yaw=float(rng.uniform(-np.pi,np.pi)))
    spec=replace(base.scenario,name=f'{split}-{family}-{seed}')
    heading=np.array([np.cos(spec.start_yaw),np.sin(spec.start_yaw)])
    left=np.array([-heading[1],heading[0]]);start=np.array(spec.start)
    requests=base.requests;finish=base.finish_after;required=.4;hops=0;maximum=0
    if family=='two_hops':
        requests=(Request(2.,6.,'jump'),Request(6.,10.,'progress'),Request(10.,16.,'jump'),
                  Request(16.,21.,'progress'),Request(21.,35.,'hold'))
        finish=25.;hops=maximum=2
    elif family=='rapid_requests':
        burst=tuple(Request(round(2.+i*.2,8),round(2.+(i+1)*.2,8),'progress' if i%2==0 else 'hold')
                    for i in range(15))
        requests=burst+(Request(5.,12.,'progress'),Request(12.,35.,'hold'))
        finish=16.
    elif family=='cancelled_hop':
        requests=(Request(2.,5.,'progress'),Request(5.,5.3,'jump'),Request(5.3,12.,'progress'),
                  Request(12.,35.,'hold'))
        finish=16.
    elif family=='cancel_preparation':
        requests=(Request(2.,2.8,'jump'),Request(2.8,11.,'progress'),Request(11.,35.,'hold'))
        finish=15.
    elif family=='oscillating_gate':
        spec=replace(spec,obstacles=(Obstacle('oscillator',tuple(start+heading*1.4),radius=.18,
            motion='oscillate',amplitude=tuple(left*1.4),period=float(rng.uniform(6.,8.))),))
    elif family=='multiple_obstacles':
        spec=replace(spec,obstacles=(Obstacle('near_side',tuple(start-heading*.5+left*.85),radius=.18),
            Obstacle('crossing',tuple(start+heading*1.4-left*1.4),radius=.18,
                     velocity=tuple(left*.42),start_after=4.),))
    elif family=='head_on':
        spec=replace(spec,obstacles=(Obstacle('approach',tuple(start+heading*3.),radius=.2,
            velocity=tuple(-heading*.22),start_after=3.),))
        # A stationary hold is inadequate against a continuing approach. This
        # deliberately documents the limited skill inventory, not a safe target.
    elif family=='low_friction':
        spec=replace(spec,friction=float(rng.uniform(.25,.5)))
        requests=(Request(2.,6.,'jump'),Request(6.,12.,'progress'),Request(12.,35.,'hold'))
        finish=16.;hops=maximum=1
    case=StreamCase(spec,family,seed,split,requests,finish_after=finish,timeout=finish+6.,
                    required_progress=required,required_jumps=hops,maximum_jumps=maximum)
    case.validate();return case


def stress_cases(seeds=range(8100,8105)):
    return [make_stress_case(family,seed) for seed in seeds for family in STRESS_FAMILIES]
