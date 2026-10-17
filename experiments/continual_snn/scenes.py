"""Causal request streams and reproducible whole-episode scene splits."""
from dataclasses import asdict,dataclass
import numpy as np

from hybrid.arena import Scenario,Obstacle

FAMILIES=('clear_stop','wait_release','late_crossing','jump_then_run','run_jump_run','sustained_jump','permanent_block','side_obstacle')
STRESS_FAMILIES=('two_hops','rapid_requests','cancelled_hop','cancel_preparation',
                 'oscillating_gate','multiple_obstacles','head_on','low_friction')


@dataclass(frozen=True)
class Request:
    start:float
    end:float
    task:str

    def validate(self):
        if (self.task not in ('progress','hold','jump') or not np.isfinite([self.start,self.end]).all()
                or self.start<0 or self.end<=self.start):
            raise ValueError('Invalid task request interval')


@dataclass(frozen=True)
class StreamCase:
    scenario:Scenario
    family:str
    seed:int
    split:str
    requests:tuple[Request,...]
    finish_after:float=16.
    timeout:float=22.
    required_progress:float=.4
    required_jumps:int=0
    hold_grace:float=2.5
    maximum_jumps:int|None=None

    def request_at(self,now):
        return next((r.task for r in self.requests if r.start<=now<r.end),'hold')

    def validate(self):
        self.scenario.validate()
        if self.family not in FAMILIES+STRESS_FAMILIES+('terrain_transfer',) or self.timeout<=self.finish_after or self.finish_after<5:
            raise ValueError('Invalid stream case')
        if self.required_progress<0 or self.required_jumps<0 or self.hold_grace<0:
            raise ValueError('Invalid stream acceptance')
        if self.maximum_jumps is not None and (type(self.maximum_jumps) is not int or self.maximum_jumps<self.required_jumps):
            raise ValueError('Invalid maximum hop count')
        previous=-1.
        for request in self.requests:
            request.validate()
            if request.start<previous:
                raise ValueError('Overlapping task requests')
            previous=request.end

    def to_dict(self):
        return asdict(self)


def make_case(family,seed,split='development',*,yaw=None):
    if family not in FAMILIES:
        raise ValueError('Unknown stream family')
    rng=np.random.default_rng(seed)
    heading=float(rng.uniform(-.15,.15)) if yaw is None else float(yaw)
    forward=np.array([np.cos(heading),np.sin(heading)])
    left=np.array([-forward[1],forward[0]])
    start=rng.uniform(-.1,.1,2)
    friction=float(rng.uniform(.75,.95))
    obstacles=()
    progress_end=float(rng.uniform(11.,12.5))
    requests=(Request(2.,progress_end,'progress'),Request(progress_end,35.,'hold'))
    progress=.4;jumps=0;finish=progress_end+4.
    if family=='wait_release':
        center=start+forward*float(rng.uniform(.95,1.15))
        obstacles=(Obstacle('clearing',tuple(center),radius=.18,
            velocity=tuple(left*float(rng.uniform(.40,.50))),start_after=float(rng.uniform(4.5,5.5))),)
    elif family=='late_crossing':
        center=start+forward*float(rng.uniform(1.35,1.55))-left*1.3
        obstacles=(Obstacle('crossing',tuple(center),radius=.18,
            velocity=tuple(left*float(rng.uniform(.38,.48))),start_after=float(rng.uniform(3.5,4.5))),)
    elif family=='jump_then_run':
        jump_end=float(rng.uniform(5.2,6.2))
        requests=(Request(2.,jump_end,'jump'),Request(jump_end,progress_end,'progress'),Request(progress_end,35.,'hold'))
        jumps=1
    elif family=='run_jump_run':
        jump_start=float(rng.uniform(4.5,5.5));jump_end=jump_start+float(rng.uniform(4.5,5.5))
        progress_end=jump_end+float(rng.uniform(4.5,5.5))
        requests=(Request(2.,jump_start,'progress'),Request(jump_start,jump_end,'jump'),
                  Request(jump_end,progress_end,'progress'),Request(progress_end,35.,'hold'))
        finish=progress_end+4.;jumps=1
    elif family=='sustained_jump':
        jump_end=float(rng.uniform(10.,12.))
        requests=(Request(2.,jump_end,'jump'),Request(jump_end,35.,'hold'))
        progress=0.;jumps=1;finish=jump_end+4.
    elif family=='permanent_block':
        obstacles=(Obstacle('blocker',tuple(start+forward*1.05),radius=.22),)
        requests=(Request(2.,30.,'hold'),)
        progress=0.
    elif family=='side_obstacle':
        obstacles=(Obstacle('side',tuple(start+forward*1.4+left*1.1),radius=.18),)
    spec=Scenario(f'{split}-{family}-{seed}',start=tuple(start),goal=tuple(start+forward*3.),
        start_yaw=heading,friction=friction,bounds=(-5.,6.,-5.,5.),obstacles=obstacles)
    result=StreamCase(spec,family,seed,split,requests,finish_after=finish,
                      timeout=finish+6.,required_progress=progress,required_jumps=jumps,
                      maximum_jumps=jumps)
    result.validate()
    return result


def cases(split='development',seeds=None):
    defaults={'development':range(4000,4002),'train':range(4100,4108),
              'validation':range(5100,5103),'test':range(6100,6105),
              'confirmation':range(7100,7105)}
    if split not in defaults:
        raise ValueError('Unknown scene split')
    return [make_case(family,seed,split) for seed in (seeds if seeds is not None else defaults[split])
            for family in FAMILIES]
