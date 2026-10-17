"""Simulation-only smooth short-forward-hop trajectory, independent of SNN weights."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class JumpConfig:
    preparation: float = .55
    thrust: float = .22
    flight: float = .16
    absorption: float = .20
    recovery: float = .60
    crouch_thigh: float = 1.10
    crouch_calf: float = -2.20
    extension_calf: float = -1.2
    forward_lean: float = .40
    flight_calf: float = -1.55
    flight_lean: float = -.10
    absorption_thigh: float = 1.03
    absorption_calf: float = -2.06
    torque_limit: float = 28.
    landing_kp: float = 35.
    landing_kd: float = 1.5
    forward_clearance: float = .50

    def validate(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all():
            raise ValueError('Jump configuration must be finite')
        if min(self.preparation,self.thrust,self.flight,self.absorption,self.recovery,
               self.torque_limit,self.landing_kp,self.landing_kd,self.forward_clearance) <= 0:
            raise ValueError('Jump durations, gains, bound and clearance must be positive')
        if self.thrust < .06 or self.preparation < .2 or self.recovery < .2:
            raise ValueError('Jump phase too short for the control timestep')


def quintic_segment(start,end,elapsed,duration):
    """C2 segment with zero first/second derivatives at both endpoints."""
    s = float(np.clip(elapsed/duration,0.,1.))
    blend = 10*s**3-15*s**4+6*s**5
    velocity = (30*s**2-60*s**3+30*s**4)/duration
    acceleration = (60*s-180*s**2+120*s**3)/duration**2
    delta = np.asarray(end)-start
    return np.asarray(start)+delta*blend,delta*velocity,delta*acceleration


class ForwardJumpTrajectory:
    def __init__(self,robot,config):
        config.validate()
        self.config,self.home= config,robot.home.copy()
        self.interrupted_at=None
        def pose(thigh,calf):
            value=np.tile([0.,thigh,calf],4)
            if np.any(value<robot.joint_limits[:,0]+.01) or np.any(value>robot.joint_limits[:,1]-.01):
                raise ValueError('Jump pose outside model joint ranges')
            return value
        crouch=pose(config.crouch_thigh,config.crouch_calf)
        extension=pose(-config.extension_calf/2+config.forward_lean,config.extension_calf)
        flight=pose(-config.flight_calf/2+config.flight_lean,config.flight_calf)
        absorption=pose(config.absorption_thigh,config.absorption_calf)
        self.poses=[self.home,crouch,extension,flight,absorption,self.home]
        self.durations=[config.preparation,config.thrust,config.flight,config.absorption,config.recovery]
        self.phase_names=['prepare','thrust','flight','absorb','recover']

    def interrupt(self,elapsed):
        if self.interrupted_at is None:
            self.interrupted_at=float(elapsed)

    def gain_blend(self,elapsed):
        def smooth(value):
            s=float(np.clip(value,0.,1.))
            return 10*s**3-15*s**4+6*s**5
        start=self.config.preparation+self.config.thrust
        rising=smooth((elapsed-start)/self.config.flight)
        recover=start+self.config.flight+self.config.absorption
        falling=1-smooth((elapsed-recover)/self.config.recovery)
        return rising*falling

    def sample(self,elapsed):
        if not np.isfinite(elapsed) or elapsed<0:
            raise ValueError('Jump time must be finite and nonnegative')
        if self.interrupted_at is not None and self.interrupted_at<self.config.preparation:
            if elapsed<self.config.preparation:
                q,v,a=quintic_segment(self.home,self.poses[1],elapsed,self.config.preparation)
                return q,v,a,'abort_prepare'
            q,v,a=quintic_segment(self.poses[1],self.home,elapsed-self.config.preparation,self.config.recovery)
            return q,v,a,'abort_recover'
        remaining=float(elapsed)
        for i,duration in enumerate(self.durations):
            if remaining<duration:
                q,v,a=quintic_segment(self.poses[i],self.poses[i+1],remaining,duration)
                return q,v,a,self.phase_names[i]
            remaining-=duration
        return self.home.copy(),np.zeros(12),np.zeros(12),'settle'
