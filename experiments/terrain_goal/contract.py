"""Explicit opt-in goal arrival and terrain contracts; legacy scoring is intact."""
from dataclasses import dataclass,asdict
import numpy as np


@dataclass(frozen=True)
class AdaptationConfig:
    goal_arrival:bool=False
    arrival_tolerance:float=.12
    stop_radius:float=.08
    resume_radius:float=.16
    terrain_mode:str='none'
    terrain_lift:float=.08
    terrain_gain:float=1.
    terrain_filter_rate:float=.12
    geometry_mode:str='circles'
    body_raise:float=0.
    motor_backend:str='classical'
    brake_rate:float=.25
    goal_decision_source:str='supervised_request'
    gait_period:float=.6
    gait_stance:float=.6
    terrain_speed:float=.35
    maximum_up_step:float=0.
    arrival_max_delay:float=6.
    approach_speed:float=0.
    selector_position_noise:float=0.
    selector_yaw_noise:float=0.

    def validate(self):
        numeric=[v for k,v in asdict(self).items() if k not in ('goal_arrival','terrain_mode','geometry_mode','motor_backend','goal_decision_source')]
        if (type(self.goal_arrival) is not bool or not np.isfinite(numeric).all()
            or not 0<self.stop_radius<self.arrival_tolerance<self.resume_radius
            or self.terrain_mode not in ('none','lift','height','preview')
            or self.geometry_mode not in ('circles','oriented_boxes')
            or self.motor_backend not in ('classical','himloco')
            or self.goal_decision_source not in ('supervised_request','neural')
            or not .04<=self.terrain_lift<=.12 or not 0<=self.terrain_gain<=1.5
            or not .01<=self.terrain_filter_rate<=.4):
            raise ValueError('Invalid opt-in terrain/goal contract')
        if not 0<=self.body_raise<=.04:raise ValueError('Body raise outside tested search envelope')
        if not .25<=self.brake_rate<=1.:raise ValueError('Invalid bounded braking rate')
        if not .5<=self.gait_period<=1. or not .5<=self.gait_stance<=.75 or not .12<=self.terrain_speed<=.5:
            raise ValueError('Invalid terrain gait envelope')
        if not 0<=self.maximum_up_step<=.1:raise ValueError('Invalid terrain eligibility envelope')
        if not 1.<=self.arrival_max_delay<=10.:raise ValueError('Invalid arrival stopping delay')
        if self.approach_speed!=0. and not .12<=self.approach_speed<=.5:raise ValueError('Invalid approach speed')
        if not 0<=self.selector_position_noise<=.15 or not 0<=self.selector_yaw_noise<=.15:
            raise ValueError('Invalid selector observation noise')


class GoalTracker:
    def __init__(self,config):
        config.validate();self.config=config;self.stopping=False

    def request(self,task,observed,goal):
        if not self.config.goal_arrival or task!='progress':return task
        distance=float(np.linalg.norm(np.asarray(goal)-observed['position']))
        if distance<=self.config.stop_radius:self.stopping=True
        elif distance>self.config.resume_radius:self.stopping=False
        return 'hold' if self.stopping else task
