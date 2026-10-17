"""Experimental existing HIMLoco run/zero-command pair in persistent physics.

No classical/learned/jump cross-controller handoff is asserted. The source
policy history persists through run, braking and hold; one PID writes torques.
"""
import numpy as np
from controllers.navigation import NavigationExperiment,WaypointController
from skills.pretrained_velocity import PretrainedVelocityPolicy


class LearnedVelocityRobot(NavigationExperiment):
    def __init__(self,model,spec,config):
        super().__init__(model)
        self.spec,self.adaptation=spec,config
        self.adapter=PretrainedVelocityPolicy(model,'himloco')
        c=self.adapter.config
        self.home[self.adapter.actuators]=self.adapter.default
        self.standing_kp,self.standing_kd=np.empty(12),np.empty(12)
        self.kp,self.kd=np.empty(12),np.empty(12)
        self.standing_kp[self.adapter.actuators]=c['fixed_kp']
        self.standing_kd[self.adapter.actuators]=c['fixed_kd']
        self.kp[self.adapter.actuators]=c['rl_kp'];self.kd[self.adapter.actuators]=c['rl_kd']
        self.ki=np.zeros(12)
        source_limits=np.empty(12);source_limits[self.adapter.actuators]=c['torque_limits']
        self.limits=np.minimum(self.limits,source_limits)
        self.current_target=self.home.copy();self.next_policy_tick=2.
        self.policy_inputs=[];self.policy_actions=[];self.policy_times=[]
        self.navigator=WaypointController(max_speed=.35)

    def velocity_targets(self,command,dt):
        command=np.asarray(command,dtype=float)
        if self.adaptation.goal_arrival and np.linalg.norm(command)>0:
            rot=self.data.body('base').xmat.reshape(3,3)
            yaw=float(np.arctan2(rot[1,0],rot[0,0]))
            command=self.navigator.command(self.data.qpos[:2],yaw,self.spec.goal)
            command[0]=max(0.,command[0])
        rate=np.array([max(.5,self.adaptation.brake_rate),max(.5,self.adaptation.brake_rate),.8])
        self.commanded_velocity+=np.clip(command-self.commanded_velocity,-rate*dt,rate*dt)
        self.activity=float(np.linalg.norm(self.commanded_velocity)>.005)
        self.gait_time+=dt
        if self.data.time>=self.next_policy_tick-.001:
            self.current_target=self.adapter.step(self.data,self.commanded_velocity)
            self.policy_inputs.append(self.adapter.last_input.copy())
            self.policy_actions.append(self.adapter.last_action.copy())
            self.policy_times.append(float(self.data.time))
            self.next_policy_tick+=.02
        return self.current_target.copy()
