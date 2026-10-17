"""Bounded local-height gait adaptation on the existing classical IK gait."""
import numpy as np
from controllers.navigation import NavigationExperiment,WaypointController


def support_height(spec,x,y):
    return max([0.]+[h for p in spec.terrain if (h:=p.surface_height(x,y)) is not None])


class TerrainGait(NavigationExperiment):
    def __init__(self,model,spec,config):
        super().__init__(model)
        self.spec,self.adaptation=spec,config
        self.lift=config.terrain_lift
        self.period=config.gait_period;self.stance=config.gait_stance
        self.height_offsets=np.zeros(4)
        self.navigator=WaypointController(max_speed=.28)

    def velocity_targets(self,command,dt):
        if np.linalg.norm(command)>0 and self.adaptation.terrain_speed!=.35:
            command=np.array(command,dtype=float);command[:2]*=self.adaptation.terrain_speed/.35
        if self.adaptation.goal_arrival and np.linalg.norm(command)>0:
            rot=self.data.body('base').xmat.reshape(3,3)
            yaw=float(np.arctan2(rot[1,0],rot[0,0]))
            command=self.navigator.command(self.data.qpos[:2],yaw,self.spec.goal)
            command[0]=max(0.,command[0])
            if self.adaptation.approach_speed:
                length=np.linalg.norm(command[:2])
                command[:2]*=self.adaptation.approach_speed/max(length,1e-12)
                # The run option remains forward; a missed stopping decision
                # cannot silently acquire an unvalidated reverse skill.
                command[0]=max(0.,command[0])
        if self.adaptation.terrain_mode in ('none','lift') and self.adaptation.brake_rate==.25:
            return super().velocity_targets(command,dt)
        rate=np.array([self.adaptation.brake_rate,self.adaptation.brake_rate,.8]) if np.linalg.norm(command)<1e-9 else np.array([.25,.25,.8])
        self.commanded_velocity+=np.clip(command-self.commanded_velocity,-rate*dt,rate*dt)
        moving=np.linalg.norm(self.commanded_velocity)>.005
        self.activity=np.clip(self.activity+(dt/.6 if moving else -dt/.6),0,1)
        self.gait_time+=dt
        result=self.home.copy()
        origin=self.ik.body('base').xpos[:2].copy()
        rot=self.data.body('base').xmat.reshape(3,3)
        yaw=float(np.arctan2(rot[1,0],rot[0,0]));c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s],[s,c]])
        base=self.data.qpos[:2]
        # Smooth local relative support heights. No body placement or force.
        desired=[]
        center_height=support_height(self.spec,*base)
        if self.adaptation.terrain_mode=='preview':
            heights=[support_height(self.spec,*(base+rotation@(info['foot'][:2]-origin))) for info in self.legs.values()]
            center_height=max(center_height,float(np.mean(heights)))
            approaching=max(heights+[support_height(self.spec,*(base+rotation@np.array([.4,side]))) for side in (-.2,.2)])>.002
            if approaching:center_height+=self.adaptation.body_raise
        for leg,info in self.legs.items():
            radius=info['foot'][:2]-origin
            point=base+rotation@radius
            local_height=support_height(self.spec,*point)
            if self.adaptation.terrain_mode=='preview':
                phase=(self.gait_time/self.period+self.phase[leg])%1
                vx,vy,wz=self.commanded_velocity
                velocity=np.array([vx-wz*radius[1],vy+wz*radius[0]])
                if phase>=self.stance:
                    fraction=(phase-self.stance)/(1-self.stance)
                    travel=3*fraction**2-2*fraction**3-.5
                    target_point=base+rotation@(radius+velocity*self.period*self.stance*travel)
                    local_height=max(local_height,support_height(self.spec,*target_point))
            desired.append(np.clip(local_height-center_height,-.09,.09) if self.adaptation.terrain_mode in ('height','preview') else 0.)
        desired=np.asarray(desired)*self.adaptation.terrain_gain
        change=self.adaptation.terrain_filter_rate*dt
        self.height_offsets+=np.clip(desired-self.height_offsets,-change,change)
        for index,(leg,info) in enumerate(self.legs.items()):
            phase=(self.gait_time/self.period+self.phase[leg])%1
            if phase<self.stance:
                fraction=phase/self.stance;travel=.5-fraction;z=0.
            else:
                fraction=(phase-self.stance)/(1-self.stance)
                travel=3*fraction**2-2*fraction**3-.5
                z=self.lift*np.sin(np.pi*fraction)**2
            radius=info['foot'][:2]-origin
            vx,vy,wz=self.commanded_velocity
            velocity=np.array([vx-wz*radius[1],vy+wz*radius[0]])
            displacement=velocity*self.period*self.stance*travel
            offset=np.r_[self.activity*displacement,self.height_offsets[index]+self.activity*z]
            result[info['acts']]=self.solve_leg(leg,info['foot']+offset)
        return result
