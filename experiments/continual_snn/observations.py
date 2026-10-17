"""Current observed state only; future scripted obstacle schedules never enter."""
from collections import deque
from dataclasses import asdict,dataclass
import numpy as np

from skills.avoidance import clearance,path_clearance
from experiments.spike_pattern_skills.observations import FEATURES
from experiments.spike_pattern_skills.dense_terrain_encoding import dense_surface_features


@dataclass(frozen=True)
class SensorConfig:
    position_noise:float=0.
    velocity_noise:float=0.
    delay:float=0.
    seed:int=0

    def validate(self):
        if (not np.isfinite([self.position_noise,self.velocity_noise,self.delay]).all()
                or min(self.position_noise,self.velocity_noise,self.delay)<0 or type(self.seed) is not int):
            raise ValueError('Invalid sensor configuration')


class SensorStream:
    def __init__(self,config=None):
        self.config=config or SensorConfig();self.config.validate()
        self.rng=np.random.default_rng(self.config.seed)
        self.queue=deque();self.last=None

    def observe(self,now,obstacles):
        if not np.isfinite(now) or now<0 or (self.last is not None and now<=self.last):
            raise ValueError('Sensor time must increase strictly')
        self.last=float(now)
        sample=[]
        for item in obstacles:
            sample.append(dict(item,
                center=(np.asarray(item['center'])+self.rng.normal(0,self.config.position_noise,2)).tolist(),
                velocity=(np.asarray(item['velocity'])+self.rng.normal(0,self.config.velocity_noise,2)).tolist()))
        self.queue.append((float(now),sample))
        cutoff=now-self.config.delay
        while len(self.queue)>1 and self.queue[1][0]<=cutoff+1e-9:
            self.queue.popleft()
        return self.queue[0]


def snapshot(robot,arena):
    d=robot.data;rot=d.body('base').xmat.reshape(3,3)
    obstacles=[]
    for spec,(center,radius,velocity) in zip(arena.spec.obstacles,arena.sensed_obstacles(d)):
        obstacles.append(dict(name=spec.name,center=center.tolist(),radius=float(radius),
            velocity=velocity.tolist(),height=spec.height,
            width=2*(spec.half_extents[1] if spec.shape=='box' else spec.radius)))
    foot_ids={v['geom'] for v in robot.legs.values()}
    contacts={g for c in d.contact for g in (int(c.geom1),int(c.geom2)) if g in foot_ids}
    stable=(robot.activity==0 and len(contacts)==4 and np.linalg.norm(d.qvel[:3])<.04
            and np.linalg.norm(d.qvel[3:6])<.2 and rot[2,2]>np.cos(np.deg2rad(12)) and d.qpos[2]>.23)
    return dict(time=float(d.time),position=d.qpos[:2].tolist(),height=float(d.qpos[2]),
        yaw=float(np.arctan2(rot[1,0],rot[0,0])),speed=float(np.linalg.norm(d.qvel[:2])),
        linear_speed=float(np.linalg.norm(d.qvel[:3])),angular_speed=float(np.linalg.norm(d.qvel[3:6])),
        upright=float(rot[2,2]),stable=bool(stable),foot_contacts=len(contacts),obstacles=obstacles)


def feature_packet(spec,observed,task):
    """Same named 22 inputs as the saved model, but honors supplied noisy data."""
    pos=np.asarray(observed['position']);yaw=observed['yaw']
    forward=np.array([np.cos(yaw),np.sin(yaw)]);left=np.array([-forward[1],forward[0]])
    obstacles=observed['obstacles']
    tuples=[(np.asarray(o['center']),o['radius'],np.asarray(o['velocity'])) for o in obstacles]
    gap=clearance(pos,tuples,spec.bounds)
    if obstacles:
        item=min(obstacles,key=lambda o:np.linalg.norm(np.asarray(o['center'])-pos))
        rel=np.asarray(item['center'])-pos;velocity=np.asarray(item['velocity'])
        front,side,vf,vl=rel@forward,rel@left,velocity@forward,velocity@left
        radius,height,width=item['radius'],item['height'],item['width']
    else:
        front=side=vf=vl=radius=height=width=0.
    goal=np.zeros(2) if task=='hold' else np.asarray(spec.goal)-pos
    values=np.asarray([bool(obstacles),front,side,vf,vl,radius,height,width,gap,
        goal@forward,goal@left,observed['speed'],observed['height'],task=='jump'],dtype=np.float32)
    features={name:float(v) for name,v in zip(FEATURES,values)}
    if spec.terrain:
        features.update(dense_surface_features(spec,observed))
    else:
        from experiments.spike_pattern_skills.terrain_encoding import TERRAIN_FEATURES
        features.update({name:0. for name in TERRAIN_FEATURES})
    return features


def request_masks(spec,observed,*,run_speed=.35,predict_horizon=1.6,jump_distance=.5,jump_horizon=3.):
    """Current measured velocity extrapolation, never future scripted schedules."""
    pos=np.asarray(observed['position']);yaw=observed['yaw']
    forward=np.array([np.cos(yaw),np.sin(yaw)])
    tuples=[(np.asarray(o['center']),o['radius'],np.asarray(o['velocity'])) for o in observed['obstacles']]
    gap=clearance(pos,tuples,spec.bounds)
    run_gap=jump_gap=float('inf')
    for dt in np.linspace(0,predict_horizon,9):
        predicted=[(c+v*dt,r,v) for c,r,v in tuples]
        endpoint=pos+forward*run_speed*dt
        run_gap=min(run_gap,clearance(endpoint,predicted,spec.bounds))
    for dt in np.linspace(0,jump_horizon,17):
        predicted=[(c+v*dt,r,v) for c,r,v in tuples]
        # Reserve longer than the nominal hop/recovery, using only current
        # velocities. It remains a finite-horizon approximation, not a proof.
        jump_gap=min(jump_gap,path_clearance(pos,[pos+forward*jump_distance],predicted,spec.bounds))
    return dict(wait=bool(gap>.08),run=bool(run_gap>.15),jump=bool(jump_gap>.35)),dict(
        current_clearance=float(gap),predicted_run_clearance=float(run_gap),
        predicted_jump_clearance=float(jump_gap),prediction_horizon=predict_horizon,
        jump_prediction_horizon=jump_horizon)


def teacher(task,masks):
    desired='jump' if task=='jump' else 'run' if task=='progress' else 'wait'
    return desired if masks.get(desired,False) else 'wait' if masks.get('wait',False) else None
