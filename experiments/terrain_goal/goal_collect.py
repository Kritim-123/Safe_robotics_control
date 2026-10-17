"""Teacher goal-arrival packets retain mission intent and actual outcomes."""
import argparse
from pathlib import Path
from .batch import run
from .contract import AdaptationConfig
from .encoder import TaskEncoder,add_task
from .goal_cases import goal_cases


class PacketObserver:
    def __init__(self):
        self.encoder=TaskEncoder('goal64');self.name='goal_packet_observer'
        self.checksum=self.encoder.checksum
    def reset(self):pass
    def augment_features(self,features,*,task,geometry,observed):return add_task(features,task)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=60);p.add_argument('--split',default='goal_training')
    p.add_argument('--backend',choices=('classical','himloco'),default='classical')
    p.add_argument('--flat-only',action='store_true')
    p.add_argument('--maximum-up-step',type=float,default=0.)
    p.add_argument('--profile',choices=('low','high'),default='low')
    p.add_argument('--stop-radius',type=float,default=.08)
    p.add_argument('--approach-speed',type=float,default=0.)
    a=p.parse_args();scenes=goal_cases(a.seed,a.count,a.split,flat_only=a.flat_only)
    cfg=AdaptationConfig(goal_arrival=True,goal_decision_source='neural',terrain_mode='lift',terrain_lift=.12 if a.profile=='high' else .08,
        geometry_mode='oriented_boxes',motor_backend=a.backend,brake_rate=.8 if a.backend=='himloco' else .5,
        maximum_up_step=a.maximum_up_step,gait_period=.8 if a.profile=='high' else .6,gait_stance=.7 if a.profile=='high' else .6,
        stop_radius=a.stop_radius,approach_speed=a.approach_speed)
    run(a.output,scenes,[('rule','rule',PacketObserver())],adaptation=cfg)


if __name__=='__main__':main()
