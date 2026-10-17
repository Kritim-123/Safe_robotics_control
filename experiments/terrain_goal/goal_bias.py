"""Fixed goal-vector bias diagnostic; models and physical feedback stay frozen."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from .assessment import primary_config
from .goal_cases import goal_cases
from .loading_v2 import load_any
from .batch import run


class BiasedGoalFeatures:
    """Observation adapter only. Published checkpoint is the unchanged predictor.

    Bias is applied once in augment_features, then the actual predictor input is
    recorded by the executor. Saving the base predictor therefore permits exact
    prediction replay from those recorded inputs without a hidden second shift.
    Other observation channels and request masks are deliberately unaffected.
    """
    def __init__(self,model,bias):
        self.model,self.bias=model,float(bias)
        self.encoder,self.checksum,self.name=model.encoder,model.checksum,model.name

    def augment_features(self,features,*,task,geometry,observed):
        f=self.model.augment_features(features,task=task,geometry=geometry,observed=observed)
        if task=='progress':f['goal_forward']=float(f['goal_forward']+self.bias)
        return f

    def predict(self,features,now=None):return self.model.predict(features,now=now)
    def reset(self):self.model.reset()
    def save(self,path):self.model.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,default=1792269720);a=p.parse_args()
    scenes=goal_cases(15200,20,'goal_bias_diagnostic',flat_only=True)
    cfg=primary_config()
    for value in (-.08,-.04,.04,.08):
        name=('minus' if value<0 else 'plus')+str(round(abs(value)*100))+'cm'
        out=a.output/name
        models=[('v1','continuous',BiasedGoalFeatures(load_any(ROOT/'docs/agents/terrain_goal_v1/primary.npz'),value)),
                ('v2','continuous',BiasedGoalFeatures(load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz'),value))]
        run(out,scenes,models,adaptation=cfg,deadline=a.deadline)
        save(out/'observation_condition.json',dict(goal_forward_additive_bias_m=value,
            scope='Fixed longitudinal goal-vector feature error only; lateral component, clearance masks, physical guards and low-level pose feedback unchanged',
            training=False,retuning=False,independent_geometries=20,
            repeat_note='Same 20 flat seed-15200 scenes repeated for four biases and both frozen models',
            checkpoint='Unchanged predictor; injected features are recorded before inference and replayed exactly'))


if __name__=='__main__':main()
