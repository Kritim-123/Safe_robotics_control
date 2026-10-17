"""Hard-case development grid: higher clearance, longer stance and slower speed."""
from dataclasses import asdict
import argparse,itertools,json
from pathlib import Path
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.study import save
from .batch import run
from .contract import AdaptationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    scenes=[c for i,c in enumerate(terrain_cases(20261317)) if (i//6,i%6) in ((3,5),(5,3),(5,5))]
    configs=[AdaptationConfig(terrain_mode=mode,terrain_lift=lift,terrain_gain=.8,body_raise=raise_,
        terrain_filter_rate=.2,brake_rate=.5,gait_period=period,gait_stance=stance,terrain_speed=speed)
        for mode,lift,raise_,period,stance,speed in itertools.product(('lift','preview'),(.10,.12),(0.,.03),(.6,.8),(.6,.7),(.25,.35))]
    save(a.output/'protocol.json',dict(split='hard_development_only',cases=[c.to_dict() for c in scenes],configs=[asdict(c) for c in configs]))
    results=[]
    for i,c in enumerate(configs):
        s=run(a.output/f'candidate-{i:03d}',scenes,[('rule','rule',None)],adaptation=c)
        results.append(dict(index=i,config=asdict(c),results=s));save(a.output/'results.json',results)


if __name__=='__main__':main()
