"""Bounded development-only motor adaptation grid; retain every failure."""
from dataclasses import asdict
import argparse,itertools,json
from pathlib import Path
from experiments.continual_snn.transfer import terrain_cases
from experiments.continual_snn.study import save
from .batch import run
from .contract import AdaptationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    scenes=[c for i,c in enumerate(terrain_cases(20261317)) if i//6 in (1,3,5) and i%6 in (0,3,5)]
    configs=[AdaptationConfig(terrain_mode='preview',terrain_lift=lift,terrain_gain=gain,body_raise=raise_,terrain_filter_rate=rate)
        for lift,gain,raise_,rate in itertools.product((.075,.085),(.8,1.),(0.,.02,.04),(.08,.2))]
    save(out/'protocol.json',dict(cases=[c.to_dict() for c in scenes],configs=[asdict(c) for c in configs],
        split='development_only_not_final',selection='Prefer zero contact/fall/hold failures, then successful traversal count'))
    results=[]
    for i,cfg in enumerate(configs):
        summary=run(out/f'candidate-{i:03d}',scenes,[('rule','rule',None)],adaptation=cfg)
        results.append(dict(index=i,config=asdict(cfg),results=summary))
        save(out/'results.json',results)


if __name__=='__main__':main()
