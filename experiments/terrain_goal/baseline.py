"""Reproduce unchanged terrain failures and collect declared teacher coverage."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.loading import load_selector
from experiments.continual_snn.study import run_batch,save
from experiments.continual_snn.transfer import terrain_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261117)
    parser.add_argument('--phase',choices=('baseline','teacher'),default='baseline')
    args=parser.parse_args()
    scenes=terrain_cases(args.seed)
    if args.phase=='teacher':
        scenes=[replace(c,split='terrain_training') for c in scenes]
        arms=[('rule','rule',None)]
    else:
        arms=[('v2','continuous',load_selector(ROOT/'docs/agents/continual_snn_v2/primary.npz')),
              ('rule','rule',None)]
    run_batch(args.output,scenes,arms)
    rows=[json.loads(s) for s in (args.output/'results.jsonl').read_text().splitlines()]
    breakdown=[]
    for arm,_,_ in arms:
        for stage in range(10):
            group=[r['summary'] for r in rows if r['arm']==arm and f'terrain-{stage:02d}-' in r['summary']['case']]
            breakdown.append(dict(arm=arm,stage=stage,trials=len(group),
                tasks=sum(s['task_fulfilled'] for s in group),progress=[s['progress'] for s in group],
                reasons=dict(Counter(s['reason'] for s in group))))
    save(args.output/'stage_breakdown.json',breakdown)


if __name__=='__main__':main()
