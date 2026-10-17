"""Fit goal stopping from distance populations; never use final scene data."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.continual_snn.study import save
from experiments.continual_snn.data import save_records
from experiments.continual_snn.fit import metrics
from .fit import records,historical
from .encoder import TaskEncoder
from .models import Selector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs='+',type=Path,required=True)
    p.add_argument('--validation',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kinds',nargs='+',default=['factor32','task96','goal64','goal_lattice64'])
    p.add_argument('--dagger',nargs='*',type=Path,default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    train=[r for r in historical('training') if r.get('source')!='static']+sum((records(f,'training') for f in a.train),[])
    train+=sum(([r for r in records(f,'training') if r['source']=='learner_visited_state_teacher_label'] for f in a.dagger),[])
    val=[r for r in historical('validation') if r.get('source')!='static']+sum((records(f,'validation') for f in a.validation),[])
    excluded=dict(training_terrain_ineligible=sum(not r.get('terrain_eligible',True) for r in train),
        validation_terrain_ineligible=sum(not r.get('terrain_eligible',True) for r in val),
        reason='Goal decoder is conditional on motor eligibility; capability refusal remains a supervisor decision and failed task in physical evaluation')
    train=[r for r in train if r.get('terrain_eligible',True)];val=[r for r in val if r.get('terrain_eligible',True)]
    save(a.output/'excluded_records.json',excluded)
    for r in train+val:r['features'].setdefault('goal_mode',0.)
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('Goal group leakage')
    save_records(a.output/'training_records.jsonl',train);save_records(a.output/'validation_records.jsonl',val)
    ranking=[]
    for kind in a.kinds:
        encoder=TaskEncoder(kind);xt=encoder.embed_many([r['features'] for r in train]);xv=encoder.embed_many([r['features'] for r in val])
        goals=[i for i,r in enumerate(val) if r['features'].get('goal_mode')]
        for alpha in (.0001,.001,.01,.1):
            model=Selector(encoder,'ridge',alpha=alpha)
            model.inner.fit_embeddings(xt,[r['label'] for r in train],[r['group'] for r in train]);model._checksum()
            filename=f'{kind}_ridge_a{alpha:g}.npz';model.save(a.output/filename)
            pred=np.argmax(model.inner.predict_embeddings(xv),axis=1)
            score=metrics([r['label'] for r in val],pred,[r['group'] for r in val])
            goal_score=metrics([val[i]['label'] for i in goals],pred[goals],[val[i]['group'] for i in goals])
            ranking.append(dict(name=filename,all_metrics=score,goal_metrics=goal_score))
            print(filename,score['group_mean_accuracy'],goal_score['group_mean_accuracy'],flush=True)
        proto=Selector(encoder,'prototype').fit(train);proto.save(a.output/f'{kind}_prototype.npz')
    save(a.output/'ranking.json',ranking)


if __name__=='__main__':main()
