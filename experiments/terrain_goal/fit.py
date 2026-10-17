"""Fit task-aware and unchanged-encoder candidates on declared training only."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.loading import load_selector
from experiments.continual_snn.prototypes import PrototypeMemory
from experiments.continual_snn.readout import SpikeReadout
from experiments.continual_snn.fit import metrics
from experiments.continual_snn.study import save
from experiments.continual_snn.data import save_records
from .models import Selector
from .encoder import TaskEncoder,add_task


def records(folder,split):
    rows=[]
    for path in sorted(Path(folder).glob('*/*/decisions.jsonl')):
        summary=json.loads((path.parent/'summary.json').read_text())
        for line in path.read_text().splitlines():
            item=json.loads(line)
            if item['teacher_request'] is None:continue
            features=dict(item['features'],**{n:item['geometry'][n] for n in ('predicted_run_clearance','predicted_jump_clearance')})
            features=add_task(features,item['task'])
            features['goal_mode']=float(summary.get('adaptation_config',{}).get('goal_arrival',False))
            features['run_active']=float(item['current']=='run')
            features['braking_active']=float(item['mode']=='braking')
            rows.append(dict(group=f'{split}/{summary["seed"]}/{summary["case"]}',split=split,
                features=features,label=item['teacher_request'],time=item['time'],
                source='terrain_teacher' if summary['mode']=='rule' else 'learner_visited_state_teacher_label',
                behavior_model_checksum=summary.get('model_checksum'),arm=path.parent.name,
                source_episode_fulfilled=summary['task_fulfilled'],
                terrain_eligible=item['geometry'].get('terrain_eligible',True),
                run_eligible=item['request_mask']['run'],
                label_source='causal_teacher_request_not_counterfactual_outcome'))
    return rows


def historical(split):
    with gzip.open(ROOT/f'docs/agents/continual_snn_v2/{split}_records.jsonl.gz','rt') as stream:
        return [dict(row,features=add_task(row['features'])) for line in stream if (row:=json.loads(line))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,nargs='+',required=True)
    parser.add_argument('--validation',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--teacher-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    train=historical('training')+sum((records(f,'training') for f in args.train),[])
    val=historical('validation')+sum((records(f,'validation') for f in args.validation),[])
    if args.teacher_only:
        train=[r for r in train if r.get('source')!='static']
        val=[r for r in val if r.get('source')!='static']
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('Split leakage')
    save_records(out/'training_records.jsonl',train);save_records(out/'validation_records.jsonl',val)
    ranking=[]
    encoders=[('unchanged80',load_selector(ROOT/'docs/agents/continual_snn_v2/primary.npz').encoder),
              ('task96',TaskEncoder('task96')),('factor32',TaskEncoder('factor32'))]
    for name,encoder in encoders:
        xt=encoder.embed_many([r['features'] for r in train]);xv=encoder.embed_many([r['features'] for r in val])
        for alpha in (.0001,.001,.01,.1):
            model=SpikeReadout(encoder,alpha=alpha).fit_embeddings(xt,[r['label'] for r in train],[r['group'] for r in train])
            score=metrics([r['label'] for r in val],np.argmax(model.predict_embeddings(xv),axis=1),[r['group'] for r in val])
            terrain=[i for i,r in enumerate(val) if r.get('source')=='terrain_teacher']
            terrain_score=metrics([val[i]['label'] for i in terrain],np.argmax(model.predict_embeddings(xv[terrain]),axis=1),[val[i]['group'] for i in terrain])
            filename=f'{name}_ridge_a{alpha:g}.npz'
            if name=='unchanged80':model.save(out/filename)
            else:
                wrapped=Selector(encoder,'ridge',alpha=alpha);wrapped.inner=model;wrapped._checksum();wrapped.save(out/filename)
            ranking.append(dict(name=filename,metrics=score,terrain_metrics=terrain_score))
            print(name,alpha,score['balanced_accuracy'],terrain_score['group_mean_accuracy'],flush=True)
        if name=='unchanged80':model=PrototypeMemory(encoder,neighbors=1,max_distance=12.).fit(train)
        else:model=Selector(encoder,'prototype',max_distance=12.).fit(train)
        model.save(out/f'{name}_prototype.npz')
    save(out/'ranking.json',ranking)


if __name__=='__main__':main()
