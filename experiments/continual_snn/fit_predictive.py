"""Add causal geometry inputs to the SAME frozen grouped training records."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .models import ROOT,BUNDLE
from .observations import request_masks
from .predictive import PredictivePacketEncoder,PREDICTIVE_FEATURES
from .readout import SpikeReadout,LABELS
from .prototypes import PrototypeMemory
from .data import save_records
from .fit import metrics
from .study import save


def geometry(spec,observed):
    _,detail=request_masks(SimpleNamespace(bounds=spec['bounds']),observed)
    return {name:detail[name] for name in PREDICTIVE_FEATURES}


def augment(original,train_folder,validation_folder):
    lookup={}
    path=ROOT/'output/spike_pattern_skills/expanded-associations-v1/reused_training.jsonl'
    for line in path.read_text().splitlines():
        row=json.loads(line);meta=row['observation']['metadata']
        lookup[('original/'+row['example']['context_id'],None)]=geometry(meta['scenario'],meta['robot_observation'])
    for line in (BUNDLE/'reencoded_labels.jsonl').read_text().splitlines():
        row=json.loads(line);meta=row['dense_observation']['metadata']
        lookup[(row['context_id'],None)]=geometry(meta['scenario'],meta['robot_observation'])
    for folder in (train_folder,validation_folder):
        for path in sorted(Path(folder).glob('*/*/decisions.jsonl')):
            spec=json.loads((path.parent/'scenario.json').read_text())['scenario']
            for line in path.read_text().splitlines():
                row=json.loads(line)
                lookup[(spec['name'],row['time'])]=geometry(spec,row['observation'])
    result=[]
    for row in original:
        extra=lookup[(row['group'],row.get('time'))]
        result.append(dict(row,features=dict(row['features'],**extra)))
    return result


def compare(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):raise ValueError('Output outside project output')
    output.mkdir(parents=True,exist_ok=False)
    original=ROOT/'output/continual_snn/readout-comparison-v1'
    training=[json.loads(s) for s in (original/'training_records.jsonl').read_text().splitlines()]
    validation=[json.loads(s) for s in (original/'validation_records.jsonl').read_text().splitlines()]
    combined=augment(training+validation,ROOT/'output/continual_snn/teacher-train-v1',
                     ROOT/'output/continual_snn/teacher-validation-v1')
    train,val=combined[:len(training)],combined[len(training):]
    save_records(output/'training_records.jsonl',train);save_records(output/'validation_records.jsonl',val)
    encoder=PredictivePacketEncoder()
    yt=[r['label'] for r in train];yv=[r['label'] for r in val]
    gt=[r['group'] for r in train];gv=[r['group'] for r in val]
    rows=[]
    for kind in ('spike','analog'):
        model=SpikeReadout(encoder,analog=kind=='analog')
        xt=np.asarray([model._embedding(r['features']) for r in train])
        xv=np.asarray([model._embedding(r['features']) for r in val])
        best=None
        for alpha in (.0001,.001,.01,.1,1.):
            model=SpikeReadout(encoder,alpha=alpha,analog=kind=='analog').fit_embeddings(xt,yt,gt)
            metric=metrics(yv,np.argmax(model.predict_embeddings(xv),axis=1),gv)
            row=dict(kind=kind,alpha=alpha,metrics=metric,checksum=model.checksum);rows.append(row)
            print(kind,alpha,metric['balanced_accuracy'],flush=True)
            if best is None or metric['balanced_accuracy']>best[0]['metrics']['balanced_accuracy']:best=(row,model)
        best[1].save(output/'models'/(kind+'.npz'))
    model=PrototypeMemory(encoder).fit(train)
    queries=[encoder.encode(r['features'])['spike_counts'] for r in val]
    for k in (1,3):
        for radius in (8.,10.,12.):
            model.neighbors=k;model.max_distance=radius;model._update_checksum()
            predictions=[model.decode_counts(q)['proposed_skill'] for q in queries]
            # Abstentions are scored as wait, the checked supervisory fallback.
            metric=metrics(yv,[LABELS.index(p or 'wait') for p in predictions],gv)
            row=dict(kind='prototype',neighbors=k,radius=radius,metrics=metric,
                     coverage=float(np.mean([p is not None for p in predictions])),checksum=model.checksum)
            rows.append(row);model.save(output/'models'/f'prototype-k{k}-r{radius:g}.npz')
            print('prototype',k,radius,metric['balanced_accuracy'],row['coverage'],flush=True)
    save(output/'ranking.json',sorted(rows,key=lambda r:r['metrics']['balanced_accuracy'],reverse=True))
    save(output/'manifest.json',dict(same_labels_groups_and_rows=True,training_rows=len(train),
        validation_rows=len(val),input_contract=encoder.contract,training_groups=len(set(gt)),
        validation_groups=len(set(gv)),groups_disjoint=not bool(set(gt)&set(gv)),
        geometry_source='current recorded obstacle positions/velocities and bounds; no future script',
        stress_development_not_used_for_fitting=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    compare(parser.parse_args().output)
