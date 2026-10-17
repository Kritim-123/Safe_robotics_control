"""Grouped offline comparison of frozen spiking representations and readouts."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import static_records,stream_records,save_records
from .encoding import PacketConfig,PacketEncoder
from .models import ROOT,FrozenBankModel
from .readout import SpikeReadout,labels_to_int
from .study import save


def metrics(labels,predictions,groups):
    y=labels_to_int(labels);predictions=np.asarray(predictions)
    counts=Counter(groups)
    weights=np.asarray([1/counts[g] for g in groups])
    matrix=np.zeros((3,3))
    for truth,predicted,weight in zip(y,predictions,weights):
        matrix[truth,predicted]+=weight
    recalls=np.diag(matrix)/np.maximum(matrix.sum(axis=1),1e-12)
    return dict(rows=len(labels),groups=len(set(groups)),frame_accuracy=float(np.mean(y==predictions)),
        group_mean_accuracy=float(np.average(y==predictions,weights=weights)),
        balanced_accuracy=float(np.mean(recalls)),class_recall=recalls.tolist(),
        group_weighted_confusion=matrix.tolist())


def compare(output,train_folder,validation_folder,*,small=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):raise ValueError('Output outside project output')
    output.mkdir(parents=True,exist_ok=False)
    historical=static_records()
    train=[r for r in historical if r['split']=='train']+stream_records(train_folder,split='train')
    validation=[r for r in historical if r['split']=='validation']+stream_records(validation_folder,split='validation')
    if {r['group'] for r in train}&{r['group'] for r in validation}:raise ValueError('Group leakage')
    if sum(r['source']=='stream' for r in train)==0 or sum(r['source']=='stream' for r in validation)==0:
        raise ValueError('Missing stream records')
    save_records(output/'training_records.jsonl',train)
    save_records(output/'validation_records.jsonl',validation)
    configurations=[PacketConfig(neurons=n) for n in (16,64,128,256)]
    configurations += [PacketConfig(neurons=n,representation='raster') for n in (16,32)]
    configurations += [PacketConfig(representation='filtered',filter_tau=tau) for tau in (1.,2.,4.)]
    configurations += [PacketConfig(neurons=n,neuron='alif',adaptation_strength=gain) for n in (16,64) for gain in (.1,.3)]
    configurations += [PacketConfig(neurons=n,neuron='synaptic',synaptic_decay=decay) for n in (16,64) for decay in (.5,.8)]
    if small:configurations=configurations[:4]
    save(output/'declared_configurations.json',[asdict(c) for c in configurations])
    x_train=[r['features'] for r in train];x_val=[r['features'] for r in validation]
    y_train=[r['label'] for r in train];y_val=[r['label'] for r in validation]
    g_train=[r['group'] for r in train];g_val=[r['group'] for r in validation]
    selection=[]
    # Run the old bank on the same causal validation packets as a diagnostic.
    baseline=FrozenBankModel();old=[]
    for features in x_val:
        proposed=baseline.predict(features)['proposed_skill']
        old.append(('wait','run','jump').index(proposed.removeprefix('dynamic/')) if proposed else 0)
    save(output/'old_bank_validation.json',metrics(y_val,old,g_val))
    with (output/'candidates.jsonl').open('x') as stream:
        for index,cfg in enumerate(configurations+[None]):
            analog=cfg is None
            encoder=PacketEncoder(cfg or PacketConfig())
            if analog:
                xt=np.asarray([[f[n] for n in encoder.feature_names] for f in x_train])/encoder.base.scales
                xv=np.asarray([[f[n] for n in encoder.feature_names] for f in x_val])/encoder.base.scales
            else:
                before=time.perf_counter();xt=encoder.embed_many(x_train);xv=encoder.embed_many(x_val)
                print('embedded',index,asdict(cfg),'seconds',round(time.perf_counter()-before,2),flush=True)
            candidate_rows=[];best=None
            for alpha in (.0001,.001,.01,.1,1.):
                before=time.perf_counter()
                model=SpikeReadout(encoder,alpha=alpha,analog=analog)
                model.fit_embeddings(xt,y_train,g_train)
                scores=model.predict_embeddings(xv);pred=np.argmax(scores,axis=1)
                metric=metrics(y_val,pred,g_val)
                row=dict(index=index,config=asdict(cfg) if cfg else None,analog=analog,alpha=alpha,
                    model_checksum=model.checksum,metrics=metric,fit_seconds=time.perf_counter()-before)
                stream.write(json.dumps(row)+'\n');stream.flush();candidate_rows.append(row)
                if best is None or (metric['balanced_accuracy'],metric['group_mean_accuracy'])>(best[0]['metrics']['balanced_accuracy'],best[0]['metrics']['group_mean_accuracy']):
                    best=(row,model)
                print(index,model.name,round(metric['balanced_accuracy'],4),round(metric['group_mean_accuracy'],4),flush=True)
            row,model=best;path=output/'models'/f'candidate-{index:02d}.npz'
            model.save(path);row=dict(row,checkpoint=str(path.relative_to(output)))
            selection.append(row);save(output/'ranking.json',sorted(selection,key=lambda r:(
                r['metrics']['balanced_accuracy'],r['metrics']['group_mean_accuracy']),reverse=True))
    report=dict(training_rows=len(train),training_groups=len(set(g_train)),
        validation_rows=len(validation),validation_groups=len(set(g_val)),
        neural_candidates=len(configurations),ridge_fits=5*(len(configurations)+1),
        train_validation_groups_disjoint=True,selection_metric='group-weighted balanced accuracy, then group mean accuracy',
        labels='historical measured static outcomes plus separate causal geometry/task teacher streams',
        trained_encoder_weights=False,baseline_encoder_preserved=baseline.encoder.checksum,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.jsonl')})
    save(output/'fit_manifest.json',report)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--small',action='store_true')
    args=parser.parse_args();compare(args.output,args.train,args.validation,small=args.small)
