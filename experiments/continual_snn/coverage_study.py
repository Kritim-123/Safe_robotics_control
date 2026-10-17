"""Version 2: targeted training-only coverage, validation selection, fresh tests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
from .models import ROOT,BUNDLE
from .loading import load_selector
from .readout import SpikeReadout,LABELS
from .prototypes import PrototypeMemory
from .compact_bank import CompactLegacyBank
from .predictive import PREDICTIVE_FEATURES
from .data import save_records
from .fit import metrics
from .scenes import cases
from .stress_scenes import make_stress_case,stress_cases
from .study import save,run_batch
from .evaluate import verify,DEADLINE
from .observations import SensorConfig

BASE=ROOT/'output/continual_snn'


def qualified_records(folder,split):
    rows=[];qualification=[]
    for path in sorted(Path(folder).glob('*/rule/decisions.jsonl')):
        summary_path=path.parent/'summary.json';summary=json.loads(summary_path.read_text())
        qualification.append(dict(case=summary['case'],task_fulfilled=summary['task_fulfilled'],
            reason=summary['reason'],summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest()))
        if not summary['task_fulfilled']:continue
        for line in path.read_text().splitlines():
            item=json.loads(line)
            if item['teacher_request'] is None:continue
            features=dict(item['features'],**{n:item['geometry'][n] for n in PREDICTIVE_FEATURES})
            rows.append(dict(group=summary['case'],split=split,features=features,label=item['teacher_request'],
                time=item['time'],source='qualified_hard_stream',
                label_source='episode_qualified_causal_teacher_not_per_window_counterfactual_outcome',
                source_episode_summary_sha256=qualification[-1]['summary_sha256']))
    return rows,qualification


def candidates(folder,extra_train,extra_val):
    old=BASE/'predictive-comparison-v1'
    train=[json.loads(s) for s in (old/'training_records.jsonl').read_text().splitlines()]+extra_train
    val=[json.loads(s) for s in (old/'validation_records.jsonl').read_text().splitlines()]+extra_val
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('Coverage group leakage')
    save_records(folder/'training_records.jsonl',train);save_records(folder/'validation_records.jsonl',val)
    save_records(folder/'additional_training_records.jsonl',extra_train)
    save_records(folder/'additional_validation_records.jsonl',extra_val)
    encoder=load_selector(ROOT/'docs/agents/continual_snn_v1/primary.npz').encoder
    xt=encoder.embed_many([r['features'] for r in train]);xv=encoder.embed_many([r['features'] for r in val])
    yt=[r['label'] for r in train];yv=[r['label'] for r in val]
    gt=[r['group'] for r in train];gv=[r['group'] for r in val]
    ranking=[];best=None
    for alpha in (.0001,.001,.01,.1,1.):
        model=SpikeReadout(encoder,alpha=alpha).fit_embeddings(xt,yt,gt)
        metric=metrics(yv,np.argmax(model.predict_embeddings(xv),axis=1),gv)
        row=dict(alpha=alpha,metrics=metric,checksum=model.checksum);ranking.append(row)
        if best is None or metric['balanced_accuracy']>best[0]['metrics']['balanced_accuracy']:best=(row,model)
    best[1].save(folder/'readout80.npz');save(folder/'ridge_validation.json',ranking)
    prototype=PrototypeMemory(encoder,neighbors=1,max_distance=12.).fit(train)
    prototype.save(folder/'prototype80.npz')
    source_encoder=DenseTerrainEncoder.load(BUNDLE/'dense_encoder.npz')
    original=BASE/'data-ablation-v1/models/motion_data_same_decoder_associations.json'
    memory=AssociationMemory.load(original,source_encoder.checksum)
    examples=list(memory.examples)
    for i,row in enumerate(extra_train):
        sensory={n:row['features'][n] for n in source_encoder.feature_names}
        examples.append(dict(context_id=f'qualified-hard/{row["group"]}/{i:05d}',
            spike_counts=source_encoder.encode(sensory)['spike_counts'],skill='dynamic/'+row['label'],
            label_source=row['label_source'],source_group=row['group'],time=row['time'],
            source_episode_summary_sha256=row['source_episode_summary_sha256']))
    memory=AssociationMemory(source_encoder.checksum,examples,neighbors=memory.neighbors,max_distance=memory.max_distance)
    memory.save(folder/'source_associations16.json')
    compact=CompactLegacyBank.from_legacy(source_encoder,memory);compact.save(folder/'compact16.npz')
    save(folder/'fit_manifest.json',dict(training_rows=len(train),training_groups=len(set(gt)),
        validation_rows=len(val),validation_groups=len(set(gv)),groups_disjoint=True,
        added_training_rows=len(extra_train),added_validation_rows=len(extra_val),
        encoder_weights_or_motor_changes=False,all_additional_rows_qualified_by_successful_episode=True,
        per_packet_labels_are_teacher_requests=True,
        compact_conflicting_patterns=int(sum(v not in (1,2,4) for v in compact.label_bits)),
        prototype_conflicting_patterns=int(sum(v not in (1,2,4) for v in prototype.label_bits))))
    return dict(readout80=best[1],prototype80=prototype,compact16=compact)


def main():
    verify()
    train=[make_stress_case(f,s,'coverage_train') for s in range(11000,11016)
           for f in ('oscillating_gate','multiple_obstacles')]
    val=[make_stress_case(f,s,'coverage_validation') for s in range(11100,11104)
         for f in ('oscillating_gate','multiple_obstacles')]
    extra_sources=['experiments/continual_snn/'+n+'.py' for n in ('coverage_study','loading','compact_bank')]
    frozen_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in extra_sources}
    save(BASE/'coverage-v2-protocol.json',dict(started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        rationale='V1 stress failures motivate fresh training-only coverage; V1 tests are development evidence for V2',
        training_cases=[c.to_dict() for c in train],validation_cases=[c.to_dict() for c in val],
        qualification='Include teacher rows only from fully task-fulfilled training/validation episodes; retain failures in raw cohorts',
        labels='Successful episode qualifies the trajectory; per-packet requests remain teacher labels, not counterfactual outcomes',
        candidates='Same frozen 80-neuron encoder/readout or prototype, and same original16 k3/r4 bank with new data',
        selection='Zero contacts/falls/bound violations; most physical validation tasks; ties prefer fewer neurons then smaller artifact',
        fresh_test_seeds=dict(stress=list(range(11200,11210)),regular=list(range(11300,11310)),noise=list(range(11300,11304))),
        old_primary_models_and_runtime_unchanged=True,source_sha256=frozen_hashes,deadline_epoch=DEADLINE))
    run_batch(BASE/'coverage-v2-train',train,[('rule','rule',None)],deadline=DEADLINE)
    run_batch(BASE/'coverage-v2-validation-teacher',val,[('rule','rule',None)],deadline=DEADLINE)
    extra_train,qtrain=qualified_records(BASE/'coverage-v2-train','train')
    extra_val,qval=qualified_records(BASE/'coverage-v2-validation-teacher','validation')
    folder=BASE/'coverage-v2-fit';folder.mkdir(exist_ok=False)
    save(folder/'qualification.json',dict(train=qtrain,validation=qval))
    if len({r['group'] for r in extra_train})<24 or len({r['group'] for r in extra_val})<6:
        raise RuntimeError('Insufficient qualified hard-scene coverage')
    new=candidates(folder,extra_train,extra_val)
    old80=load_selector(ROOT/'docs/agents/continual_snn_v1/primary.npz')
    old16=load_selector(ROOT/'docs/agents/continual_snn_v1/compact_motion_bank16.npz')
    validation=cases('validation',range(5100,5104))+val
    arms=[('v1_readout80','continuous',old80),('v1_compact16','continuous',old16)]
    arms += [(name,'continuous',model) for name,model in new.items()]+[('rule','rule',None)]
    result=run_batch(BASE/'coverage-v2-physical-validation',validation,arms,deadline=DEADLINE)
    eligible=[]
    for name,model in new.items():
        row=next(r for r in result['groups'] if r['arm']==name)
        if row['trials']!=len(validation) or row['collisions'] or row['falls']:continue
        neurons=16 if name=='compact16' else 80
        eligible.append((row['tasks'],-neurons,-(folder/(name+'.npz')).stat().st_size,name))
    if not eligible:raise RuntimeError('No qualified V2 selector; models remain diagnostic')
    selected_name=max(eligible)[-1];selected=new[selected_name]
    check_cases=dict(stress=stress_cases(range(11200,11210)),regular=cases('confirmation',range(11300,11310)),
                     noise=cases('confirmation',range(11300,11304)))
    freeze=dict(frozen_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),selected=selected_name,
        checksum=selected.checksum,checkpoint=str((folder/(selected_name+'.npz')).relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256((folder/(selected_name+'.npz')).read_bytes()).hexdigest(),
        validation=result,source_sha256=frozen_hashes,
        cases={name:[c.to_dict() for c in scenes] for name,scenes in check_cases.items()},
        training_or_hyperparameter_changes_after_freeze=False)
    save(BASE/'coverage-v2-freeze.json',freeze)
    test_arms=[('v1_readout80','continuous',old80),('v1_compact16','continuous',old16),
               ('v2_selected','continuous',selected),('rule','rule',None)]
    for phase,scenes in check_cases.items():
        verify()
        for path,checksum in frozen_hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=checksum:raise ValueError('V2 source changed')
        sensor=SensorConfig(.03,.015,.1,20261019) if phase=='noise' else None
        output=BASE/('coverage-v2-confirm-'+phase)
        run_batch(output,scenes,test_arms,sensor=sensor,deadline=DEADLINE)
    save(BASE/'coverage-v2-completed.json',dict(selected=selected_name,checksum=selected.checksum,
        finished_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        original_study_unmodified=True,all_tests_prespecified_before_their_outcomes=True))


if __name__=='__main__':main()
