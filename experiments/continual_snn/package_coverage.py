"""Package the selected V2 candidate and verify its training-only reproduction."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from .models import ROOT
from .loading import load_selector
from .prototypes import PrototypeMemory
from .compact_bank import CompactLegacyBank
from .readout import SpikeReadout
from .study import save

BASE=ROOT/'output/continual_snn'
RELEASE=ROOT/'docs/agents/continual_snn_v2'


def package():
    freeze=json.loads((BASE/'coverage-v2-freeze.json').read_text())
    source=ROOT/freeze['checkpoint']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=freeze['checkpoint_sha256']:raise ValueError('Frozen V2 candidate changed')
    RELEASE.mkdir(exist_ok=False)
    shutil.copyfile(source,RELEASE/'primary.npz')
    shutil.copyfile(BASE/'coverage-v2-freeze.json',RELEASE/'frozen_selection.json')
    manifest={}
    sources={'training_records.jsonl.gz':BASE/'coverage-v2-fit/training_records.jsonl',
             'validation_records.jsonl.gz':BASE/'coverage-v2-fit/validation_records.jsonl'}
    model=load_selector(RELEASE/'primary.npz')
    if isinstance(model,CompactLegacyBank):sources['source_associations16.json.gz']=BASE/'coverage-v2-fit/source_associations16.json'
    for name,path in sources.items():
        data=path.read_bytes();target=RELEASE/name
        with target.open('xb') as stream:
            with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as packed:packed.write(data)
        manifest[name]=dict(source=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(data).hexdigest(),
            compressed_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=len(data),compressed_bytes=target.stat().st_size)
    save(RELEASE/'reproduction_data.json',dict(files=manifest,final_evaluation_training_rows=0,
        labels='94 measured historical contexts plus separate teacher requests; new trajectories qualified by successful whole episodes',
        successful_episode_is_not_a_per_packet_counterfactual_outcome=True))
    fit=json.loads((BASE/'coverage-v2-fit/fit_manifest.json').read_text())
    card=dict(status='selected_on_validation_awaiting_fresh_confirmation',selected=freeze['selected'],
        checksum=model.checksum,encoder_checksum=model.encoder.checksum,
        encoder_weights_trained=False,new_motor_policy_trained=False,per_episode_learning=False,
        input_features=list(model.encoder.feature_names),training=fit,
        selector_kind='count-pattern association memory' if isinstance(model,PrototypeMemory) else type(model).__name__,
        patterns=len(model.counts) if hasattr(model,'counts') else None,
        neighbors=getattr(model,'neighbors',None),max_count_distance=getattr(model,'max_distance',None),
        frozen_utc=freeze['frozen_utc'],fresh_confirmation_pending=True,
        release_criterion='Zero contacts/falls/bound violations; stress success exceeds both V1 baselines; regular/noise success matches or exceeds both V1 baselines',
        torque_bounds_nm=dict(wait=24,run=24,jump=28),
        note='V1 test failures motivated new training-only coverage; V2 uses separately frozen fresh test seeds')
    save(RELEASE/'model_card.json',card)
    return model


def reproduce(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):raise ValueError('Output outside project output')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((RELEASE/'reproduction_data.json').read_text());payloads={}
    for name,item in manifest['files'].items():
        compressed=(RELEASE/name).read_bytes();data=gzip.decompress(compressed)
        if hashlib.sha256(data).hexdigest()!=item['source_sha256'] or hashlib.sha256(compressed).hexdigest()!=item['compressed_sha256']:
            raise ValueError('Coverage reproduction data changed')
        payloads[name]=data
    train=[json.loads(s) for s in payloads['training_records.jsonl.gz'].decode().splitlines()]
    val=[json.loads(s) for s in payloads['validation_records.jsonl.gz'].decode().splitlines()]
    if {r['group'] for r in train}&{r['group'] for r in val}:raise ValueError('Group leakage')
    reference=load_selector(RELEASE/'primary.npz')
    if isinstance(reference,CompactLegacyBank):
        from experiments.spike_pattern_skills.associations import AssociationMemory
        path=output/'source_associations16.json';path.write_bytes(payloads['source_associations16.json.gz'])
        memory=AssociationMemory.load(path,reference.encoder.checksum)
        model=CompactLegacyBank.from_legacy(reference.encoder,memory)
    elif isinstance(reference,PrototypeMemory):
        model=PrototypeMemory(reference.encoder,neighbors=reference.neighbors,max_distance=reference.max_distance).fit(train)
    else:
        model=SpikeReadout(reference.encoder,alpha=reference.alpha).fit([r['features'] for r in train],
            [r['label'] for r in train],[r['group'] for r in train])
    model.save(output/'rebuilt.npz')
    result=dict(training_rows=len(train),training_groups=len({r['group'] for r in train}),
        validation_rows=len(val),groups_disjoint=True,model_checksum_identical=model.checksum==reference.checksum,
        encoder_weights_trained=False,evaluation_rows_used_for_training=0)
    save(output/'reproduction.json',result)
    if not result['model_checksum_identical']:raise AssertionError(result)
    print(result,flush=True)


def finalize():
    card=json.loads((RELEASE/'model_card.json').read_text());phases={};bound_violations=[]
    expected=dict(stress=80,regular=80,noise=32)
    complete=True
    for phase,number in expected.items():
        folder=BASE/('coverage-v2-confirm-'+phase)
        aggregate=json.loads((folder/'aggregate.json').read_text())
        groups={r['arm']:r for r in aggregate['groups']}
        control=json.loads((BASE/('coverage-v2-control-'+phase)/'aggregate.json').read_text())['groups']
        groups.update({r['arm']:r for r in control});phases[phase]=groups
        complete &= all(groups.get(arm,{}).get('trials',0)==number for arm in ('v1_readout80','v1_compact16','v2_selected','rule'))
        complete &= groups.get('v1_prototype80',{}).get('trials',0)==number
        for line in (folder/'results.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['arm']=='v2_selected' and row['summary']['max_torque_limit_ratio']>1+1e-10:
                bound_violations.append(row['summary']['case'])
    if not complete:raise RuntimeError('Fresh V2 confirmation incomplete')
    accepted=not bound_violations
    for phase,groups in phases.items():
        selected=groups['v2_selected'];best=max(groups[n]['tasks'] for n in ('v1_readout80','v1_compact16','v1_prototype80'))
        accepted &= not selected['collisions'] and not selected['falls']
        accepted &= selected['tasks']>best if phase=='stress' else selected['tasks']>=best
    card.update(status='validated_for_this_simulation_contract' if accepted else 'diagnostic_candidate_not_promoted',
        fresh_confirmation_pending=False,fresh_confirmation=phases,release_criterion_met=bool(accepted),bound_violations=bound_violations,
        release_criterion='Zero contacts/falls/bound violations; stress success exceeds all three V1 controls including the same prototype architecture; regular/noise success matches or exceeds them')
    save(RELEASE/'model_card.json',card);print('V2 release criterion met:',bool(accepted),flush=True)


def provenance():
    for source,name in ((BASE/'coverage-v2-protocol.json','training_protocol.json'),
                        (BASE/'coverage-v2-fit/fit_manifest.json','fit_manifest.json'),
                        (BASE/'coverage-v2-fit/qualification.json','episode_qualification.json'),
                        (BASE/'coverage-v2-control-protocol.json','matched_control_protocol.json')):
        shutil.copyfile(source,RELEASE/name)
    evidence=[]
    for batch in ('coverage-v2-train','coverage-v2-validation-teacher'):
        for path in sorted((BASE/batch).glob('*/rule/summary.json')):
            evidence.append(dict(source=path.relative_to(ROOT).as_posix(),summary=json.loads(path.read_text()),
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save(RELEASE/'teacher_episode_outcomes.json',evidence)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',action='store_true')
    parser.add_argument('--reproduce',type=Path);parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--provenance',action='store_true')
    args=parser.parse_args()
    if args.package:package()
    if args.reproduce:reproduce(args.reproduce)
    if args.finalize:finalize()
    if args.provenance:provenance()
