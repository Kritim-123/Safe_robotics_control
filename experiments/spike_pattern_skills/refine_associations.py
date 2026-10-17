"""Re-encode training observations after a training-set sampling coverage audit."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import uuid

from .associations import AssociationMemory, supervise
from .dense_terrain_encoding import DenseTerrainEncoder, dense_observation
from .encoder import RandomLIFEncoder
from .execute_associations import execute_case
from .learn_associations import digest, write_json
from .learn_expanded import STUDY_OUTPUT, aggregate, conflict_summary, load_models
from .outcome_cases import learning_cases
from .record import ROOT
from .situations import situations


def fit(source, folder):
    folder.mkdir(parents=True, exist_ok=False)
    collection = json.loads((source/'collection_manifest.json').read_text())
    for name, checksum in collection['files_sha256'].items():
        if digest(source/name) != checksum:
            raise ValueError('Original collection changed: '+name)
    legacy = RandomLIFEncoder.load(source/'legacy_encoder.npz')
    encoder = DenseTerrainEncoder.extend(legacy)
    encoder.save(folder/'dense_encoder.npz')
    original = load_models(source)['original'][1]
    legacy.save(folder/'legacy_encoder.npz'); original.save(folder/'original.json')
    examples, records = [], []
    for old in map(json.loads, (source/'reused_training.jsonl').read_text().splitlines()):
        e, item = old['example'], old['observation']
        case = next(c for c in situations() if c.scenario.name == e['context_id'])
        observed = dense_observation(case.scenario, item['metadata']['robot_observation'], case.jump_requested,
            observation_id=item['observation_id'], context_id=e['context_id'])
        counts = encoder.encode(observed['features'])['spike_counts']
        if counts != e['spike_counts']:
            raise ValueError('Flat response changed')
        examples.append(dict(e, context_id='original/'+e['context_id'], spike_counts=counts,
                             split='train', source='original verified training'))
    cases = {c.scenario.name: c for c in learning_cases()}
    labels = list(map(json.loads, (source/'labels.jsonl').read_text().splitlines()))
    for label in labels:
        case = cases[label['context_id']]
        trial = json.loads((source/label['evidence'][0]).read_text())
        observed = trial['observation']['metadata']['robot_observation']
        item = dense_observation(case.scenario, observed, case.jump_requested,
            observation_id=case.scenario.name+'-decision', context_id=case.scenario.name)
        counts = encoder.encode(item['features'])['spike_counts']
        record = dict(label, dense_counts=counts, dense_observation=item)
        records.append(record)
        if label['split'] == 'train' and label['skill']:
            examples.append(dict(context_id=case.scenario.name, skill=label['skill'], spike_counts=counts,
                label_role=label['label_role'], split='train', evidence=label['evidence']))
    memory = AssociationMemory(encoder.checksum, examples, neighbors=3, max_distance=4.)
    memory.save(folder/'dense.json')
    with (folder/'reencoded_labels.jsonl').open('x', encoding='utf-8') as stream:
        for r in records:
            stream.write(json.dumps(r)+'\n')
    validation = []
    for r in (r for r in records if r['split'] == 'validation'):
        proposal = memory.decode(r['dense_counts']); chosen = supervise(proposal, r['eligibility'])
        validation.append(dict(context_id=r['context_id'], oracle_skill=r['skill'], proposal=proposal,
                               supervision=chosen, agreement=chosen['executed_skill'] == r['skill']))
    write_json(folder/'validation.json', validation)
    report = dict(reason='Training-set audit: a narrow threshold fell between coarse scan samples.',
        source_collection=str(source), source_manifest_sha256=digest(source/'collection_manifest.json'),
        source_files_sha256=collection['files_sha256'], encoder_checksum=encoder.checksum,
        association_checksum=memory.checksum, **conflict_summary(examples),
        labels=dict(Counter(e['skill'] for e in examples)), new_physical_training_trials=0,
        candidate_outcomes_reused=204, validation_agreement=sum(r['agreement'] for r in validation),
        validation_proposals=sum(r['proposal']['proposed_skill'] is not None for r in validation),
        fresh_evaluation_seeds=dict(terrain=20261023, flat=20261024),
        source_sha256={name: digest(ROOT/name) for name in (
            'experiments/spike_pattern_skills/dense_terrain_encoding.py',
            'experiments/spike_pattern_skills/refine_associations.py',
            'experiments/spike_pattern_skills/execute_associations.py')},
        files_sha256={name: digest(folder/name) for name in ('dense_encoder.npz','dense.json',
            'legacy_encoder.npz','original.json','reencoded_labels.jsonl','validation.json')})
    write_json(folder/'fit_manifest.json', report)
    print(json.dumps({k:report[k] for k in ('examples','unique_counts','conflicting_vectors','validation_agreement')}, indent=2), flush=True)
    return report


def fresh_cases():
    return [replace(c, scenario=replace(c.scenario, name='fresh-'+c.scenario.name))
            for c in learning_cases(20261023,20261024) if c.split == 'held_out']


def models(folder):
    legacy = RandomLIFEncoder.load(folder/'legacy_encoder.npz')
    dense = DenseTerrainEncoder.load(folder/'dense_encoder.npz')
    return dict(original=(legacy,AssociationMemory.load(folder/'original.json', legacy.checksum)),
                dense=(dense,AssociationMemory.load(folder/'dense.json', dense.checksum)))


def evaluate(folder):
    manifest = json.loads((folder/'fit_manifest.json').read_text())
    for name, checksum in manifest['files_sha256'].items():
        if digest(folder/name) != checksum:
            raise ValueError('Frozen fit changed')
    candidates = models(folder)
    output = folder/'evaluation'; output.mkdir(exist_ok=False)
    jobs = [(c, mode, 'held_out') for c in fresh_cases() for mode in ('original','dense','direct_run')]
    regression = [c for c in situations() if c.family in ('clear','static_cylinder','static_box','low_bar',
        'crossing_left_to_right','crossing_right_to_left') and c.variant in (6,7)]
    jobs += [(c,mode,'regression') for c in regression for mode in ('original','dense')]
    rows = []
    with (output/'results.jsonl').open('x',encoding='utf-8') as stream:
        for case,name,split in jobs:
            encoder,memory = candidates['original' if name == 'direct_run' else name]
            episode = output/'episodes'/(case.scenario.name+'-'+name)
            result = execute_case(case,memory,encoder,episode,
                duration=getattr(case,'duration',4.),timeout=getattr(case,'timeout',12.),
                selection='direct_run' if name == 'direct_run' else 'association')
            summary, measured, decision = result['summary'],result['outcome'],result['decision']
            row = dict(context_id=case.scenario.name,family=case.family,stage=getattr(case,'stage',-1),
                model=name,split=split,task=measured['task'],task_fulfilled=measured['task_fulfilled'],
                primitive_completed=summary['success'],skill=summary['skill'],reason=summary['reason'],
                collision=summary['collision'],fall=summary['fall'],max_torque=summary['max_torque'],
                max_torque_limit_ratio=summary['max_torque_limit_ratio'],
                proposed_skill=decision.get('proposal',{}).get('proposed_skill'),
                proposal_reason=decision.get('proposal',{}).get('reason'),
                fallback=decision.get('supervision',{}).get('fallback',True),
                result_path=str((episode/'result.json').relative_to(output)).replace('\\','/'))
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            # The existing aggregation helper recognizes 'terrain'; use an explicit
            # temporary alias only for grouping, then preserve the dense model name.
            groups = aggregate([dict(r,model='terrain' if r['model']=='dense' else r['model']) for r in rows])
            for group in groups['groups']:
                if group['model']=='terrain':
                    group['model']='dense'
            write_json(output/'aggregate.json',groups)
            print(split,case.scenario.name,name,summary['skill'],summary['reason'],measured['task_fulfilled'],flush=True)
    for name,checksum in manifest['files_sha256'].items():
        if digest(folder/name)!=checksum:
            raise ValueError('Evaluation mutated frozen model')
    held={g['model']:g for g in groups['groups'] if g['split']=='held_out'}
    reg={g['model']:g for g in groups['groups'] if g['split']=='regression'}
    groups['promotion_eligible']=bool(held['dense']['task_fulfilled']>held['original']['task_fulfilled']
        and not held['dense']['collisions'] and not held['dense']['falls']
        and not reg['dense']['collisions'] and not reg['dense']['falls']
        and reg['dense']['task_fulfilled']>=reg['original']['task_fulfilled'])
    groups['training_files_unchanged']=True
    write_json(output/'aggregate.json',groups)
    print(json.dumps(groups,indent=2),flush=True)
    return groups


def demo(folder,names,viewer=False,video=False):
    encoder,memory=models(folder)['dense']
    cases=learning_cases()
    output=folder/'demos'/uuid.uuid4().hex
    for name in names:
        case=next(c for c in cases if c.scenario.name==name)
        result=execute_case(case,memory,encoder,output/name,viewer=viewer,video=video,
                            duration=case.duration,timeout=case.timeout)
        print(json.dumps(dict(context_id=name,outcome=result['outcome']),indent=2),flush=True)
        if result['summary']['reason']=='viewer closed':
            break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','fit','evaluate','demo'),default='all')
    parser.add_argument('--source',type=Path,default=STUDY_OUTPUT)
    parser.add_argument('--output',type=Path,default=ROOT/'output/spike_pattern_skills/dense-associations-v1')
    parser.add_argument('--cases',nargs='+',default=['new-terrain-00-flat-00','new-requested_hop-00'])
    parser.add_argument('--viewer',action='store_true');parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    source=args.source if args.source.is_absolute() else ROOT/args.source
    folder=args.output if args.output.is_absolute() else ROOT/args.output
    if args.phase in ('all','fit'):fit(source,folder)
    if args.phase in ('all','evaluate'):evaluate(folder)
    if args.phase=='demo':demo(folder,args.cases,args.viewer,args.video)
