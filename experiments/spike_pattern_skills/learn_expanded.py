"""Collect matched outcomes, fit expanded banks, and compare on untouched contexts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from skills.dynamic import execute_dynamic_skill
from .associations import AssociationMemory, SKILLS, supervise
from .encoder import RandomLIFEncoder
from .execute_associations import DEFAULT_ASSOCIATION, execute_case, measure_case_outcome
from .learn_associations import DEFAULT_OUTPUT, ENCODER_PATH, digest, save_trajectory, write_json
from .observations import observation_from_simulation
from .outcome_cases import learning_cases
from .record import ROOT
from .situations import situations
from .terrain_encoding import TerrainAwareEncoder, terrain_observation

STUDY_OUTPUT = ROOT/'output/spike_pattern_skills/expanded-associations-v1'


def load_models(folder):
    legacy = RandomLIFEncoder.load(folder/'legacy_encoder.npz')
    terrain = TerrainAwareEncoder.load(folder/'terrain_encoder.npz')
    return dict(original=(legacy, AssociationMemory.load(folder/'original.json', legacy.checksum)),
        expanded_legacy=(legacy, AssociationMemory.load(folder/'expanded_legacy.json', legacy.checksum)),
        terrain=(terrain, AssociationMemory.load(folder/'terrain.json', terrain.checksum)))


def conflict_summary(examples):
    groups = defaultdict(list)
    for e in examples:
        groups[tuple(e['spike_counts'])].append(e)
    conflicts = [[dict(context_id=e['context_id'], skill=e['skill']) for e in group]
                 for group in groups.values() if len({e['skill'] for e in group}) > 1]
    return dict(examples=len(examples), unique_counts=len(groups), conflicting_vectors=len(conflicts),
                conflicting_examples=sum(map(len, conflicts)), conflicts=conflicts)


def collect(folder):
    folder.mkdir(parents=True, exist_ok=False)
    legacy = RandomLIFEncoder.load(ENCODER_PATH)
    terrain = TerrainAwareEncoder.extend(legacy)
    legacy.save(folder/'legacy_encoder.npz'); terrain.save(folder/'terrain_encoder.npz')
    original = AssociationMemory.load(DEFAULT_ASSOCIATION, legacy.checksum)
    original.save(folder/'original.json')
    sources = ['skills/dynamic.py', 'hybrid/arena.py', 'hybrid/terrain.py', 'controllers/navigation.py',
        'controllers/walking.py', 'experiments/spike_pattern_skills/encoder.py',
        'experiments/spike_pattern_skills/terrain_encoding.py', 'experiments/spike_pattern_skills/observations.py',
        'experiments/spike_pattern_skills/associations.py', 'experiments/spike_pattern_skills/outcome_cases.py',
        'experiments/spike_pattern_skills/curriculum.py', 'experiments/spike_pattern_skills/execute_associations.py',
        'experiments/spike_pattern_skills/learn_expanded.py']
    manifest = dict(protocol='docs/agents/expanded_association_protocol.md',
        source_sha256={name: digest(ROOT/name) for name in sources},
        legacy_checksum=legacy.checksum, terrain_encoder_checksum=terrain.checksum,
        original_association_checksum=original.checksum, terrain_seed=20261019, flat_seed=20261020,
        python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
        parameters=dict(neighbors=3, max_distance=4., terrain_extension_seed=4242))
    cases = learning_cases()
    write_json(folder/'case_manifest.json', [dict(context_id=c.scenario.name, family=c.family,
        variant=c.variant, split=c.split, seed=c.scene_seed, stage=c.stage, jump_requested=c.jump_requested,
        duration=c.duration, timeout=c.timeout, traversal_x=c.traversal_x, scenario=c.scenario.to_dict()) for c in cases])
    write_json(folder/'collection_manifest.json', manifest)
    examples = dict(expanded_legacy=[], terrain=[])
    # Reuse only original training examples, verifying their actual causal trial records.
    old_manifest = json.loads((DEFAULT_OUTPUT/'training_manifest.json').read_text())
    if digest(DEFAULT_OUTPUT/'attempts.jsonl') != old_manifest['files_sha256']['attempts.jsonl']:
        raise ValueError('Original trial index changed')
    attempts = {r['summary_path']: r for r in map(json.loads, (DEFAULT_OUTPUT/'attempts.jsonl').read_text().splitlines())}
    with (folder/'reused_training.jsonl').open('x', encoding='utf-8') as stream:
        for e in original.examples:
            path = e['evidence'][0]
            if digest(DEFAULT_OUTPUT/path) != attempts[path]['summary_sha256']:
                raise ValueError('Original training evidence changed')
            trial = json.loads((DEFAULT_OUTPUT/path).read_text())
            item = trial['pattern']['observation']
            if legacy.encode(item['features'])['spike_counts'] != e['spike_counts']:
                raise ValueError('Original observation does not reproduce its training pattern')
            oldcase = next(c for c in situations() if c.scenario.name == e['context_id'])
            newitem = terrain_observation(oldcase.scenario, item['metadata']['robot_observation'],
                oldcase.jump_requested, observation_id=item['observation_id'], context_id=e['context_id'])
            extra = terrain.encode(newitem['features'])
            if extra['spike_counts'] != e['spike_counts']:
                raise ValueError('Flat observations must preserve original counts')
            for name in examples:
                examples[name].append(dict(e, context_id='original/'+e['context_id'],
                    split='train', source='original verified training', evidence=[path]))
            stream.write(json.dumps(dict(example=e, observation=item, terrain_observation=newitem,
                evidence_path=str(DEFAULT_OUTPUT/path), evidence_sha256=digest(DEFAULT_OUTPUT/path)))+'\n')
    labels, trials, errors = [], [], []
    with (folder/'attempts.jsonl').open('x', encoding='utf-8') as attempts_file, \
            (folder/'labels.jsonl').open('x', encoding='utf-8') as labels_file:
        for case in (c for c in cases if c.split != 'held_out'):
            directory = folder/'trials'/case.scenario.name
            directory.mkdir(parents=True)
            write_json(directory/'scenario.json', case.scenario.to_dict())
            candidates, observed_features = [], []
            for identifier in SKILLS:
                skill = identifier.removeprefix('dynamic/')
                try:
                    history, summary = execute_dynamic_skill(skill, case.scenario, seed=case.scene_seed,
                                                            duration=case.duration, timeout=case.timeout)
                    observed = summary['decision_observation']
                    if observed is None:
                        raise RuntimeError('Candidate ended before selection observation')
                    item = observation_from_simulation(case.scenario, observed, case.jump_requested,
                        observation_id=case.scenario.name+'-decision', context_id=case.scenario.name)
                    newitem = terrain_observation(case.scenario, observed, case.jump_requested,
                        observation_id=case.scenario.name+'-decision', context_id=case.scenario.name)
                    old_pattern, new_pattern = legacy.encode(item['features']), terrain.encode(newitem['features'])
                    observed_features.append(newitem['features'])
                    measured = measure_case_outcome(case, history, summary)
                    valid = summary['success'] and not summary['collision'] and not summary['fall'] and summary['max_torque_limit_ratio'] <= 1+1e-9
                    required = 'jump' if case.jump_requested else 'run'
                    utility = measured['utility'] if skill == required and measured['task_fulfilled'] else None
                    if skill == 'wait' and valid:
                        utility = 0.
                    row = dict(context_id=case.scenario.name, family=case.family, stage=case.stage,
                        split=case.split, attempted_skill=identifier, eligible=summary['eligibility'][skill],
                        success=summary['success'], reason=summary['reason'], task_fulfilled=measured['task_fulfilled'],
                        utility=utility, collision=summary['collision'], fall=summary['fall'],
                        max_torque=summary['max_torque'], max_torque_limit_ratio=summary['max_torque_limit_ratio'],
                        legacy_counts=old_pattern['spike_counts'], terrain_counts=new_pattern['spike_counts'],
                        summary_path=str((directory/(skill+'.json')).relative_to(folder)).replace('\\', '/'))
                    save_trajectory(directory/(skill+'.csv'), history)
                    write_json(directory/(skill+'.json'), dict(summary=summary, outcome=measured,
                        observation=item, terrain_observation=newitem, legacy_pattern=old_pattern, terrain_pattern=new_pattern))
                    row['summary_sha256'] = digest(directory/(skill+'.json'))
                    row['trajectory_sha256'] = digest(directory/(skill+'.csv'))
                    attempts_file.write(json.dumps(row)+'\n'); attempts_file.flush()
                    trials.append(row); candidates.append(row)
                except Exception as error:
                    failure = dict(context_id=case.scenario.name, skill=identifier, error_type=type(error).__name__, error=str(error))
                    errors.append(failure); write_json(directory/(skill+'-error.json'), failure)
            if len(candidates) != 3:
                continue
            if any(item != observed_features[0] for item in observed_features):
                raise ValueError('Matched candidates have different pre-execution inputs')
            competing = [c for c in candidates if c['utility'] is not None]
            winner = max(competing, key=lambda c: c['utility']) if competing else None
            label = dict(context_id=case.scenario.name, family=case.family, stage=case.stage, split=case.split,
                skill=winner['attempted_skill'] if winner else None,
                label_role=('task_fulfilled' if winner['task_fulfilled'] else 'local_hold_fallback') if winner else 'no_successful_candidate',
                legacy_counts=candidates[0]['legacy_counts'], terrain_counts=candidates[0]['terrain_counts'],
                eligibility=summary['eligibility'], evidence=[c['summary_path'] for c in candidates])
            labels.append(label); labels_file.write(json.dumps(label)+'\n'); labels_file.flush()
            if case.split == 'train' and winner:
                for name, key in (('expanded_legacy', 'legacy_counts'), ('terrain', 'terrain_counts')):
                    examples[name].append(dict(context_id=label['context_id'], skill=label['skill'],
                        spike_counts=label[key], label_role=label['label_role'], split='train', evidence=label['evidence']))
            print(case.scenario.name, case.split, label['skill'], [c['reason'] for c in candidates], flush=True)
    write_json(folder/'collection_errors.json', errors)
    if errors or len(trials) != 204 or len(labels) != 68:
        raise RuntimeError('Incomplete candidate collection; evidence and errors retained')
    for name, encoder in (('expanded_legacy', legacy), ('terrain', terrain)):
        AssociationMemory(encoder.checksum, examples[name], neighbors=3, max_distance=4.).save(folder/(name+'.json'))
    models = load_models(folder)
    validation = []
    for label in (label for label in labels if label['split'] == 'validation'):
        for name, (_, memory) in models.items():
            counts = label['terrain_counts' if name == 'terrain' else 'legacy_counts']
            proposal = memory.decode(counts); chosen = supervise(proposal, label['eligibility'])
            validation.append(dict(context_id=label['context_id'], model=name, oracle_skill=label['skill'],
                proposal=proposal, supervision=chosen, agreement=chosen['executed_skill'] == label['skill']))
    write_json(folder/'validation.json', validation)
    manifest.update(attempts=len(trials), new_train_contexts=sum(c.split == 'train' for c in cases),
        validation_contexts=14, held_out_contexts=28, reused_training_contexts=40,
        candidate_failed_or_rejected=sum(not t['success'] for t in trials),
        contacts=sum(t['collision'] for t in trials), falls=sum(t['fall'] for t in trials),
        max_torque=max(t['max_torque'] for t in trials),
        banks={name: dict(checksum=m.checksum, labels=dict(Counter(e['skill'] for e in m.examples)),
                         **conflict_summary(m.examples)) for name, (_, m) in models.items()},
        files_sha256={name: digest(folder/name) for name in ('original.json', 'expanded_legacy.json', 'terrain.json',
            'legacy_encoder.npz', 'terrain_encoder.npz', 'case_manifest.json', 'attempts.jsonl',
            'labels.jsonl', 'reused_training.jsonl', 'validation.json', 'collection_errors.json')})
    write_json(folder/'collection_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('attempts','candidate_failed_or_rejected','contacts','falls')}, indent=2), flush=True)
    return manifest


def aggregate(rows):
    groups = []
    for split in ('held_out', 'regression'):
        for model in ('original', 'expanded_legacy', 'terrain', 'direct_run'):
            group = [r for r in rows if r['split'] == split and r['model'] == model]
            if not group:
                continue
            groups.append(dict(split=split, model=model, episodes=len(group),
                task_fulfilled=sum(r['task_fulfilled'] for r in group),
                primitive_completed=sum(r['primitive_completed'] for r in group),
                collisions=sum(r['collision'] for r in group), falls=sum(r['fall'] for r in group),
                proposals=sum(r['proposed_skill'] is not None for r in group),
                fallbacks=sum(r['fallback'] for r in group), skills=dict(Counter(r['skill'] for r in group)),
                reasons=dict(Counter(r['reason'] for r in group)),
                max_torque=max(r['max_torque'] for r in group),
                max_torque_limit_ratio=max(r['max_torque_limit_ratio'] for r in group)))
    return dict(episodes=len(rows), groups=groups)


def evaluate(folder):
    manifest = json.loads((folder/'collection_manifest.json').read_text())
    for name, checksum in manifest['files_sha256'].items():
        if digest(folder/name) != checksum:
            raise ValueError('Frozen collection artifact changed: '+name)
    models = load_models(folder)
    output = folder/'evaluation'; output.mkdir(exist_ok=False)
    jobs = [(c, mode, 'held_out') for c in learning_cases() if c.split == 'held_out'
            for mode in (*models, 'direct_run')]
    regression = [c for c in situations() if c.family in ('clear','static_cylinder','static_box','low_bar',
                    'crossing_left_to_right','crossing_right_to_left') and c.variant in (6,7)]
    jobs += [(c, mode, 'regression') for c in regression for mode in models]
    rows = []
    with (output/'results.jsonl').open('x', encoding='utf-8') as stream:
        for case, name, split in jobs:
            encoder, memory = models['original' if name == 'direct_run' else name]
            episode = output/'episodes'/(case.scenario.name+'-'+name)
            result = execute_case(case, memory, encoder, episode,
                duration=getattr(case, 'duration', 4.), timeout=getattr(case, 'timeout', 12.),
                selection='direct_run' if name == 'direct_run' else 'association')
            summary, measured = result['summary'], result['outcome']
            decision = result['decision']
            row = dict(context_id=case.scenario.name, family=case.family, stage=getattr(case, 'stage', -1),
                model=name, split=split, task=measured['task'], task_fulfilled=measured['task_fulfilled'],
                primitive_completed=summary['success'], skill=summary['skill'], reason=summary['reason'],
                collision=summary['collision'], fall=summary['fall'], max_torque=summary['max_torque'],
                max_torque_limit_ratio=summary['max_torque_limit_ratio'],
                proposed_skill=decision.get('proposal', {}).get('proposed_skill'),
                proposal_reason=decision.get('proposal', {}).get('reason'),
                fallback=decision.get('supervision', {}).get('fallback', True),
                result_path=str((episode/'result.json').relative_to(output)).replace('\\', '/'))
            rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            write_json(output/'aggregate.json', aggregate(rows))
            print(split, case.scenario.name, name, summary['skill'], summary['reason'], measured['task_fulfilled'], flush=True)
    for name, checksum in manifest['files_sha256'].items():
        if digest(folder/name) != checksum:
            raise ValueError('Evaluation changed frozen training artifacts')
    report = aggregate(rows)
    held = {g['model']: g for g in report['groups'] if g['split'] == 'held_out'}
    reg = {g['model']: g for g in report['groups'] if g['split'] == 'regression'}
    report['promotion_eligible'] = {name: bool(held[name]['task_fulfilled'] > held['original']['task_fulfilled']
        and not held[name]['collisions'] and not held[name]['falls']
        and not reg[name]['collisions'] and not reg[name]['falls']
        and reg[name]['task_fulfilled'] >= reg['original']['task_fulfilled']) for name in ('expanded_legacy','terrain')}
    report['training_files_unchanged'] = True
    write_json(output/'aggregate.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def demo(folder, model, names, viewer=False, video=False):
    import uuid
    encoder, memory = load_models(folder)[model]
    cases = learning_cases()
    chosen = [next(c for c in cases if c.scenario.name == name) for name in names]
    output = folder/'demos'/uuid.uuid4().hex
    for case in chosen:
        result = execute_case(case, memory, encoder, output/case.scenario.name,
            duration=case.duration, timeout=case.timeout, viewer=viewer, video=video)
        print(json.dumps(dict(context_id=case.scenario.name, outcome=result['outcome']), indent=2), flush=True)
        if result['summary']['reason'] == 'viewer closed':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all','collect','evaluate','demo'), default='all')
    parser.add_argument('--output', type=Path, default=STUDY_OUTPUT)
    parser.add_argument('--model', choices=('original','expanded_legacy','terrain'), default='terrain')
    parser.add_argument('--cases', nargs='+', default=['new-terrain-00-flat-00','new-requested_hop-00'])
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    folder = args.output if args.output.is_absolute() else ROOT/args.output
    if args.phase in ('all','collect'):
        collect(folder)
    if args.phase in ('all','evaluate'):
        evaluate(folder)
    if args.phase == 'demo':
        demo(folder, args.model, args.cases, args.viewer, args.video)
