"""Collect matched skill outcomes and fit a frozen count-pattern association bank."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from skills.dynamic import execute_dynamic_skill
from .associations import AssociationMemory, SKILLS, outcome, supervise
from .encoder import RandomLIFEncoder
from .observations import observation_from_simulation
from .record import ROOT
from .situations import situations

DEFAULT_OUTPUT = ROOT/'output/spike_pattern_skills/associations-v1'
ENCODER_PATH = ROOT/'docs/agents/spike_pattern_100/model.npz'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_trajectory(path, history):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        if history:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(history)


def collect(output=DEFAULT_OUTPUT):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT/output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty collection directory')
    output.mkdir(parents=True, exist_ok=True)
    encoder = RandomLIFEncoder.load(ENCODER_PATH)
    encoder.save(output/'encoder.npz')
    cases = [case for case in situations() if case.variant <= 4]
    source_paths = ['skills/dynamic.py', 'controllers/walking.py', 'controllers/navigation.py',
                    'experiments/spike_pattern_skills/associations.py',
                    'experiments/spike_pattern_skills/learn_associations.py',
                    'experiments/spike_pattern_skills/encoder.py',
                    'experiments/spike_pattern_skills/observations.py',
                    'experiments/spike_pattern_skills/situations.py']
    source_hashes = {name: digest(ROOT/name) for name in source_paths}
    labels, trials, errors = [], [], []
    with (output/'attempts.jsonl').open('x', encoding='utf-8') as attempts, (output/'labels.jsonl').open('x', encoding='utf-8') as label_stream:
        for case in cases:
            split = 'train' if case.variant < 4 else 'validation'
            folder = output/'trials'/case.scenario.name
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder/'scenario.json', case.scenario.to_dict())
            candidates = []
            for identifier in SKILLS:
                skill = identifier.removeprefix('dynamic/')
                try:
                    history, summary = execute_dynamic_skill(skill, case.scenario, seed=case.scene_seed)
                    observed = summary['decision_observation']
                    if observed is None:
                        raise RuntimeError('Episode ended before the causal decision observation')
                    observation = observation_from_simulation(case.scenario, observed, case.jump_requested,
                        observation_id=case.scenario.name+'-decision', context_id=case.scenario.name)
                    firing = encoder.record(observation)
                    measured = outcome(summary, case.jump_requested)
                    save_trajectory(folder/(skill+'.csv'), history)
                    write_json(folder/(skill+'.json'), dict(summary=summary, outcome=measured, pattern=firing))
                    trial = dict(context_id=case.scenario.name, family=case.family, variant=case.variant,
                        split=split, attempted_skill=identifier, eligible=summary['eligibility'][skill],
                        success=summary['success'], reason=summary['reason'], collision=summary['collision'],
                        fall=summary['fall'], max_torque=summary['max_torque'],
                        torque_limit_ratio=summary['max_torque_limit_ratio'], **measured,
                        spike_counts=firing['recording']['spike_counts'],
                        summary_path=str((folder/(skill+'.json')).relative_to(output)).replace('\\', '/'),
                        summary_sha256=digest(folder/(skill+'.json')), trace_sha256=digest(folder/(skill+'.csv')))
                    candidates.append((trial, firing, summary['eligibility']))
                    trials.append(trial)
                    attempts.write(json.dumps(trial, allow_nan=False)+'\n'); attempts.flush()
                except Exception as error:
                    failure = dict(context_id=case.scenario.name, attempted_skill=identifier,
                                   error_type=type(error).__name__, error=str(error))
                    errors.append(failure)
                    write_json(folder/(skill+'-error.json'), failure)
            if len(candidates) != 3:
                continue
            if any(c[1]['recording'] != candidates[0][1]['recording'] for c in candidates):
                raise AssertionError('Matched candidate observations/patterns differ before execution')
            competing = [c for c in candidates if c[0]['utility'] is not None]
            winner = max(competing, key=lambda c: c[0]['utility']) if competing else None
            label = dict(context_id=case.scenario.name, family=case.family, variant=case.variant,
                split=split, spike_counts=candidates[0][1]['recording']['spike_counts'],
                encoder_checksum=encoder.checksum, skill=winner[0]['attempted_skill'] if winner else None,
                label_role=('task_fulfilled' if winner[0]['task_fulfilled'] else 'local_hold_fallback') if winner else 'no_successful_candidate',
                task='in_place_jump' if case.jump_requested else 'forward_progress',
                eligibility=candidates[0][2], evidence=[c[0]['summary_path'] for c in candidates])
            labels.append(label)
            label_stream.write(json.dumps(label, allow_nan=False)+'\n'); label_stream.flush()
            print(case.scenario.name, split, label['skill'], [c[0]['reason'] for c in candidates], flush=True)
    write_json(output/'collection_errors.json', errors)
    if errors or len(trials) != 150 or len(labels) != 50:
        raise RuntimeError('Incomplete collection; all attempted records and errors were retained')
    examples = [{k: label[k] for k in ('context_id', 'spike_counts', 'skill', 'label_role', 'evidence')}
                for label in labels if label['split'] == 'train' and label['skill'] is not None]
    memory = AssociationMemory(encoder.checksum, examples, neighbors=3, max_distance=4.)
    memory.save(output/'association.json')
    validation = []
    for label in labels:
        if label['split'] != 'validation':
            continue
        proposal = memory.decode(label['spike_counts'])
        chosen = supervise(proposal, label['eligibility'])
        validation.append(dict(context_id=label['context_id'], oracle_skill=label['skill'],
            proposal=proposal, supervision=chosen,
            agreement=chosen['executed_skill'] == label['skill'] if label['skill'] else None))
    write_json(output/'validation.json', validation)
    manifest = dict(protocol='docs/agents/pattern_association_protocol.md', encoder_checksum=encoder.checksum,
        association_checksum=memory.checksum, train_contexts=40, validation_contexts=10, held_out_contexts=50,
        attempts=len(trials), failed_or_rejected_attempts=sum(not trial['success'] for trial in trials),
        contacts=sum(trial['collision'] for trial in trials), falls=sum(trial['fall'] for trial in trials),
        max_torque=max(trial['max_torque'] for trial in trials),
        training_associations=len(examples), labels=dict(Counter(e['skill'] for e in examples)),
        validation_agreement=sum(item['agreement'] is True for item in validation),
        validation_labeled=sum(item['agreement'] is not None for item in validation),
        validation_proposals=sum(item['proposal']['proposed_skill'] is not None for item in validation),
        source_sha256=source_hashes,
        files_sha256={name: digest(output/name) for name in ('encoder.npz', 'association.json',
            'attempts.jsonl', 'labels.jsonl', 'validation.json', 'collection_errors.json')},
        note='Encoder frozen; decoder parameters fixed before collection; validation does not update associations.')
    write_json(output/'training_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    collect(**vars(parser.parse_args()))
