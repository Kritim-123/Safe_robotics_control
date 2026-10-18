"""Disclosed outcome collection for the selected profile, then count associations."""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v4.train import scene
from experiments.obstacle_v5.run import cases as old_cases
from .develop import teacher
from .model import RefinedHead, ROOT, MODEL, PROFILE, NAMES, V3
from .runner import execute


def scenes(seed, count, split):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        distance = float(rng.uniform(.75, 1.2))
        spec = scene('unused', distance, rng.uniform(.02, .04), rng.uniform(.012, .02), rng.uniform(.8, .9))
        result.append(replace(spec, name=f'v5r-{split}-{seed}-{i:03d}', goal=(distance+1.26, 0.)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    cases = old_cases('test', 20)+scenes(17601, 60, 'train')
    records = []; outcomes = []
    for spec in cases:
        folder = args.output/spec.name
        result = execute(spec, folder, head=teacher(PROFILE))
        outcomes.append(result)
        (args.output/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outcomes))
        if result['jump_entry']:
            packets = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
            actuation = result['jump_entry']['time']
            packet_rows = [r for r in packets if r['eligible'] and r['phase'] == 'wait'
                           and actuation-.201 <= r['time'] <= actuation+1e-9]
            for row in packet_rows:
                records.append(dict(source='episode_jump_outcome_association', case=spec.name,
                                    features=row['jump_features'], label=int(result['jump_completed']),
                                    observation_time=row['time'], actuation_time=actuation,
                                    is_actuation_packet=abs(row['time']-actuation) < 1e-9,
                                    outcome=result))
        print(spec.name, result['reason'], result['jump_completed'], flush=True)
    measured = len(records)
    if not any(r['label'] for r in records):
        raise ValueError('No successful measured jumps')
    rng = np.random.default_rng(1760101)
    for i in range(96):
        features = dict(records[i % measured]['features'])
        key, value = (('height', rng.uniform(.06, .5)), ('target_speed', rng.uniform(.05, .5)),
                      ('landing_clearance', rng.uniform(-.3, .14)), ('stable', 0.),
                      ('depth', rng.uniform(.07, .3)), ('alignment', rng.uniform(.15, 1.)),
                      ('front', rng.uniform(.5, 1.5)), ('present', 0.))[i % 8]
        features[key] = float(value)
        records.append(dict(source='eligibility_negative_not_physics_outcome', features=features, label=0))
    w, b = RefinedHead.encoder_arrays()
    metadata = dict(version='V5-refined-count-outcomes', features=list(NAMES),
                    v3_sha256=hashlib.sha256(V3.read_bytes()).hexdigest(),
                    jump_profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                    reset_each_packet=True, lif_steps=16, beta=.9, threshold=1.,
                    original_cells=168, fine_cells=96, encoder_weights_learned=False,
                    training_seed=17601, physics_trials=len(outcomes), measured_packet_rows=measured,
                    contract_negative_rows=96, prior_v5_test_cases_used_for_development=True,
                    packet_labels='Actual actuation packet and up to two preceding eligible stable packets share the episode crossing outcome')
    temporary = RefinedHead(w, b, np.zeros((1, len(b))), np.zeros(1), metadata)
    model = RefinedHead(w, b, np.array([temporary.encode(r['features']) for r in records]),
                        np.array([r['label'] for r in records]), metadata)
    model.save(args.model)
    with gzip.open(args.model.with_name(args.model.stem+'_training_records.jsonl.gz'), 'xt', encoding='utf-8') as stream:
        for r in records:
            stream.write(json.dumps(r)+'\n')
    summary = dict(physics_trials=len(outcomes), crossings=sum(r['jump_completed'] for r in outcomes),
                   goals=sum(r['success'] for r in outcomes), contacts=sum(r['collision'] for r in outcomes),
                   falls=sum(r['fall'] for r in outcomes), measured_packet_rows=measured,
                   total_training_rows=len(records), head_checksum=model.checksum,
                   training_correct=sum((model.predict(r['features'])['proposed_skill'] == 'low_bar_jump') == bool(r['label']) for r in records))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
