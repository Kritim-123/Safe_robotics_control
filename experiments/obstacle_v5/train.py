"""Teacher-proposal collection, physics-labelled walk-up jump associations."""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v4.model import JumpHead, NAMES
from experiments.obstacle_v4.train import scene
from .model import WalkUpHead, MODEL
from .run import nominal
from .runner import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17501)
    scenes = [nominal()]
    for i in range(31):
        distance = float(rng.uniform(.75, 1.2))
        spec = scene('unused', distance, rng.uniform(.02, .04), rng.uniform(.012, .02), rng.uniform(.8, .9))
        scenes.append(replace(spec, name=f'v5-train-{i:02d}', goal=(distance+1.26, 0.)))
    base = JumpHead.load(); original = base.predict
    base.predict = lambda f: dict(original(f), proposed_skill='low_bar_jump',
                                 reason='teacher_forced_training_proposal', teacher_forced=True)
    records = []; outcomes = []
    for spec in scenes:
        folder = args.output/spec.name
        result = execute(spec, folder, head=base)
        outcomes.append(result)
        (args.output/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in outcomes))
        if result['jump_entry']:
            packets = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
            row = next(r for r in packets if abs(r['time']-result['jump_entry']['time']) < 1e-9)
            records.append(dict(source='measured_walk_up_jump_outcome', case=spec.name,
                                features=row['jump_features'], label=int(result['jump_completed']),
                                outcome=result))
        print(spec.name, result['reason'], 'crossing', result['jump_completed'], flush=True)
    measured = len(records)
    if not measured or not any(r['label'] for r in records):
        raise ValueError('No measured successful training jumps')
    for i in range(80):
        source = records[i % measured]
        features = dict(source['features'])
        key, value = (('height', rng.uniform(.06, .5)), ('target_speed', rng.uniform(.05, .5)),
                      ('landing_clearance', rng.uniform(-.3, .14)), ('stable', 0.),
                      ('depth', rng.uniform(.07, .3)), ('alignment', rng.uniform(.15, 1.)),
                      ('front', rng.uniform(.5, 1.5)), ('present', 0.))[i % 8]
        features[key] = float(value)
        records.append(dict(source='eligibility_negative_not_physics_outcome', features=features, label=0))
    metadata = dict(base.metadata, version='V5-walk-up-observation-count-associations',
                    features=list(NAMES), training_seed=17501, physics_trials=len(outcomes),
                    measured_association_rows=measured, contract_negative_rows=80,
                    inherited_encoder_head_checksum=base.checksum,
                    description='V4 LIF encoder; new measured walk-up outcome associations; no jump-request input')
    model = WalkUpHead(base.weights, base.biases,
                       np.array([base.encode(r['features']) for r in records]),
                       np.array([r['label'] for r in records]), metadata)
    model.save(args.model)
    with gzip.open(args.model.with_name(args.model.stem+'_training_records.jsonl.gz'), 'xt', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')
    summary = dict(physics_trials=len(outcomes), crossings=sum(r['jump_completed'] for r in outcomes),
                   goals=sum(r['success'] for r in outcomes), contacts=sum(r['collision'] for r in outcomes),
                   falls=sum(r['fall'] for r in outcomes), measured_association_rows=measured,
                   training_rows=len(records), head_checksum=model.checksum,
                   training_correct=sum((model.predict(r['features'])['proposed_skill'] == 'low_bar_jump') == bool(r['label'])
                                        for r in records))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
