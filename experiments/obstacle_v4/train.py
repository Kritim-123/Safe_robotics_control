"""Collect actual frozen-profile jump outcomes before fitting count associations."""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from hybrid.arena import Scenario, Obstacle
from skills.jump_profile import JumpConfig
from skills.obstacle_jump import execute_obstacle_jump, PROFILE
from .model import ROOT, MODEL, V3, NAMES, JumpHead, observe


def scene(name='nominal', distance=.24, height=.04, depth=.02, friction=.85):
    return Scenario('v4-'+name, goal=(1.5, 0.), bounds=(-4., 5., -4., 4.), friction=friction,
                    obstacles=(Obstacle('bar', (distance, 0.), shape='box',
                                        half_extents=(depth/2, .6), height=height),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(17401)
    scenarios = [scene()]+[scene(f'train-{i:02d}', rng.uniform(.23, .24), rng.uniform(.02, .04),
                                rng.uniform(.012, .02), rng.uniform(.8, .9)) for i in range(31)]
    config = JumpConfig(**json.loads(PROFILE.read_text()))
    records = []
    for spec in scenarios:
        captured = []

        def frame(m, d, row):
            if not captured and d.time >= 2.:
                rot = d.body('base').xmat.reshape(3, 3)
                features, _ = observe(spec, d.qpos[:2], np.arctan2(rot[1, 0], rot[0, 0]),
                                      np.linalg.norm(d.qvel[:2]), row['foot_contacts'] == 4
                                      and np.linalg.norm(d.qvel[:3]) < .04
                                      and np.linalg.norm(d.qvel[3:6]) < .2
                                      and rot[2, 2] > np.cos(np.deg2rad(12)), float(d.time))
                captured.append(features)

        rows, outcome = execute_obstacle_jump(spec, 'bar', config, frame_callback=frame)
        record = dict(source='measured_jump_outcome', features=captured[0],
                      label=int(outcome['success']), outcome=outcome)
        records.append(record)
        print(spec.name, outcome['success'], outcome['reason'], flush=True)
        np.savez_compressed(args.output/(spec.name+'.npz'),
                            qpos=np.array([r['qpos'] for r in rows]), qvel=np.array([r['qvel'] for r in rows]),
                            torque=np.array([r['torque'] for r in rows]), time=np.array([r['time'] for r in rows]))
    # Explicit contract negatives; these labels are not asserted motor outcomes.
    for i in range(80):
        base = dict(records[i % len(records)]['features'])
        feature, value = (('height', float(rng.uniform(.06, .5))),
                          ('target_speed', float(rng.uniform(.05, .5))),
                          ('landing_clearance', float(rng.uniform(-.3, .14))),
                          ('stable', 0.), ('depth', float(rng.uniform(.07, .3))),
                          ('alignment', float(rng.uniform(.15, 1.))),
                          ('front', float(rng.uniform(.5, 1.5))),
                          ('present', 0.))[i % 8]
        base[feature] = value
        records.append(dict(source='eligibility_negative_not_physics_outcome', features=base, label=0))
    weights, biases = JumpHead.seeded_arrays()
    metadata = dict(version='V4-observation-low-bar', features=list(NAMES), seed=20261017,
                    lif_steps=16, beta=.9, threshold=1., reset_each_packet=True,
                    v3_sha256=hashlib.sha256(V3.read_bytes()).hexdigest(),
                    jump_profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                    training_seed=17401, physics_trials=32, contract_negative_rows=80,
                    description='1-nearest LIF spike-count outcome association; no jump request input')
    temporary = JumpHead(weights, biases, np.zeros((1, len(biases))), np.zeros(1), metadata)
    bank = np.array([temporary.encode(r['features']) for r in records])
    model = JumpHead(weights, biases, bank, np.array([r['label'] for r in records]), metadata)
    model.save(args.model)
    with gzip.open(args.model.parent/'training_records.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')
    result = dict(model_checksum=model.checksum, successful_jumps=sum(r['label'] for r in records[:32]),
                  physics_trials=32, training_rows=len(records),
                  training_correct=sum((model.predict(r['features'])['proposed_skill'] == 'low_bar_jump') == bool(r['label'])
                                       for r in records))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
