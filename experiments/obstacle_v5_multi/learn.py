"""Separate measured-outcome bank extension; never used by evaluation at runtime."""
import argparse
from contextlib import redirect_stdout
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v5_refinement.model import RefinedHead, ROOT, PROFILE
from experiments.obstacle_v5_refinement.develop import teacher
from .run import course
from .runner import execute

ASSETS = ROOT/'docs/agents/obstacle_v5_multi'
MODEL = ASSETS/'primary.npz'


def training_courses():
    rng = np.random.default_rng(21701)
    return [course(count=2+i%3, spacing=float(rng.uniform(2.15, 2.6)),
                   height=float(rng.uniform(.02, .04)), friction=float(rng.uniform(.8, .9)),
                   first=float(rng.uniform(.8, 1.15))) for i in range(12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=MODEL)
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep physics evidence inside output/')
    folder.mkdir(parents=True, exist_ok=False)
    old = RefinedHead.load(); records = []; outcomes = []
    for i, spec in enumerate(training_courses()):
        trial = folder/f'train-{i:02d}'
        with (folder/f'train-{i:02d}.log').open('w') as log, redirect_stdout(log):
            result = execute(spec, trial, timeout=240., head=teacher(PROFILE))
        outcomes.append(result)
        packets = [json.loads(s) for s in (trial/'decisions.jsonl').read_text().splitlines()]
        successes = {r['target'] for r in result['jump_outcomes']}
        for entry in result['jump_entries']:
            for row in packets:
                if (row['eligible'] and row['phase'] == 'wait'
                        and entry['time']-.201 <= row['time'] <= entry['time']+1e-9):
                    records.append(dict(source='measured_per_jump_outcome', training_trial=i,
                                        target=entry['target'], observation_time=row['time'],
                                        actuation_time=entry['time'], features=row['jump_features'],
                                        label=int(entry['target'] in successes)))
        print(i, result['reason'], result['completed_jumps'], flush=True)
    if not records:
        raise ValueError('No measured associations collected')
    metadata = dict(old.metadata, version='V5-multi-count-outcomes', parent_checksum=old.checksum,
                    multi_training_seed=21701, multi_physics_trials=len(outcomes),
                    multi_measured_packet_rows=len(records),
                    multi_label_contract='Each jump verified separately; eligible packets within .201 s of actuation')
    model = RefinedHead(old.weights, old.biases,
                        np.vstack([old.bank, [old.encode(r['features']) for r in records]]),
                        np.r_[old.labels, [r['label'] for r in records]], metadata,
                        old.scales, old.centers)
    model.save(args.model)
    with gzip.open(args.model.with_name('training_records.jsonl.gz'), 'xt', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(r)+'\n' for r in records))
    with gzip.open(args.model.with_name('training_outcomes.jsonl.gz'), 'xt', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(r)+'\n' for r in outcomes))
    summary = dict(trials=len(outcomes), goals=sum(r['success'] for r in outcomes),
                   jumps=sum(r['completed_jumps'] for r in outcomes),
                   contacts=sum(r['collision'] for r in outcomes), falls=sum(r['fall'] for r in outcomes),
                   measured_rows=len(records), checksum=model.checksum)
    args.model.with_name('training_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
