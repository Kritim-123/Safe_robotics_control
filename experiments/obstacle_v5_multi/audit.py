"""Independent replay of both heads, source hashes and continuous state traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v5_refinement.model import ROOT, V3
from experiments.terrain_goal.loading_v2 import load_any
from .learn import ASSETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=ASSETS)
    args = parser.parse_args()
    errors = []; checked = 0
    for file in ('obstacle_v5_refinement/frozen_selection.json', 'obstacle_v5/frozen_selection.json'):
        frozen = json.loads((ROOT/'docs/agents'/file).read_text())['files']
        for path, digest in frozen.items():
            checked += 1
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
                errors.append(['inherited hash', path])
    frozen = json.loads((args.artifacts/'assessment_freeze.json').read_text())
    for path, digest in frozen.items():
        checked += 1
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            errors.append(['assessment hash', path])
    packets = 0; episodes = 0; max_step = 0.; timelines = []
    for folder in sorted(args.batch.glob('case-*')):
        if not folder.is_dir():
            continue
        episodes += 1
        events = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        reset_times = [e['time'] for e in events if e.get('source')=='verified full crossing; fresh obstacle selection']
        base = load_any(V3); base.reset(); previous = -1.
        for line in (folder/'decisions.jsonl').read_text().splitlines():
            row = json.loads(line)
            if any(previous < t <= row['time'] for t in reset_times):
                base.reset()
            packets += 1
            if base.predict(row['v3_features'], now=row['time']) != row['v3_proposal']:
                errors.append([folder.name, row['time'], 'V3 replay mismatch'])
            previous = row['time']
        with np.load(folder/'trace.npz') as trace:
            values = trace['values']
        delta = np.linalg.norm(np.diff(values[:, 1:4], axis=0), axis=1)
        max_step = max(max_step, float(delta.max(initial=0.)))
        if not np.all(np.isfinite(values)) or delta.max(initial=0.) >= .02 or np.any(np.diff(values[:, 0]) <= 0):
            errors.append([folder.name, 'nonfinite/discontinuous physics trace'])
        timelines.append(dict(case=folder.name, events=events))
    result = dict(source_model_checks=checked, v3_packets_replayed=packets, episodes=episodes,
                  max_base_step_m=max_step, errors=errors)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    (args.artifacts/'skill_transitions.json').write_text(json.dumps(timelines, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit('Audit failed')


if __name__ == '__main__':
    main()
