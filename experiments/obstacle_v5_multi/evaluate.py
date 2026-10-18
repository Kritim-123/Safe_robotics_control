"""Frozen autonomous assessment on courses disjoint from collection seed."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v5_refinement.model import RefinedHead, ROOT
from .learn import MODEL, ASSETS
from .run import course
from .runner import execute


def assessment_courses():
    rng = np.random.default_rng(22701)
    return [course(count=2+i%3, spacing=float(rng.uniform(2.15, 2.55)),
                   height=float(rng.uniform(.02, .04)), friction=float(rng.uniform(.8, .9)),
                   first=float(rng.uniform(.8, 1.15))) for i in range(9)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); folder = args.output.resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep evidence inside output/')
    folder.mkdir(parents=True, exist_ok=False)
    # Freeze before any held-out outcome is available.
    sources = [*Path(__file__).parent.glob('*.py'), MODEL,
               ROOT/'docs/agents/obstacle_v5_refinement/profile.json']
    freeze = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (ASSETS/'assessment_freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    head = RefinedHead.load(MODEL); rows = []
    specs = assessment_courses()+[course(), course(spacing=2., height=.025), course(spacing=1.8, height=.025)]
    for i, spec in enumerate(specs):
        with (folder/f'case-{i:02d}.log').open('w') as log, redirect_stdout(log):
            result = execute(spec, folder/f'case-{i:02d}', head=head, timeout=240.)
        rows.append(result)
        print(i, len(spec.obstacles), result['reason'], result['completed_jumps'], flush=True)
    errors = []; packets = 0
    for i, row in enumerate(rows):
        decisions = [json.loads(s) for s in (folder/f'case-{i:02d}'/'decisions.jsonl').read_text().splitlines()]
        for p in decisions:
            packets += 1
            if head.predict(p['jump_features']) != p['jump_proposal']:
                errors.append([i, p['time'], 'neural replay mismatch'])
        for entry in row['jump_entries']:
            confirmations = [p for p in decisions if entry['time']-.101 <= p['time'] <= entry['time']+1e-8]
            if len(confirmations) != 2 or not all(p['eligible'] and p['jump_proposal']['proposed_skill']=='low_bar_jump' for p in confirmations):
                errors.append([i, entry['time'], 'unconfirmed jump'])
        if row['teacher_forced_collection'] or row['explicit_jump_requests'] or not row['torques_released']:
            errors.append([i, 'autonomy/release violation'])
    with gzip.open(ASSETS/'assessment_results.jsonl.gz', 'xt', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(r)+'\n' for r in rows))
    summary = dict(seed=22701, held_out_courses=9, additional_spacing_demos=3,
                   total_courses=len(rows), goals=sum(r['success'] for r in rows),
                   expected_jumps=sum(len(s.obstacles) for s in specs),
                   completed_jumps=sum(r['completed_jumps'] for r in rows),
                   contacts=sum(r['collision'] for r in rows), falls=sum(r['fall'] for r in rows),
                   reasons=[r['reason'] for r in rows],
                   neural_packets_replayed=packets, audit_errors=errors, head_checksum=head.checksum)
    (ASSETS/'assessment_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
