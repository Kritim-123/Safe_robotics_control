"""Frozen close-spacing assessment, previous-world regressions and controls."""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from hybrid.arena import Obstacle
from experiments.obstacle_v5_multi.run import course
from experiments.obstacle_v5_multi.evaluate import assessment_courses
from experiments.obstacle_v5_multi.learn import MODEL
from experiments.obstacle_v5_refinement.model import ROOT, RefinedHead, PROFILE, V3
from .runner import execute


def fresh_courses():
    rng = np.random.default_rng(23701)
    return [course(count=2+i%3, spacing=float(rng.uniform(1.7, 2.1)),
                   height=float(rng.uniform(.02, .04)), friction=float(rng.uniform(.8, .9)),
                   first=float(rng.uniform(.8, 1.15))) for i in range(12)]


def control_courses():
    spec = course(count=2, spacing=1.8)
    return [replace(spec, obstacles=spec.obstacles+(Obstacle('block', (1.7, 0.), radius=.18),)),
            course(count=2, spacing=1.5),
            replace(spec, obstacles=(spec.obstacles[0], replace(spec.obstacles[1], height=.08)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path)
    args = parser.parse_args(); folder = args.output.resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep physics evidence inside output/')
    folder.mkdir(parents=True, exist_ok=False)
    assets = args.artifacts or folder/'artifacts'; assets.mkdir(parents=True, exist_ok=False)
    files = [*Path(__file__).parent.glob('*.py'), *ROOT.glob('experiments/obstacle_v5_multi/*.py'),
             ROOT/'tests/test_obstacle_v5_close.py', MODEL, PROFILE, V3]
    freeze = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (assets/'assessment_freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    head = RefinedHead.load(MODEL)
    entries = [('known', course(spacing=s, height=.025)) for s in (1.8, 2.)]
    entries += [('fresh', s) for s in fresh_courses()]
    entries += [('wide_regression', s) for s in assessment_courses()]
    entries += [('control', s) for s in control_courses()]
    results = []; errors = []; replayed = 0
    for i, (group, spec) in enumerate(entries):
        trial = folder/f'case-{i:02d}'
        with (folder/f'case-{i:02d}.log').open('w') as log, redirect_stdout(log):
            result = execute(spec, trial, timeout=240., head=head)
        result['group'] = group; results.append(result)
        packets = [json.loads(s) for s in (trial/'decisions.jsonl').read_text().splitlines()]
        for row in packets:
            replayed += 1
            if head.predict(row['jump_features']) != row['jump_proposal']:
                errors.append([i, row['time'], 'jump head replay'])
        for entry in result['jump_entries']:
            preceding = [r for r in packets if entry['time']-.101 <= r['time'] <= entry['time']+1e-8]
            if len(preceding)!=2 or not all(r['eligible'] and r['jump_proposal']['proposed_skill']=='low_bar_jump' for r in preceding):
                errors.append([i, entry['time'], 'unconfirmed jump'])
        if result['teacher_forced_collection'] or result['explicit_jump_requests'] or not result['torques_released']:
            errors.append([i, 'autonomy/release'])
        print(i, group, result['reason'], result['completed_jumps'], flush=True)
        (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary = dict(seed=23701, head_checksum=head.checksum, jump_packets_replayed=replayed, errors=errors, groups={})
    for group in ('known', 'fresh', 'wide_regression', 'control'):
        rows = [r for r in results if r['group']==group]
        summary['groups'][group] = dict(cases=len(rows), goals=sum(r['success'] for r in rows),
                expected_jumps=sum(len(r['scenario']['obstacles']) for r in rows),
                completed_jumps=sum(r['completed_jumps'] for r in rows), contacts=sum(r['collision'] for r in rows),
                falls=sum(r['fall'] for r in rows), reasons=[r['reason'] for r in rows])
    with gzip.open(assets/'assessment_results.jsonl.gz', 'xt', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(r)+'\n' for r in results))
    (assets/'assessment_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if errors:
        raise SystemExit('Replay audit failed')


if __name__ == '__main__':
    main()
