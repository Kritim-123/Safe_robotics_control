"""Matched frozen-checkpoint retention study; artifacts include all failed runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from hybrid.arena import scenario, SCENARIOS, STRESS_SCENARIOS
from hybrid.system import HybridSystem
from hybrid.run import save_result

VARIANTS = {
    'baseline': dict(mode='baseline'),
    'rules': dict(mode='rules', expanded_skills=True),
    'legacy_snn': dict(mode='snn'),
    'reset': dict(mode='snn', memory='reset', expanded_skills=True),
    'retain': dict(mode='snn', memory='retain', expanded_skills=True),
}


def run_job(job):
    name, seed, variant, noisy, output = job
    spec = scenario(name, seed)
    if noisy:
        spec = replace(spec, position_noise=.04, velocity_noise=.01, sensor_delay=.3,
                       sensor_filter_tau=.2, sensing_seed=seed)
    result = HybridSystem(spec, replanning=True, **VARIANTS[variant]).run()
    row = result.summary
    row.update(seed=seed, variant=variant, noisy=noisy,
               choices=[e['skill'] for e in result.events if e['target'] == 'SKILL'],
               choice_times=[e['time'] for e in result.events if e['target'] == 'SKILL'])
    save_result(result, Path(output)/'runs'/f'{name}-{seed}-{variant}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/hybrid/memory-study'))
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS+STRESS_SCENARIOS,
                        default=['static', 'crossing', 'head_on', 'oscillating_gate', 'late_crossing'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[810, 811, 812])
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--noisy', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or any(s < 0 for s in args.seeds):
        parser.error('Positive workers and nonnegative seeds required')
    if any(len(v) != len(set(v)) for v in (args.scenarios, args.seeds, args.variants)):
        parser.error('Duplicate scenarios/seeds/variants would overwrite artifacts')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s, v, args.noisy, str(output)) for n in args.scenarios for s in args.seeds for v in args.variants]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('scenario','seed','variant','success','reason')}), flush=True)
    rows.sort(key=lambda r: (r['scenario'], r['seed'], r['variant']))
    pairs = []
    for name in args.scenarios:
        for seed in args.seeds:
            pair = {r['variant']: r for r in rows if r['scenario'] == name and r['seed'] == seed}
            if 'reset' in pair and 'retain' in pair:
                a, b = pair['reset'], pair['retain']
                pairs.append(dict(scenario=name, seed=seed,
                                  success_delta=int(b['success'])-int(a['success']),
                                  elapsed_delta=b['elapsed']-a['elapsed'],
                                  clearance_delta=b['min_footprint_clearance']-a['min_footprint_clearance'],
                                  changed_choices=a['choices'] != b['choices']))
    report = dict(protocol='Frozen four-action weights; matched reset/retain at sensing cadence; expanded mapped skills; replan enabled',
                  tau=.5, max_gap=.5, microsteps=16, beta=.9, seeds=args.seeds, noisy=args.noisy,
                  variants=VARIANTS, results=rows, pairs=pairs)
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
