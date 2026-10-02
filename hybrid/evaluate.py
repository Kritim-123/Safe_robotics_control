"""Paired-seed comparison of baseline, rule selector and SNN selector."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
from pathlib import Path

from hybrid.arena import SCENARIOS, STRESS_SCENARIOS, scenario
from hybrid.run import save_result
from hybrid.system import HybridSystem


def evaluate_one(job):
    name, seed, mode, output, sensing, replanning = job
    spec = replace(scenario(name, seed), **sensing, sensing_seed=seed)
    result = HybridSystem(spec, mode, replanning=replanning).run()
    result.summary['seed'] = seed
    save_result(result, Path(output)/f'{name}-seed{seed}-{mode}')
    return result.summary


def evaluate(output, names, seeds, modes, workers, sensing=None, replanning=False):
    names, seeds, modes = list(names), list(seeds), list(modes)
    for label, values in [('scenarios', names), ('seeds', seeds), ('modes', modes)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f'{label} must be nonempty and unique; duplicate jobs would overwrite records')
    if workers < 1 or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError('Use positive workers and nonnegative integer seeds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sensing = sensing or {}
    jobs = [(name, seed, mode, str(output/'runs'), sensing, replanning) for name in names for seed in seeds for mode in modes]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(evaluate_one, job): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({k: result[k] for k in ('scenario', 'seed', 'mode', 'success', 'reason', 'elapsed')}), flush=True)
    results.sort(key=lambda r: (r['scenario'], r['seed'], r['mode']))
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    aggregates = []
    for name in names:
        for mode in modes:
            group = [r for r in results if r['scenario'] == name and r['mode'] == mode]
            successes = [r for r in group if r['success']]
            aggregates.append(dict(scenario=name, mode=mode, trials=len(group), successes=len(successes),
                                   collisions=sum(r['collision'] for r in group),
                                   falls=sum(r['fall'] for r in group),
                                   mean_success_time=(sum(r['elapsed'] for r in successes)/len(successes) if successes else None),
                                   mean_final_distance=sum(r['final_distance'] for r in group)/len(group)))
    report = dict(seeds=list(seeds), modes=modes, scenarios=names, sensing=sensing, replanning=replanning, total_trials=len(results),
                  aggregates=aggregates, results=results)
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/hybrid/evaluation'))
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS + STRESS_SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[100, 101, 102])
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'rules', 'snn'), default=['baseline', 'rules', 'snn'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--replan', action='store_true', help='Enable experimental moving-hazard preemption')
    parser.add_argument('--position-noise', type=float, default=0.0, help='Obstacle position standard deviation (m)')
    parser.add_argument('--velocity-noise', type=float, default=0.0, help='Obstacle velocity standard deviation (m/s)')
    parser.add_argument('--sensor-delay', type=float, default=0.0, help='Delay of obstacle snapshots (s)')
    parser.add_argument('--sensor-filter-tau', type=float, default=0.0, help='Optional observation smoothing time constant (s)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    evaluate(args.output, args.scenarios, args.seeds, args.modes, args.workers,
             dict(position_noise=args.position_noise, velocity_noise=args.velocity_noise,
                  sensor_delay=args.sensor_delay, sensor_filter_tau=args.sensor_filter_tau), replanning=args.replan)


if __name__ == '__main__':
    main()
