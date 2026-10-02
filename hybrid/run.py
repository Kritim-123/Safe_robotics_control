"""Run the Go2 hybrid controller and save reproducible experiment records."""
import argparse
import csv
import json
from pathlib import Path

from hybrid.arena import SCENARIOS, STRESS_SCENARIOS, Scenario, scenario
from hybrid.snn import DEFAULT_MODEL
from hybrid.system import HybridSystem


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', result.summary), ('events', result.events), ('scenario', result.scenario)]:
        (directory/f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    with (directory/'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
        if result.history:
            writer = csv.DictWriter(stream, fieldnames=list(result.history[0]))
            writer.writeheader()
            for row in result.history:
                writer.writerow({**row, 'obstacle_positions': json.dumps(row['obstacle_positions'])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS + STRESS_SCENARIOS, default='crossing')
    parser.add_argument('--config', type=Path, help='Custom scenario JSON, overrides --scenario/--seed')
    parser.add_argument('--mode', choices=('baseline', 'rules', 'snn'), default='snn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--replan', action='store_true', help='Experimental active-skill preemption for moving hazards')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, default=Path('output/hybrid/latest'))
    parser.add_argument('--video', type=Path, help='Optional MP4 recorded offscreen at 4x playback')
    args = parser.parse_args()
    recorder = None
    if args.video:
        from hybrid.recording import VideoRecorder
        recorder = VideoRecorder(args.video)
    try:
        spec = Scenario.load(args.config) if args.config else scenario(args.scenario, args.seed)
        result = HybridSystem(spec, args.mode, args.model, replanning=args.replan).run(
            viewer=not args.headless, record_callback=recorder)
        if recorder is not None:
            recorder.finish(result.summary)
    finally:
        if recorder is not None:
            recorder.close()
    save_result(result, args.output)
    print(json.dumps(result.summary, indent=2, allow_nan=False))
    raise SystemExit(0 if result.summary['success'] else 1)


if __name__ == '__main__':
    main()
