"""Known-case regression, fresh paired assessment and finite live refined V5."""
import argparse
import json
from pathlib import Path
import time

from experiments.obstacle_v5.run import cases as old_cases, controls, nominal
from experiments.obstacle_v5.runner import execute as original_execute
from .model import ROOT
from .runner import execute
from .train import scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('demo', 'regression', 'validation', 'fresh'), default='demo')
    parser.add_argument('--scene', choices=('nominal', 'former-contact'), default='former-contact')
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--viewer', action='store_true'); parser.add_argument('--video', action='store_true')
    parser.add_argument('--viewer-speed', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        raise ValueError('Count must be 1 to 100')
    folder = (args.output or ROOT/'output/obstacle_v5_refinement'/('demo-'+str(time.time_ns()))).resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep outputs inside project output/')
    if args.phase == 'demo':
        spec = nominal() if args.scene == 'nominal' else old_cases('test', 20)[8]
        print(json.dumps(execute(spec, folder, viewer=args.viewer, video=args.video,
                                 viewer_speed=args.viewer_speed), indent=2))
        return
    folder.mkdir(parents=True, exist_ok=False)
    cases = old_cases('test', 20) if args.phase == 'regression' else scenes(18601 if args.phase == 'validation' else 19601,
                                                                         args.count, args.phase)
    cases += controls()
    (folder/'cases.json').write_text(json.dumps([s.to_dict() for s in cases], indent=2)+'\n')
    rows = []
    for spec in cases:
        for arm, runner in ([('baseline', original_execute), ('refined', execute)] if args.phase == 'fresh' else [('refined', execute)]):
            row = dict(arm=arm, **runner(spec, folder/spec.name/arm))
            rows.append(row)
            (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            print(arm, spec.name, row['reason'], row['success'], flush=True)
    positive_names = {s.name for s in cases[:-7]}
    result = {}
    for arm in sorted({r['arm'] for r in rows}):
        positives = [r for r in rows if r['arm'] == arm and r['case'] in positive_names]
        negatives = [r for r in rows if r['arm'] == arm and r['case'] not in positive_names and 'clear' not in r['case']]
        clear = next(r for r in rows if r['arm'] == arm and 'control-clear' in r['case'])
        result[arm] = dict(cases=len(positives), goals=sum(r['success'] for r in positives),
                           crossings=sum(r['jump_completed'] for r in positives),
                           contacts=sum(r['collision'] for r in positives), falls=sum(r['fall'] for r in positives),
                           refusals=sum(r['reason'] == 'refused' for r in positives),
                           overshoots=sum(r['reason'] == 'launch approach overshoot' for r in positives),
                           negative_jumps=sum(r['jump_initiated'] for r in negatives),
                           negative_contacts=sum(r['collision'] for r in negatives),
                           clear_success=clear['success'], clear_jump=clear['jump_initiated'])
    if args.phase == 'fresh':
        paired = {s.name: {r['arm']:r['success'] for r in rows if r['case'] == s.name} for s in cases[:-7]}
        result['paired'] = dict(gained=sum(not p['baseline'] and p['refined'] for p in paired.values()),
                                lost=sum(p['baseline'] and not p['refined'] for p in paired.values()))
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
