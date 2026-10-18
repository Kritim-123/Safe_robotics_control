"""Full-episode motor development on disclosed known failures, never fresh tests."""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path

from hybrid.arena import Scenario, Obstacle
from skills.jump_profile import JumpConfig
from experiments.obstacle_v4.model import JumpHead, ROOT
from experiments.obstacle_v5.run import nominal
from .runner import execute
from .model import ASSETS


def saved_scene(folder):
    data = json.loads((Path(folder)/'summary.json').read_text())['scenario']
    data['obstacles'] = tuple(Obstacle(**o) for o in data['obstacles'])
    if data.get('terrain'):
        raise ValueError('This refinement is flat-ground only')
    data['terrain'] = ()
    return Scenario(**data)


def teacher(profile):
    old = JumpHead.load()
    metadata = dict(old.metadata, jump_profile_sha256=hashlib.sha256(Path(profile).read_bytes()).hexdigest())
    head = JumpHead(old.weights, old.biases, old.bank, old.labels, metadata)
    original = head.predict
    head.predict = lambda f: dict(original(f), proposed_skill='low_bar_jump', teacher_forced=True,
                                  reason='teacher_forced_training_proposal')
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    base = JumpConfig(**json.loads((ASSETS/'baseline_profile.json').read_text()))
    scenes = [nominal(), saved_scene(ROOT/'output/obstacle_v5/test-v1/v5-test-08'),
              saved_scene(ROOT/'output/obstacle_v5/training-v2/v5-train-15'),
              saved_scene(ROOT/'output/obstacle_v5/training-v2/v5-train-16')]
    configs = [base]
    for lean in (base.forward_lean-.04, base.forward_lean+.04, base.forward_lean+.08):
        for thrust_scale in (.95, 1.05):
            configs.append(replace(base, forward_lean=lean, thrust=base.thrust*thrust_scale))
    configs += [replace(base, flight=base.flight+.04), replace(base, flight=base.flight+.08),
                replace(base, absorption_thigh=1.15, absorption_calf=-2.3),
                replace(base, absorption_thigh=1.2, absorption_calf=-2.4)]
    if args.round == 2:
        base = replace(base, forward_lean=base.forward_lean+.04, thrust=base.thrust*.95)
        scenes = [scenes[1], scenes[0], scenes[2], scenes[3]]
        configs = [replace(base, absorption_thigh=t, absorption_calf=-2*t) for t in (.9, .95, .98)]
        configs += [replace(base, absorption=base.absorption+dt) for dt in (.04, .08, .12)]
        configs += [replace(base, recovery=t) for t in (.8, 1.)]
        configs += [replace(base, landing_kp=kp, landing_kd=kd) for kp, kd in ((50., 2.), (55., 2.4))]
        configs += [replace(base, absorption_thigh=.95, absorption_calf=-1.9,
                            absorption=base.absorption+.04, recovery=.8)]
    if args.round == 3:
        base = replace(base, forward_lean=base.forward_lean+.04, thrust=base.thrust*.95)
        hard_root = ROOT/'output/obstacle_v5_refinement/motor-confirmation-v1/candidate-02'
        scenes = [saved_scene(hard_root/'motor-confirm-00'), saved_scene(hard_root/'motor-confirm-06'),
                  scenes[1], scenes[0], scenes[2], scenes[3]]
        configs = [replace(base, absorption_thigh=t, absorption_calf=-2*t) for t in (.75, .8, .85, .88)]
        configs += [replace(base, forward_lean=base.forward_lean+lean,
                            absorption_thigh=t, absorption_calf=-2*t)
                    for lean in (.08, .12) for t in (.9, .95)]
        configs += [replace(base, thrust=base.thrust/.95*.9, absorption_thigh=t, absorption_calf=-2*t)
                    for t in (.9, .95)]
        configs += [replace(base, flight_calf=-2.65, flight_lean=-.1, absorption_thigh=t, absorption_calf=-2*t)
                    for t in (.9, .95)]
    all_rows = []; aggregates = []
    for i, config in enumerate(configs):
        folder = args.output/f'candidate-{i:02d}'; folder.mkdir()
        profile = folder/'profile.json'; profile.write_text(json.dumps(asdict(config), indent=2)+'\n')
        rows = []
        for spec in scenes:
            row = execute(spec, folder/spec.name, head=teacher(profile), profile_path=profile)
            rows.append(row); all_rows.append(dict(candidate=i, **row))
            (args.output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows))
            if args.round >= 2 and not row['success']:
                break  # Disclosed adaptive development screening, not a test score.
        agg = dict(candidate=i, completed=sum(r['success'] for r in rows), contacts=sum(r['collision'] for r in rows),
                   evaluated=len(rows), planned=len(scenes),
                   falls=sum(r['fall'] for r in rows),
                   min_margin=min((r['final_collision_geometry_margin'] or -1.) for r in rows), profile=str(profile.resolve()))
        aggregates.append(agg); (args.output/'aggregate.json').write_text(json.dumps(aggregates, indent=2)+'\n')
        print(json.dumps(agg), flush=True)


if __name__ == '__main__':
    main()
