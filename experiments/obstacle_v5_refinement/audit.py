"""Recompute every saved neural proposal and verify model/source provenance."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.obstacle_v5.model import WalkUpHead
from experiments.terrain_goal.loading_v2 import load_any
from .model import ROOT, ASSETS, V3, RefinedHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads((ASSETS/'frozen_selection.json').read_text())
    mismatches = [p for p, digest in freeze['files'].items()
                  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    inherited = json.loads((ROOT/'docs/agents/obstacle_v5/frozen_selection.json').read_text())
    inherited_bad = [p for p, digest in inherited['files'].items()
                     if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    rows = [json.loads(s) for s in (args.batch/'results.jsonl').read_text().splitlines()]
    errors = []; counts = {'baseline':0, 'refined':0}; v3_count = 0
    for row in rows:
        arm = row['arm']; folder = args.batch/row['case']/arm
        head = WalkUpHead.load() if arm == 'baseline' else RefinedHead.load()
        base = load_any(V3); base.reset()
        if row['head_checksum'] != head.checksum:
            errors.append([row['case'], 'head checksum'])
        if row['teacher_forced_collection'] or row['explicit_jump_requests'] or not row['torques_released']:
            errors.append([row['case'], 'execution contract'])
        for line in (folder/'decisions.jsonl').read_text().splitlines():
            packet = json.loads(line)
            counts[arm] += 1; v3_count += 1
            if head.predict(packet['jump_features']) != packet['jump_proposal']:
                errors.append([row['case'], packet['time'], arm])
            if base.predict(packet['v3_features'], now=packet['time']) != packet['v3_proposal']:
                errors.append([row['case'], packet['time'], 'V3'])
    paired_world_mismatches = []
    for name in {r['case'] for r in rows}:
        pair = [r for r in rows if r['case'] == name]
        if len(pair) == 2 and pair[0]['scenario'] != pair[1]['scenario']:
            paired_world_mismatches.append(name)
    result = dict(source_model_checks=len(freeze['files']), source_model_mismatches=mismatches,
                  inherited_v5_checks=len(inherited['files']), inherited_v5_mismatches=inherited_bad,
                  packet_replays=counts, v3_packet_replays=v3_count, decision_errors=errors,
                  paired_world_mismatches=paired_world_mismatches,
                  max_actuator_limit_ratio=max(r['max_torque_limit_ratio'] for r in rows),
                  episodes=len(rows))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if mismatches or inherited_bad or errors or paired_world_mismatches:
        raise SystemExit('Audit failed')


if __name__ == '__main__':
    main()
