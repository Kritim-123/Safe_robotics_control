"""Sequential, frozen-command evaluation of external Go2 velocity skills."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

from hybrid.arena import Scenario
from skills.pretrained_velocity import COMMANDS, POLICY_ROOT, execute_pretrained_skill

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = ((6200, .2, .7), (6201, -.2, .9), (6202, .4, .65))


def evaluate(output=ROOT/'output/pretrained_velocity/holdout'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for policy in ('robot_lab', 'himloco'):
        for context_id, yaw, friction in CONTEXTS:
            for skill in COMMANDS:
                spec = Scenario(f'external_{context_id}', goal=(3., 0.),
                                bounds=(-4., 6., -4., 4.), start_yaw=yaw, friction=friction)
                _, summary = execute_pretrained_skill(policy, skill, scenario=spec,
                                                      output=output/policy/str(context_id)/skill)
                summary['context_id'] = context_id
                results.append(summary)
                print(policy, context_id, skill, summary['success'], summary['reason'], flush=True)
    fingerprints = {file: hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
                    for file in ('skills/pretrained_velocity.py', 'controllers/walking.py',
                                 'hybrid/arena.py', 'hybrid/evaluate_pretrained_velocity.py')}
    report = dict(date='2026-10-16', contexts=CONTEXTS, source_sha256=fingerprints,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__),
                  results=results, success=sum(r['success'] for r in results), attempts=len(results),
                  note='Contexts vary yaw/friction; IDs record provenance and do not inject randomness.')
    (output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    # Availability is published only by a separate audit after reviewing the
    # complete report and tests. Running evaluation cannot enable a policy.
    print(json.dumps(dict(success=report['success'], attempts=len(results))))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/pretrained_velocity/holdout')
    evaluate(**vars(parser.parse_args()))
