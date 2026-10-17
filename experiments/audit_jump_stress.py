"""Recompute stress measurements from traces and distinguish pre-start failures."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.smooth_jump import ROOT, save


def audit(folder):
    folder = Path(folder).resolve()
    rows = [json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()]
    declared = json.loads((folder/'cases.json').read_text())
    config = json.loads((folder/'frozen_config.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    assert len(rows) == len(declared) == 120
    assert len({r['scenario']['name'] for r in rows}) == 120
    assert {r['scenario']['name'] for r in rows} == {c['scenario']['name'] for c in declared}
    for source, digest in manifest['sources'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    hashes = {}
    for row in rows:
        s, m = row['summary'], row['metrics']
        assert row['config'] == config
        trace_path = folder/row['scenario']['name']/'trace.npz'
        hashes[str(trace_path.relative_to(folder))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        matrix = np.load(trace_path)['values']
        assert np.isfinite(matrix).all()
        assert np.all(np.diff(matrix[:, 0]) > 0)
        assert abs(matrix[-1, 0]-s['elapsed']) < 1e-10
        assert s['max_torque_limit_ratio'] <= 1.+1e-10
        started = s['decision_observation'] is not None and s['eligibility']['jump']
        row['phase'] = 'initiated' if started else 'refused' if s['reason'] == 'failed initiation' else 'setup'
        assert (s['actuator_limit'] == 28.) == started
        if started:
            active = matrix[(matrix[:, 0] >= 2.) & (matrix[:, 0] < 4.5)]
            yaw = s['decision_observation']['yaw']
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            delta = matrix[-1, 1:3]-np.array(s['decision_observation']['position'])
            assert abs(delta@forward-m['forward_progress']) < 1e-9
            acc = float(np.max(np.linalg.norm(active[:, 7:10], axis=1)))
            slew = float(np.sqrt(np.mean(np.diff(active[:, 14:], axis=0)**2)))
            assert abs(acc-m['peak_base_acceleration']) < 1e-9
            assert abs(slew-m['torque_step_rms']) < 1e-9
            # Recompute continuous flight from every physics sample, independent
            # of the active-window landing-metric implementation.
            flags = matrix[matrix[:, 0] > s['decision_observation']['time'], 13].astype(bool)
            streak = longest = 0
            for flag in flags:
                streak = streak+1 if flag else 0
                longest = max(longest, streak)
            assert abs(longest*.002-s['longest_airtime']) < 1e-9
        else:
            # The stored displacement here is settling drift, never jump travel.
            assert not m['qualifies']
        if row['phase'] == 'setup':
            row['assessment'] = 'setup_failure'
        else:
            row['assessment'] = row['outcome']
    categories = ['forward_qualified', 'completed_short_or_backward', 'setup_failure',
                  'refused', 'execution_failure', 'controlled_interruption']
    groups = []
    for family in dict.fromkeys(row['family'] for row in rows):
        group = [row for row in rows if row['family'] == family]
        groups.append(dict(family=family, trials=len(group),
                           phases=dict(Counter(r['phase'] for r in group)),
                           assessment=dict(Counter(r['assessment'] for r in group))))
    totals = dict(Counter(r['assessment'] for r in rows))
    result = dict(trials=len(rows), phases=dict(Counter(r['phase'] for r in rows)),
                  assessment=totals, groups=groups,
                  trace_audit='120 finite traces, frozen sources/config, torque bounds; 85 initiated traces recomputed',
                  falls=sum(r['summary']['fall'] for r in rows),
                  collisions_by_phase=dict(Counter(r['phase'] for r in rows if r['summary']['collision'])),
                  traces_sha256=hashes)
    save(folder/'audit.json', result)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    left = np.zeros(len(groups))
    colors = ['#2a9d69', '#ddae3e', '#ba4747', '#788ba2', '#d77837', '#527ac4']
    labels = ['Forward hop qualified', 'Completed, too short', 'Setup failed before jump',
              'Initiation refused', 'Started, execution failed', 'Controlled interruption']
    for category, color, label in zip(categories, colors, labels):
        values = np.array([g['assessment'].get(category, 0) for g in groups])
        ax.barh([g['family'].replace('_', ' ') for g in groups], values,
                left=left, color=color, label=label)
        left += values
    ax.invert_yaxis(); ax.set_xlim(0, 10); ax.set_xticks(range(11))
    ax.set_xlabel('Cases (10 declared per family)')
    ax.set_title('Frozen forward jump: 120 MuJoCo conditions\nSetup failures and refusals are separate from launched jumps')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(folder/'outcomes.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(assessment=totals, phases=result['phases'],
                         collisions_by_phase=result['collisions_by_phase'], falls=result['falls']), indent=2))
    return rows, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    audit(args.input)
