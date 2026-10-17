"""Audit frozen-policy evidence and publish the explicit skill availability list.

Run from the project root after reviewing the reports:
python -m hybrid.audit_pretrained_velocity
This uses Torch only to compare imported graphs against NumPy inference.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from skills.pretrained_velocity import numeric_forward, POLICY_ROOT, COMMANDS, execute_pretrained_skill

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
report = json.loads((ROOT/'output/pretrained_velocity/final/evaluation.json').read_text())
development = []
for policy in ('robot_lab', 'himloco'):
    for skill in COMMANDS:
        _, result = execute_pretrained_skill(policy, skill,
            output=ROOT/'output/pretrained_velocity/audit-development'/policy/skill)
        development.append(result)
        print('development verification', policy, skill, result['success'], flush=True)
validation = dict(date='2026-10-16', conditions='4 s commands; yaw .2/-.2/.4 rad and friction .7/.9/.65; original project dynamics',
    thresholds=dict(forward_reverse_m=.25, lateral_m=.2, turn_rad=.4, cross_axis_drift_m=.3,
        final_linear_speed_m_s=.04, final_angular_speed_rad_s=.2, recovery_dwell_s=.8),
    evaluated_contexts=[6200, 6201, 6202], policy_training='Frozen upstream checkpoints; no local training',
    raw_report='output/pretrained_velocity/final/evaluation.json',
    development_report='output/pretrained_velocity/audit-development/',
    raw_report_sha256=hashlib.sha256((ROOT/'output/pretrained_velocity/final/evaluation.json').read_bytes()).hexdigest(),
    source_sha256=report['source_sha256'], skills={}, actual_observation_parity={})
for policy in ('robot_lab', 'himloco'):
    folder = POLICY_ROOT/policy
    manifest = json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'model.npz', allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    original = torch.jit.load(str(folder/'source_policy.pt'), map_location='cpu').eval()
    observations = []
    for p in sorted((ROOT/'output/pretrained_velocity/final'/policy).glob('*/*/inference.npz')):
        with np.load(p, allow_pickle=False) as archive:
            observations.extend(archive['inputs'][::20])
    samples = np.asarray(observations, dtype=np.float32)
    with torch.inference_mode():
        reference = original(torch.from_numpy(samples)).numpy()
    converted = np.stack([numeric_forward(weights, policy, x) for x in samples])
    np.testing.assert_allclose(converted, reference, rtol=2e-5, atol=2e-5)
    validation['actual_observation_parity'][policy] = dict(samples=len(samples),
        max_absolute_error=float(np.max(np.abs(converted-reference))))
    for skill in COMMANDS:
        runs = [r for r in report['results'] if r['policy'] == policy and r['skill'] == skill]
        dev = next(r for r in development if r['policy'] == policy and r['skill'] == skill)
        passed = sum(r['success'] for r in runs)
        validation['skills'][f'pretrained/{policy}/{skill}'] = dict(enabled=passed == 3 and dev['success'],
            successes=passed, attempts=len(runs), development_success=dev['success'],
            source_policy_sha256=manifest['artifacts']['source_policy.pt'], export_sha256=manifest['artifacts']['model.npz'],
            results=[{k: r[k] for k in ('context_id', 'success', 'reason', 'signed_progress', 'cross_axis_drift',
                'elapsed', 'recovered', 'collision', 'fall', 'peak_torque', 'max_torque_limit_ratio',
                'saturation_fraction', 'final_speed', 'final_angular_speed', 'max_tilt_deg')} for r in runs])
(POLICY_ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8')
print(json.dumps(validation['actual_observation_parity'], indent=2))
