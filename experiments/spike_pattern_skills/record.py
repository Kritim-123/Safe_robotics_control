"""Capture settled MuJoCo observations, or encode named observations from JSON.

python -m experiments.spike_pattern_skills.record
No random network output is used to move the robot or select a skill.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .encoder import RandomLIFEncoder
from .observations import observation_from_simulation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('config.example.json')


def capture_observation(family, simulation_seed):
    from hybrid.behavior_dataset import make_context
    spec, jump_requested = make_context(family, simulation_seed)
    return capture_scenario(spec, jump_requested, f'{family}-{simulation_seed}', simulation_seed)


def capture_scenario(spec, jump_requested, context_id, simulation_seed):
    """Record a current observation after standing; abstain before skill entry."""
    from skills.dynamic import execute_dynamic_skill
    spec.validate()
    captured = []
    def capture_only(observation, eligibility):
        item = observation_from_simulation(spec, observation, jump_requested,
            observation_id=f'{context_id}-t{observation["time"]:.3f}', context_id=context_id)
        item['metadata']['eligibility_at_capture'] = eligibility
        captured.append(item)
        return None  # Intentional abstention: end capture before any skill runs.
    _, summary = execute_dynamic_skill(None, spec, selector=capture_only, seed=simulation_seed)
    if not captured:
        raise RuntimeError(f'No observation captured: {context_id}: {summary["reason"]}')
    captured[0]['metadata']['capture_episode'] = dict(
        termination=summary['reason'], capture_only=True, skill_executed=False,
        collision=summary['collision'], fall=summary['fall'], peak_torque=summary['max_torque'])
    return captured[0]


def run(output='output/spike_pattern_skills/latest', config=DEFAULT_CONFIG, observations=None, model=None):
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = ROOT/config_path
    settings = json.loads(config_path.read_text(encoding='utf-8'))
    if settings.get('schema_version') != 1 or settings.get('status') != 'recording_v1':
        raise ValueError('Unsupported or draft configuration')
    from .observations import FEATURES, UNITS, SCALES
    if settings['observation'] != dict(source='mujoco_state_initially', feature_names=list(FEATURES),
                                       feature_units=list(UNITS), normalization='divide_fixed_scales_then_clip',
                                       scales=list(SCALES), clip=3.):
        raise ValueError('Configuration observation contract does not match v1')
    c = settings['encoder']
    required = dict(neuron_model='LIF', weights='seeded_random_then_frozen',
                    initialization_distribution='normal_weights_std_1_over_sqrt_inputs_biases_std_0.1',
                    reset_between_observations=True, save_actual_weights_and_biases=True)
    if any(c.get(k) != v for k, v in required.items()):
        raise ValueError('Configuration encoder contract does not match v1')
    if settings['decoder'].get('learn_encoder_weights') is not False:
        raise ValueError('This milestone requires a frozen random encoder')
    if settings.get('recording') != dict(save_binary_raster=True, save_spike_counts=True,
                                         pattern='sorted_unique_active_neuron_ids', save_encoder_checksum=True):
        raise ValueError('Recording v1 requires raster, counts, active IDs and encoder checksum')
    if type(settings.get('simulation_seed')) is not int or settings['simulation_seed'] < 0:
        raise ValueError('Simulation seed must be a nonnegative integer')
    encoder = (RandomLIFEncoder.load(Path(model) if Path(model).is_absolute() else ROOT/model)
               if model is not None else RandomLIFEncoder.initialize(
                   settings['network_seed'], c['pattern_neurons'], c['window_steps'],
                   c['beta_per_step'], c['threshold']))
    for field in ('pattern_neurons', 'window_steps', 'beta_per_step', 'threshold', 'reset_between_observations'):
        if c[field] != encoder.metadata[field]:
            raise ValueError('Loaded model/config mismatch: '+field)
    if settings['network_seed'] != encoder.metadata['seed']:
        raise ValueError('Loaded model/config seed mismatch')
    output = Path(output)
    if not output.is_absolute():
        output = ROOT/output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory; existing recordings are preserved')
    output.mkdir(parents=True, exist_ok=True)
    if observations is None:
        entries = [capture_observation(family, settings['simulation_seed'])
                   for family in ('clear', 'far_obstacle', 'blocked', 'crossing', 'jump_clear', 'jump_restricted')]
    else:
        path = Path(observations)
        entries = json.loads((path if path.is_absolute() else ROOT/path).read_text(encoding='utf-8'))
        if not isinstance(entries, list) or not entries:
            raise ValueError('Observation JSON must contain a nonempty list')
    ids = [e['observation_id'] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate observation IDs')
    records = [encoder.record(e) for e in entries]
    encoder.save(output/'model.npz')
    reloaded = RandomLIFEncoder.load(output/'model.npz')
    repeated = [reloaded.record(e) for e in entries]
    if repeated != records or [encoder.record(e) for e in entries] != records:
        raise AssertionError('Save/load or repeated-inference mismatch')
    (output/'config.json').write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
    (output/'observations.json').write_text(json.dumps(entries, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'records.jsonl').open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    patterns = {}
    for record in records:
        key = ','.join(map(str, record['recording']['active_neuron_ids']))
        patterns.setdefault(key, []).append(record['observation']['observation_id'])
    source_files = list(Path(__file__).parent.glob('*.py'))
    runtime = dict(encoder.metadata['runtime'])
    if any(e['source'] == 'mujoco_state' for e in entries):
        import mujoco
        runtime['mujoco'] = mujoco.__version__
        source_files.extend(ROOT/name for name in ('hybrid/behavior_dataset.py', 'hybrid/arena.py',
                            'skills/dynamic.py', 'skills/avoidance.py', 'controllers/walking.py',
                            'controllers/navigation.py'))
    summary = dict(schema_version=1, observations=len(records), encoder_checksum=encoder.checksum,
                   seed=encoder.metadata['seed'], unique_active_id_patterns=len(patterns),
                   shared_patterns={k: ids for k, ids in patterns.items() if len(ids) > 1},
                   silent_observations=sum(r['recording']['silent'] for r in records),
                   repeated_inference_exact=True, save_load_exact=True,
                   skills_selected=0, skill_labels_assigned=0,
                   sources=sorted({e['source'] for e in entries}),
                   runtime=runtime,
                   file_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                for name in ('model.npz', 'records.jsonl', 'config.json', 'observations.json')},
                   source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(source_files)})
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/spike_pattern_skills/latest')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--observations', help='JSON list of named observations instead of MuJoCo capture')
    parser.add_argument('--model', help='Replay a saved numeric encoder archive')
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
