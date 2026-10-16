"""Run and optionally publish the predeclared 100-situation recording study.

python -m experiments.spike_pattern_skills.study --output output/spike_pattern_skills/study-100
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .analyze import analyze
from .encoder import RandomLIFEncoder
from .observations import FEATURES
from .record import ROOT, capture_scenario
from .situations import FAMILIES, situations


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_counts(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = np.asarray([r['recording']['spike_counts'] for r in records])
    fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
    mesh = ax.imshow(counts, aspect='auto', cmap='viridis', vmin=0, vmax=16, interpolation='nearest')
    ax.set(xticks=range(16), xlabel='Output neuron ID',
           yticks=np.arange(4.5, 100, 10), yticklabels=[f.replace('_', ' ') for f in FAMILIES],
           title='Frozen random LIF encoder: spike counts in 100 MuJoCo situations\n10 variants per family; 16 internal encoding steps per observation')
    for boundary in np.arange(9.5, 99.5, 10):
        ax.axhline(boundary, color='white', linewidth=.6)
    fig.colorbar(mesh, ax=ax, label='Spikes in the observation window', ticks=[0, 4, 8, 12, 16])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def publish(output, destination):
    destination = Path(destination)
    if not destination.is_absolute():
        destination = ROOT/destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose an empty publication directory')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('model.npz', 'situations.csv', 'analysis.json', 'scenarios.json', 'counts_heatmap.png'):
        shutil.copyfile(output/name, destination/name)
    packed = gzip.compress((output/'records.jsonl').read_bytes(), mtime=0)
    (destination/'records.jsonl.gz').write_bytes(packed)
    manifest = dict(study=json.loads((output/'manifest.json').read_text()),
        raw_records_sha256=sha256(output/'records.jsonl'),
        archived_files_sha256={p.name: sha256(p) for p in sorted(destination.iterdir())})
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    # Checksums refer to exact bytes. Git must not rewrite line endings here.
    (destination/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n', encoding='ascii')


def run(output='output/spike_pattern_skills/study-100', model=None, study_seed=20261016, publish_to=None):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT/output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    cases = situations(study_seed)
    encoder = RandomLIFEncoder.initialize(42) if model is None else RandomLIFEncoder.load(
        Path(model) if Path(model).is_absolute() else ROOT/model)
    if encoder.metadata['pattern_neurons'] != 16 or encoder.metadata['window_steps'] != 16:
        raise ValueError('Study v1 requires 16 neurons and 16 internal steps')
    output.mkdir(parents=True, exist_ok=True)
    encoder.save(output/'model.npz')
    protocol = [dict(family=c.family, variant=c.variant, scene_seed=c.scene_seed,
                     jump_requested=c.jump_requested, scenario=c.scenario.to_dict()) for c in cases]
    (output/'scenarios.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    records, failures = [], []
    with (output/'records.jsonl').open('x', encoding='utf-8') as stream:
        for case in cases:
            try:
                observation = capture_scenario(case.scenario, case.jump_requested, case.scenario.name, case.scene_seed)
                observation['metadata'].update(study_family=case.family, variant=case.variant, scene_seed=case.scene_seed)
                record = encoder.record(observation)
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                records.append(record)
            except Exception as error:
                failures.append(dict(context_id=case.scenario.name, error_type=type(error).__name__, error=str(error)))
            if (case.variant+1) % 10 == 0:
                print(f'{case.family}: {len(records)} observations recorded, {len(failures)} failures', flush=True)
    (output/'failures.json').write_text(json.dumps(failures, indent=2)+'\n', encoding='utf-8')
    if failures or len(records) != 100:
        raise RuntimeError('Study incomplete; failed attempts retained in failures.json. No replacement cases were added.')
    loaded = RandomLIFEncoder.load(output/'model.npz')
    if records != [loaded.record(r['observation']) for r in records]:
        raise AssertionError('Reloaded encoder replay mismatch')
    analysis = analyze(records)
    (output/'analysis.json').write_text(json.dumps(analysis, indent=2)+'\n', encoding='utf-8')
    fields = ['context_id', 'family', 'variant', 'scene_seed', 'jump_requested', 'active_neuron_ids',
              'total_spikes', 'encoder_checksum'] + list(FEATURES) + [f'neuron_{i:02d}_count' for i in range(16)]
    # jump_requested is already a feature; do not emit a duplicate CSV column.
    fields = list(dict.fromkeys(fields))
    with (output/'situations.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for record in records:
            obs, firing = record['observation'], record['recording']
            writer.writerow(dict(context_id=obs['context_id'], family=obs['metadata']['study_family'],
                variant=obs['metadata']['variant'], scene_seed=obs['metadata']['scene_seed'],
                active_neuron_ids=json.dumps(firing['active_neuron_ids']),
                total_spikes=sum(firing['spike_counts']), encoder_checksum=encoder.checksum,
                **obs['features'], **{f'neuron_{i:02d}_count': c for i, c in enumerate(firing['spike_counts'])}))
    plot_counts(records, output/'counts_heatmap.png')
    sources = list(Path(__file__).parent.glob('*.py')) + [ROOT/p for p in (
        'hybrid/behavior_dataset.py', 'hybrid/arena.py', 'skills/dynamic.py',
        'skills/avoidance.py', 'controllers/walking.py', 'controllers/navigation.py')]
    import mujoco
    manifest = dict(date='2026-10-16', protocol='10 families x 10 variants, one settled observation each',
        study_seed=study_seed, families=list(FAMILIES), attempted=100, recorded=len(records), failed=len(failures),
        encoder_metadata=encoder.metadata, encoder_checksum=encoder.checksum, exact_reload_replay=True,
        obstacle_contacts=sum(r['observation']['metadata']['capture_episode']['collision'] for r in records),
        falls=sum(r['observation']['metadata']['capture_episode']['fall'] for r in records),
        max_standing_torque=max(r['observation']['metadata']['capture_episode']['peak_torque'] for r in records),
        unique_feature_vectors=len({tuple(r['observation']['features'][f] for f in FEATURES) for r in records}),
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(sources)},
        files_sha256={p.name: sha256(p) for p in sorted(output.iterdir())},
        runtime=dict(encoder.metadata['runtime'], mujoco=mujoco.__version__),
        skills_executed=0, labels_assigned=0,
        limitations='Ground-truth nearest-obstacle observations; one frame per scene; no task/skill accuracy measurement.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if publish_to is not None:
        publish(output, publish_to)
    print(json.dumps({k: v for k, v in analysis.items() if k not in ('neurons', 'family_summary', 'representations')}, indent=2))
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/spike_pattern_skills/study-100')
    parser.add_argument('--model')
    parser.add_argument('--study-seed', type=int, default=20261016)
    parser.add_argument('--publish-to', help='Also save a compact, complete archive suitable for Git')
    run(**vars(parser.parse_args()))
