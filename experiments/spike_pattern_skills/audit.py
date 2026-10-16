"""Independently verify the published 100-case archive and LIF arithmetic."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .encoder import RandomLIFEncoder
from .observations import FEATURES
from .record import ROOT


def audit(archive='docs/agents/spike_pattern_100'):
    folder = Path(archive)
    if not folder.is_absolute():
        folder = ROOT/folder
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['archived_files_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived file checksum mismatch: '+name)
    raw = gzip.decompress((folder/'records.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest['raw_records_sha256']:
        raise ValueError('Uncompressed records checksum mismatch')
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    if len(records) != 100 or len({r['observation']['context_id'] for r in records}) != 100:
        raise ValueError('Expected exactly 100 unique context records')
    model = RandomLIFEncoder.load(folder/'model.npz')
    with (folder/'situations.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(records):
        raise ValueError('CSV/record count mismatch')
    cells = 0
    for record, row in zip(records, csv_rows):
        observation, firing = record['observation'], record['recording']
        if model.record(observation) != record:
            raise ValueError('Exact saved-model replay mismatch')
        if observation['context_id'] != row['context_id']:
            raise ValueError('CSV/record ordering mismatch')
        # A scalar reference checks every neuron/time-step independently of the
        # vectorized encoder, including its reset state and subtractive reset.
        inputs = [max(-3., min(3., observation['features'][name]/float(model.scales[i])))
                  for i, name in enumerate(FEATURES)]
        for neuron in range(model.metadata['pattern_neurons']):
            current = sum(float(w)*x for w, x in zip(model.weights[neuron], inputs))+float(model.biases[neuron])
            membrane, count = 0., 0
            for step in range(model.metadata['window_steps']):
                before = model.metadata['beta_per_step']*membrane+current
                spike = int(before >= model.metadata['threshold'])
                membrane = before-model.metadata['threshold']*spike
                if (firing['output_raster'][step][neuron] != spike
                        or abs(firing['membrane_before_reset'][step][neuron]-before) > 1e-11
                        or abs(firing['membrane_after_reset'][step][neuron]-membrane) > 1e-11):
                    raise ValueError('Independent scalar LIF reference mismatch')
                count += spike; cells += 1
            if firing['spike_counts'][neuron] != count or int(row[f'neuron_{neuron:02d}_count']) != count:
                raise ValueError('Recorded/CSV spike count mismatch')
        if firing['active_neuron_ids'] != [i for i, n in enumerate(firing['spike_counts']) if n]:
            raise ValueError('Active-ID/count mismatch')
    analysis = json.loads((folder/'analysis.json').read_text())
    for name, key in (('active_ids', 'active_neuron_ids'), ('spike_counts', 'spike_counts'), ('raster', 'output_raster')):
        signatures = [json.dumps(r['recording'][key]) for r in records]
        unique = len(set(signatures))
        pairs = sum(signatures[i] == signatures[j] for i in range(100) for j in range(i))
        if (analysis['representations'][name]['unique_patterns'] != unique
                or analysis['representations'][name]['shared_pairs'] != pairs):
            raise ValueError('Independent representation-collision count mismatch')
    return dict(records=100, exact_model_replays=100, scalar_neuron_steps_checked=cells,
                archived_file_hashes_checked=len(manifest['archived_files_sha256']),
                csv_counts_match=True, collision_counts_match=True, encoder_checksum=model.checksum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='docs/agents/spike_pattern_100')
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
