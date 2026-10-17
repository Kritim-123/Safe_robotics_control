"""Rebuild both released association models from portable training-only data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
from .models import ROOT,BUNDLE
from .loading import load_selector
from .readout import SpikeReadout
from .compact_bank import CompactLegacyBank
from .prototypes import PrototypeMemory
from .study import save


def reproduce(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):raise ValueError('Reproduction output must stay inside project output')
    output.mkdir(parents=True,exist_ok=False)
    release=ROOT/'docs/agents/continual_snn_v1'
    manifest=json.loads((release/'reproduction_data.json').read_text())
    payloads={}
    for name,item in manifest['files'].items():
        compressed=(release/name).read_bytes();data=gzip.decompress(compressed)
        if (hashlib.sha256(compressed).hexdigest()!=item['compressed_sha256']
                or hashlib.sha256(data).hexdigest()!=item['source_sha256']):raise ValueError('Data checksum mismatch')
        payloads[name]=data
    records=[json.loads(line) for line in payloads['training_records.jsonl.gz'].decode().splitlines()]
    validation=[json.loads(line) for line in payloads['validation_records.jsonl.gz'].decode().splitlines()]
    if {r['group'] for r in records}&{r['group'] for r in validation}:raise AssertionError('Group leakage')
    reference=load_selector(release/'primary.npz')
    rebuilt=SpikeReadout(reference.encoder,alpha=.0001).fit([r['features'] for r in records],
        [r['label'] for r in records],[r['group'] for r in records])
    rebuilt.save(output/'readout80.npz')
    source=output/'source_associations.json';source.write_bytes(payloads['association_sources.json.gz'])
    encoder=DenseTerrainEncoder.load(BUNDLE/'dense_encoder.npz')
    original=AssociationMemory.load(source,encoder.checksum)
    compact=CompactLegacyBank.from_legacy(encoder,original);compact.save(output/'compact16.npz')
    reference_compact=load_selector(release/'compact_motion_bank16.npz')
    reference_prototype=load_selector(release/'association_memory.npz')
    prototype=PrototypeMemory(reference_prototype.encoder,neighbors=reference_prototype.neighbors,
        max_distance=reference_prototype.max_distance).fit(records)
    prototype.save(output/'prototype80.npz')
    report=dict(training_rows=len(records),training_groups=len({r['group'] for r in records}),
        validation_rows=len(validation),validation_groups=len({r['group'] for r in validation}),
        groups_disjoint=True,readout80_checksum_identical=rebuilt.checksum==reference.checksum,
        compact16_checksum_identical=compact.checksum==reference_compact.checksum,
        prototype80_checksum_identical=prototype.checksum==reference_prototype.checksum,
        frozen_encoder_arrays_loaded_from_published_checkpoint=True,
        evaluation_outcomes_or_parameters_used_for_retraining=False)
    save(output/'reproduction.json',report)
    if not all(report[k] for k in ('readout80_checksum_identical','compact16_checksum_identical','prototype80_checksum_identical')):
        raise AssertionError(report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    reproduce(parser.parse_args().output)
