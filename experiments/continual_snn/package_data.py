"""Portable, deterministic compressed training/provenance records for release."""
import gzip
import hashlib
import json
from .models import ROOT
from .study import save


def main():
    folder=ROOT/'docs/agents/continual_snn_v1'
    sources={
        'training_records.jsonl.gz':ROOT/'output/continual_snn/predictive-comparison-v1/training_records.jsonl',
        'validation_records.jsonl.gz':ROOT/'output/continual_snn/predictive-comparison-v1/validation_records.jsonl',
        'association_sources.json.gz':ROOT/'output/continual_snn/data-ablation-v1/models/motion_data_same_decoder_associations.json'}
    manifest={}
    for name,source in sources.items():
        data=source.read_bytes();target=folder/name
        with target.open('xb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:compressed.write(data)
        packed=target.read_bytes()
        if gzip.decompress(packed)!=data:raise AssertionError('Archive did not preserve source bytes')
        manifest[name]=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(data).hexdigest(),
            compressed_sha256=hashlib.sha256(packed).hexdigest(),bytes=len(data),compressed_bytes=len(packed))
    save(folder/'reproduction_data.json',dict(files=manifest,
        training_labels='Historical measured context outcomes and separate causal task/geometry teacher requests; source fields preserved',
        counts_are_not_uninterrupted_episode_spike_accumulations=True,
        final_evaluation_rows_used_for_training=0))
    print({name:item['compressed_bytes'] for name,item in manifest.items()},flush=True)


if __name__=='__main__':main()
