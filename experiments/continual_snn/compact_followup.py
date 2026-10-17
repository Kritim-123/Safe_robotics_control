"""Post-hoc matched alternative arm; no new environments or model tuning."""
import hashlib
import json
import shutil
from .models import ROOT
from .loading import load_selector
from .evaluate import BASE,DEADLINE,plan,verify
from .study import save,run_batch


def main():
    checkpoint=ROOT/'docs/agents/continual_snn_v1/compact_motion_bank16.npz'
    model=load_selector(checkpoint);model_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    paths=['experiments/continual_snn/'+name+'.py' for name in ('compact_bank','compact_followup','loading')]
    paths += ['experiments/spike_pattern_skills/'+name+'.py' for name in ('encoder','terrain_encoding','dense_terrain_encoding','associations')]
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    source=BASE/'compact-followup-source-v1';source.mkdir(exist_ok=False)
    for path in paths:
        destination=source/path;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,destination)
    save(BASE/'compact-followup-protocol-v1.json',dict(
        status='Post-hoc matched alternative on the original frozen cases; not independent extra environments',
        model_checksum=model.checksum,checkpoint_sha256=model_hash,source_sha256=hashes,
        primary_frozen_evaluation_unchanged=True,training_or_parameter_changes=False,
        source_bank_policy='Exact legacy decoder with added training-stream data; lossless multiplicity-preserving compaction',
        cases_and_sensors='Same eight batches as frozen-evaluation-v1.json',
        main_arms=['compact_once','compact_repeat'],other_arms=['compact_repeat'],deadline_epoch=DEADLINE))
    for phase,(scenes,_,sensor) in plan().items():
        verify()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=model_hash:raise ValueError('Compact checkpoint changed')
        for path,checksum in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=checksum:raise ValueError('Compact selector source changed')
        arms=[('compact_once','one_shot',model),('compact_repeat','continuous',model)] if phase=='main' else [('compact_repeat','continuous',model)]
        output=BASE/('compact-'+phase+'-v1');run_batch(output,scenes,arms,sensor=sensor,deadline=DEADLINE)
        manifest=json.loads((output/'manifest.json').read_text())
        manifest['additional_source_freeze']='compact-followup-protocol-v1.json and compact-followup-source-v1/'
        manifest['source_sha256'].update(hashes)
        for path in paths:
            target=output/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/path,target)
        save(output/'manifest.json',manifest)


if __name__=='__main__':main()
