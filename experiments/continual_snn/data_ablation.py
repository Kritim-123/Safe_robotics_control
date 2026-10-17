"""Isolate motion-teacher data with the original 16-neuron bank algorithm."""
import copy
import argparse
import hashlib
import json
from .models import ROOT,FrozenBankModel,load_model
from .scenes import cases
from .study import run_batch,save
from experiments.spike_pattern_skills.associations import AssociationMemory


class ExpandedLegacyBank(FrozenBankModel):
    name='legacy16_plus_motion_data_same_decoder'

    def predict(self,features,now=None):
        result=super().predict(features,now)
        references=result.pop('neighbors',[])
        # Only condense logging; retain the exact legacy decoding algorithm.
        result['matched_associations']=len(references)
        result['minimum_distance']=min((r['distance'] for r in references),default=None)
        return result


def main(stress=False):
    base=ROOT/'output/continual_snn';old=FrozenBankModel();expanded=ExpandedLegacyBank()
    source=base/'readout-comparison-v1/training_records.jsonl'
    examples=copy.deepcopy(old.memory.examples)
    for index,line in enumerate(source.read_text().splitlines()):
        row=json.loads(line)
        if row['source']!='stream':continue
        packet=expanded.encoder.encode(row['features'])
        examples.append(dict(context_id=f'motion/{row["group"]}/{index:05d}',
            spike_counts=packet['spike_counts'],skill='dynamic/'+row['label'].removeprefix('dynamic/'),
            label_source='causal_geometry_task_teacher_not_measured_counterfactual_outcome',
            source_group=row['group'],time=row['time']))
    expanded.memory=AssociationMemory(expanded.encoder.checksum,examples,
        neighbors=old.memory.neighbors,max_distance=old.memory.max_distance)
    expanded.checksum=expanded.encoder.checksum+':'+expanded.memory.checksum
    from .stress_scenes import stress_cases
    scenes=stress_cases(range(10200,10205)) if stress else cases('test',range(10100,10105))
    label='data-ablation-stress' if stress else 'data-ablation'
    paths=['experiments/continual_snn/data_ablation.py']+[
        'experiments/spike_pattern_skills/'+name+'.py' for name in (
            'encoder','terrain_encoding','dense_terrain_encoding','associations')]
    report=dict(frozen_models=True,no_hyperparameter_selection=True,
        original_examples=len(old.memory.examples),expanded_examples=len(examples),
        added_label_source='causal teacher observations only',
        encoder_checksum_unchanged=old.encoder.checksum==expanded.encoder.checksum,
        neighbors=old.memory.neighbors,max_distance=old.memory.max_distance,
        training_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        cases=[c.to_dict() for c in scenes],
        primary_evaluation_is_unchanged=True,
        status='Exploratory stress follow-up after regular data-ablation outcomes; models/hyperparameters unchanged' if stress else 'Frozen data-only comparison')
    save(base/(label+'-protocol-v1.json'),report)
    arms=[('original94','continuous',old),('motion_data_same_decoder','continuous',expanded),
        ('readout16','continuous',load_model(base/'readout-comparison-v1/models/candidate-00.npz')),
        ('primary80','continuous',load_model(ROOT/'docs/agents/continual_snn_v1/primary.npz'))]
    output=base/(label+'-v1');run_batch(output,scenes,arms)
    # Preserve the additional legacy dependency/wrapper source closure explicitly.
    import shutil
    for path in paths:
        target=output/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,target)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['source_sha256'].update(report['source_sha256'])
    manifest['additional_source_closure_recorded_after_run']=True
    save(output/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stress',action='store_true')
    main(parser.parse_args().stress)
