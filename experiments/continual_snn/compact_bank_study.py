"""Verify legacy-bank compaction before exposing it as an optional selector."""
import hashlib
import json
import shutil
import time
import numpy as np
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
from .models import ROOT
from .data_ablation import ExpandedLegacyBank
from .compact_bank import CompactLegacyBank
from .scenes import make_case
from .stress_scenes import make_stress_case
from .study import save,run_batch


def main():
    base=ROOT/'output/continual_snn';folder=base/'compact-bank-build-v1';folder.mkdir(exist_ok=False)
    original=base/'data-ablation-v1/models'
    source=ExpandedLegacyBank()
    source.encoder=DenseTerrainEncoder.load(original/'motion_data_same_decoder_encoder.npz')
    source.memory=AssociationMemory.load(original/'motion_data_same_decoder_associations.json',source.encoder.checksum)
    source.checksum=source.encoder.checksum+':'+source.memory.checksum
    compact=CompactLegacyBank.from_legacy(source.encoder,source.memory)
    compact.save(folder/'compact16.npz');loaded=CompactLegacyBank.load(folder/'compact16.npz')
    scenes=[make_case('clear_stop',10500,'compact_parity'),make_case('run_jump_run',10501,'compact_parity'),
        make_case('wait_release',10502,'compact_parity'),make_stress_case('two_hops',10503,'compact_parity'),
        make_stress_case('oscillating_gate',10504,'compact_parity')]
    save(folder/'protocol.json',dict(source_memory_checksum=source.memory.checksum,compact_checksum=compact.checksum,
        encoder_unchanged=True,policy_or_hyperparameter_updates=False,
        goal='Preserve all legacy decisions including duplicate multiplicity, exact conflicts, distance ties and abstention',
        cases=[c.to_dict() for c in scenes]))
    queries={tuple(row) for row in source.memory.counts}
    validation=base/'readout-comparison-v1/validation_records.jsonl'
    for line in validation.read_text().splitlines():
        row=json.loads(line);queries.add(tuple(source.encoder.encode(row['features'])['spike_counts']))
    rng=np.random.default_rng(20261017)
    for _ in range(500):
        counts=compact.counts[int(rng.integers(len(compact.counts)))].astype(int)
        queries.add(tuple(np.clip(counts+rng.integers(-2,3,16),0,16)))
    queries.update(tuple(row) for row in rng.integers(0,17,(1000,16)))
    queries.update([(0,)*16,(16,)*16,tuple([0,16]*8)])
    violations=[];before=time.perf_counter()
    for query in sorted(queries):
        a,b=source.memory.decode(query),loaded.decode_counts(query)
        if (a['proposed_skill']!=b['proposed_skill'] or a['reason']!=b['reason']
                or len(a['neighbors'])!=b['matched_associations']):
            violations.append(dict(query=list(query),original=a,compact=b))
    report=dict(queries=len(queries),all_training_and_validation_signatures_included=True,
        random_and_locally_perturbed_queries_included=True,violations=violations,
        original_rows=len(source.memory.examples),unique_patterns=len(compact.counts),
        conflicting_patterns=int(sum(v not in (1,2,4) for v in compact.label_bits)),
        compaction_ratio=len(source.memory.examples)/len(compact.counts),
        seconds=time.perf_counter()-before)
    save(folder/'decoder-equivalence.json',report)
    if violations:raise AssertionError('Compaction changed decoding')
    output=base/'compact-bank-physical-parity-v1'
    run_batch(output,scenes,[('expanded_source','continuous',source),('compact16','continuous',loaded)],full_trace=True)
    comparisons=[]
    for case in scenes:
        with np.load(output/case.scenario.name/'expanded_source/trace.npz') as a,np.load(output/case.scenario.name/'compact16/trace.npz') as b:
            equal=np.array_equal(a['values'],b['values'])
        comparisons.append(dict(case=case.scenario.name,physics_trace_bit_identical=equal))
    save(folder/'physical-equivalence.json',comparisons)
    if not all(row['physics_trace_bit_identical'] for row in comparisons):raise AssertionError('Physical trajectories changed')
    # Capture the additional custom selector and legacy-source dependency closure.
    paths=['experiments/continual_snn/'+name+'.py' for name in ('compact_bank','compact_bank_study','loading','data_ablation')]
    paths += ['experiments/spike_pattern_skills/'+name+'.py' for name in ('encoder','terrain_encoding','dense_terrain_encoding','associations')]
    manifest=json.loads((output/'manifest.json').read_text())
    for path in paths:
        target=output/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
        manifest['source_sha256'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    save(output/'manifest.json',manifest)
    shutil.copyfile(folder/'compact16.npz',ROOT/'docs/agents/continual_snn_v1/compact_motion_bank16.npz')
    print(report,comparisons,flush=True)


if __name__=='__main__':main()
