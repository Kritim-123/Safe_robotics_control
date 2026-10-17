"""Portable training-only release and exact model reproduction."""
import argparse,gzip,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from .encoder import TaskEncoder
from .models import Selector
from .factored import FactoredGoalSelector,fit_count_examples
from .assessment import primary_config
from dataclasses import asdict


def minimal_records(rows):
    e=TaskEncoder('goal_lattice64');result=[]
    for index,r in enumerate(rows):
        f=r['features']
        if not f.get('goal_mode') or not f['task_progress'] or not r.get('run_eligible',f['predicted_run_clearance']>.15):continue
        if np.hypot(f['goal_forward'],f['goal_left'])>.3 or r['label'] not in ('run','wait'):continue
        result.append(dict(group=r['group'],goal_forward=f['goal_forward'],goal_left=f['goal_left'],
            stop=r['label']=='wait',count_sum=int(sum(e.encode(f)['spike_counts'][32:])),source_training_row=index))
    return result


def reproduce(release,output):
    output.mkdir(parents=True,exist_ok=False)
    with gzip.open(release/'base_training_records.jsonl.gz','rt') as stream:base_rows=[json.loads(s) for s in stream]
    base=Selector(TaskEncoder('factor32'),'ridge',alpha=.001).fit(base_rows)
    reference=FactoredGoalSelector.load(release/'primary.npz')
    if base.checksum!=reference.base.checksum:raise AssertionError('Base weights did not reproduce')
    with gzip.open(release/'goal_count_training_records.jsonl.gz','rt') as stream:rows=[json.loads(s) for s in stream]
    encoder=TaskEncoder('goal_lattice64');examples=[]
    for r in rows:
        f={n:0. for n in encoder.feature_names}
        f.update(goal_mode=1.,goal_forward=r['goal_forward'],goal_left=r['goal_left'],task_progress=1.,
            predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
        actual=int(sum(encoder.encode(f)['spike_counts'][32:]))
        if actual!=r['count_sum']:raise AssertionError('Published goal-count encoding changed')
        examples.append((actual,r['stop'],r['group']))
    model,report=fit_count_examples(base,examples);model.save(output/'primary.npz')
    if model.checksum!=reference.checksum:raise AssertionError('Final goal model did not reproduce')
    save(output/'reproduction.json',dict(base_checksum=base.checksum,model_checksum=model.checksum,
        same_model_checksum=True,goal_count_rows=len(rows),base_rows=len(base_rows),threshold=model.threshold,
        byte_identical=(output/'primary.npz').read_bytes()==(release/'primary.npz').read_bytes()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=ROOT/'docs/agents/terrain_goal_v1')
    p.add_argument('--reproduce',type=Path)
    a=p.parse_args()
    if a.reproduce:return reproduce(a.release,a.reproduce)
    a.release.mkdir(parents=True,exist_ok=False)
    raw=ROOT/'output/terrain_goal';source=raw/'factored-goal-teacher-v4/factored.npz'
    model=FactoredGoalSelector.load(source);shutil.copyfile(source,a.release/'primary.npz')
    base_path=raw/'fit-teacher-v2/training_records.jsonl'
    payload=base_path.read_bytes();(a.release/'base_training_records.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
    rows=[json.loads(s) for s in (raw/'fit-goals-teacher-v4/training_records.jsonl').read_text().splitlines()]
    minimal=minimal_records(rows)
    payload=(''.join(json.dumps(r,allow_nan=False)+'\n' for r in minimal)).encode()
    (a.release/'goal_count_training_records.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
    if len(minimal)!=model.training_rows:raise AssertionError('Release data row mismatch')
    source_paths=list((ROOT/'experiments/terrain_goal').glob('*.py'))
    source_paths+=list((ROOT/'experiments/continual_snn').glob('*.py'))
    source_paths+=[ROOT/p for p in ('controllers/navigation.py','controllers/walking.py','skills/dynamic.py','skills/jump_profile.py','skills/avoidance.py','hybrid/arena.py','hybrid/terrain.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    save(a.release/'frozen_selection.json',dict(frozen_epoch=time.time(),model_checksum=model.checksum,
        base_checksum=model.base.checksum,encoder_checksum=model.encoder.checksum,
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),configuration=asdict(primary_config()),source_sha256=hashes,
        selection='20/20 strict flat goals; 22/30 terrain goals matching rule; 8/10 final-runtime goals versus 9/10 rule; zero contact/fall in those cohorts. Prefer explicit monotonic near-goal hold and fewer learned parameters.',
        aggregation_result='Teacher-only and aggregated data fit the identical threshold 92. Release retains the smaller teacher-only set; failed learner traces remain research evidence.',
        test_seeds=dict(main=14000,ood=14200,regular=list(range(14500,14510)),stress=list(range(14600,14610))),
        no_test_data_in_training=True,learned_encoder_weights=False,motor_training=False,
        goal_training_rows=len(minimal),base_training_rows=sum(1 for s in base_path.read_text().splitlines() if s)))
    save(a.release/'training_data_manifest.json',dict(
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.release.glob('*.gz')},
        base_original_manifest=json.loads(base_path.with_suffix('.manifest.json').read_text()),
        goal_source_full_manifest=json.loads((raw/'fit-goals-teacher-v4/training_records.manifest.json').read_text()),
        labels='Causal task/geometry/arrival teacher preferences. Minimal goal records retain source row indices and exact count sums; no counterfactual motor outcome assertion.'))
    print('Frozen release',model.checksum,'count threshold',model.threshold,flush=True)


if __name__=='__main__':main()
