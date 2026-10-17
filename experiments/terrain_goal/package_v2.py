"""Freeze/reproduce V2 only after the new confirmed-entry validation cohorts."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
from dataclasses import asdict
from experiments.continual_snn.models import ROOT,BUNDLE
from experiments.continual_snn.study import save
from .assessment import primary_config
from .hysteresis import HystereticGoalSelector
from .package import reproduce as reproduce_base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,default=ROOT/'docs/agents/terrain_goal_v2')
    p.add_argument('--reproduce',type=Path)
    a=p.parse_args();v1=ROOT/'docs/agents/terrain_goal_v1';raw=ROOT/'output/terrain_goal'
    if a.reproduce:
        a.reproduce.mkdir(parents=True,exist_ok=False)
        reproduce_base(v1,a.reproduce/'base')
        model=HystereticGoalSelector.from_path(a.reproduce/'base/primary.npz');model.save(a.reproduce/'primary.npz')
        original=HystereticGoalSelector.load(a.release/'primary.npz')
        if model.checksum!=original.checksum:raise AssertionError('V2 did not reproduce')
        save(a.reproduce/'reproduction.json',dict(model_checksum=model.checksum,same_model_checksum=True,
            byte_identical=(a.reproduce/'primary.npz').read_bytes()==(a.release/'primary.npz').read_bytes()))
        return
    paths=[raw/'hysteresis-confirmed-flat-validation-v2',raw/'hysteresis-confirmed-terrain-validation-v2',raw/'hysteresis-clean-validation-v2']
    qualifications=[]
    for path,n in zip(paths,(20,20,10)):
        groups=json.loads((path/'aggregate.json').read_text())['groups'];lookup={r['arm']:r for r in groups}
        if set(lookup)!=set(('v1','v2','rule')) or any(r['trials']!=n for r in groups):raise ValueError('Validation is incomplete')
        if lookup['v2']['collisions'] or lookup['v2']['falls'] or lookup['v2']['max_torque']>24.+1e-9:raise ValueError('V2 safety qualification failed')
        qualifications.append(dict(batch=path.name,groups=groups))
    if qualifications[0]['groups'][1]['tasks']<19:raise ValueError('Flat-noise qualification failed')
    for item in qualifications:
        d={g['arm']:g for g in item['groups']}
        if d['v2']['tasks']<d['v1']['tasks']:raise ValueError('V2 regressed validation')
    noisy=[{g['arm']:g['tasks'] for g in q['groups']} for q in qualifications[:2]]
    if sum(x['v2'] for x in noisy)<=sum(x['v1'] for x in noisy):raise ValueError('No noisy-goal improvement')
    frozen=json.loads((v1/'frozen_selection.json').read_text())
    for name,digest in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('V1 source changed: '+name)
    a.release.mkdir(parents=True,exist_ok=False)
    model=HystereticGoalSelector.from_path(v1/'primary.npz')
    for path in paths:
        checkpoint=HystereticGoalSelector.load(path/'models/v2.npz')
        if checkpoint.checksum!=model.checksum:raise ValueError('Validation used a different V2')
    model.save(a.release/'primary.npz')
    sources=list((ROOT/'experiments/terrain_goal').glob('*.py'))
    hashes=dict(frozen['source_sha256'])
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    transitive=[]
    for name in ('controllers','hybrid','skills','experiments/spike_pattern_skills'):
        transitive+=list((ROOT/name).glob('*.py'))
    transitive += [p for p in (ROOT/'unitree_go2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    transitive += [p for p in BUNDLE.iterdir() if p.suffix in ('.json','.npz')]
    transitive_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in transitive}
    save(a.release/'frozen_selection.json',dict(frozen_epoch=time.time(),model_checksum=model.checksum,
        checkpoint_sha256=hashlib.sha256((a.release/'primary.npz').read_bytes()).hexdigest(),
        base_model_checksum=model.base.checksum,parameters=model.metadata(),configuration=asdict(primary_config()),
        source_sha256=hashes,transitive_source_and_asset_sha256=transitive_hashes,qualification=qualifications,
        tests=dict(clean_seed=15100,clean_cases=200,mild_seed=15100,mild_cases=100,severe_seed=15101,severe_cases=100),
        no_encoder_or_motor_weight_changes=True,no_fresh_v2_test_data_used=True,
        evidence_status='Frozen after new validation and before independent V2 confirmation; V1 noisy tests motivated this iteration'))
    print('Frozen V2',model.checksum,flush=True)


if __name__=='__main__':main()
