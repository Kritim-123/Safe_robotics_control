"""Qualify, freeze and reproduce the single new-data request candidate."""
import argparse,gzip,hashlib,json,shutil,time
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from .request_refinement import RAW,FIT
from .loading_v2 import load_any
from .models import Selector
from .encoder import TaskEncoder
from .factored import FactoredGoalSelector
from .hysteresis import HystereticGoalSelector


def rows(folder):
    path=folder/'results.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def qualify():
    val=rows(RAW/'request-refinement-validation-v3');reg=rows(RAW/'request-refinement-regression-v3')
    bench=rows(RAW/'request-refinement-benchmark-v3')
    def summary(group):
        return dict(trials=len(group),success=sum(r['summary']['task_fulfilled'] for r in group),
            contacts=sum(r['summary']['collision'] for r in group),falls=sum(r['summary']['fall'] for r in group),
            bound_violations=sum(r['summary']['max_torque_limit_ratio']>1.+1e-9 for r in group),
            hold_violations=sum(r['summary']['hold_violation_packets'] for r in group))
    vg={a:summary([r for r in val if r['arm']==a]) for a in ('v2','candidate','rule')}
    rg,bg=summary(reg),summary(bench)
    complete=all(r['trials']==40 for r in vg.values()) and rg['trials']==160 and bg['trials']==60
    def count(arm,boundary):return sum(r['summary']['task_fulfilled'] for r in val if r['arm']==arm
        and r['summary']['case'].startswith('request_boundary_validation')==boundary)
    def no_lost_success(reference,candidate):
        previous={r['summary']['case'] for r in reference if r['summary']['task_fulfilled']}
        actual={r['summary']['case'] for r in candidate if r['summary']['task_fulfilled']}
        return sorted(previous-actual)
    lost_reg=no_lost_success([r for r in rows(RAW/'final-selector-regression-v1') if r['arm']=='new'],reg)
    lost_bench=no_lost_success([r for r in rows(RAW/'final-original-benchmark-v2') if r['arm']=='v2'],bench)
    candidate=load_any(FIT/'candidate.npz')
    checksums={r['summary']['model_checksum'] for r in val+reg+bench if r['arm']=='candidate'}
    prior=load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz')
    safety=all(not r[k] for r in (vg['candidate'],rg,bg)
               for k in ('contacts','falls','bound_violations','hold_violations'))
    gates=dict(complete=complete,boundary_gain_at_least_two=count('candidate',True)>=count('v2',True)+2,
        no_other_validation_regression=count('candidate',False)>=count('v2',False),
        regression_at_least_150=rg['success']>=150,no_lost_regression_success=not lost_reg,
        original_at_least_44=bg['success']>=44,no_lost_original_success=not lost_bench,safety=safety,
        same_candidate=checksums=={candidate.checksum},same_encoder=candidate.encoder.checksum==prior.encoder.checksum)
    report=dict(status='pending' if not complete else 'qualified' if all(gates.values()) else 'rejected',gates=gates,
        validation=vg,regression=rg,benchmark=bg,lost_regression_cases=lost_reg,lost_original_cases=lost_bench,
        boundary_success={a:count(a,True) for a in ('v2','candidate','rule')},
        other_success={a:count(a,False) for a in ('v2','candidate','rule')})
    save(RAW/'request-refinement-qualification.json',report)
    return report


def reproduce(release,output):
    output.mkdir(parents=True,exist_ok=False)
    with gzip.open(release/'training_records.jsonl.gz','rt') as stream:training=[json.loads(s) for s in stream]
    base=Selector(TaskEncoder('factor32'),'ridge',alpha=.001).fit(training)
    old=FactoredGoalSelector.load(ROOT/'docs/agents/terrain_goal_v1/primary.npz')
    model=FactoredGoalSelector(base,old.threshold,training_groups=old.training_groups,training_rows=old.training_rows)
    model.save(output/'factored.npz');final=HystereticGoalSelector.from_path(output/'factored.npz');final.save(output/'primary.npz')
    identical=(output/'primary.npz').read_bytes()==(release/'primary.npz').read_bytes()
    if not identical:raise AssertionError('Request-refined release did not reproduce exactly')
    save(output/'reproduction.json',dict(byte_identical=True,model_checksum=final.checksum,training_rows=len(training)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true')
    p.add_argument('--release',type=Path,default=ROOT/'docs/agents/terrain_goal_v3')
    p.add_argument('--reproduce',type=Path);a=p.parse_args()
    if a.reproduce:return reproduce(a.release,a.reproduce)
    q=qualify();print(q['status'],q['gates'],flush=True)
    if not a.freeze:return
    if q['status']!='qualified':raise ValueError('Candidate has not passed all declared qualification checks')
    model=load_any(FIT/'candidate.npz');prior=json.loads((ROOT/'docs/agents/terrain_goal_v2/frozen_selection.json').read_text())
    fitting=json.loads((FIT/'fit_protocol.json').read_text())
    if hashlib.sha256((ROOT/'experiments/terrain_goal/request_refinement.py').read_bytes()).hexdigest()!=fitting['fitting_source_sha256']:
        raise AssertionError('Fitting script changed after the single candidate fit')
    for name in ('request-refinement-validation-v3','request-refinement-regression-v3','request-refinement-benchmark-v3'):
        if load_any(RAW/name/'models/candidate.npz').checksum!=model.checksum:
            raise AssertionError('Qualification used another candidate')
    for name,digest in dict(prior['source_sha256'],**prior['transitive_source_and_asset_sha256']).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Frozen V2 dependency changed: '+name)
    a.release.mkdir(parents=True,exist_ok=False);shutil.copyfile(FIT/'candidate.npz',a.release/'primary.npz')
    training=(FIT/'training_records.jsonl').read_bytes()
    (a.release/'training_records.jsonl.gz').write_bytes(gzip.compress(training,mtime=0))
    shutil.copyfile(FIT/'fit_protocol.json',a.release/'fit_protocol.json')
    # Freeze inherited runtime plus the new fitting/reproduction code. Report
    # rendering and figure collection cannot affect the predictor or physics.
    hashes=dict(prior['source_sha256'])
    for name in ('request_refinement.py','package_refinement.py'):
        path=ROOT/'experiments/terrain_goal'/name
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(a.release/'frozen_selection.json',dict(frozen_epoch=time.time(),model_checksum=model.checksum,
        checkpoint_sha256=hashlib.sha256((a.release/'primary.npz').read_bytes()).hexdigest(),
        source_sha256=hashes,transitive_source_and_asset_sha256=prior['transitive_source_and_asset_sha256'],
        configuration=prior['configuration'],qualification=q,
        training_sha256=hashlib.sha256(training).hexdigest(),training_rows=sum(bool(s) for s in training.splitlines()),
        final_tests=dict(clean_seed=15500,clean_count=100,noise_seed=15501,noise_count=40),
        changes='Only request-readout data and fitted coefficients; same encoder, goal counts, hold state, motors and guards',
        source_scope='Inherited frozen runtime and new fitting/reproduction scripts; reporting code is archived separately',
        evidence='Frozen before independent V3 confirmation; V2 final errors motivated this iteration'))
    print('Frozen request-refined candidate',model.checksum,flush=True)


if __name__=='__main__':main()
