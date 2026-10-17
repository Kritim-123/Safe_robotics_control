"""Training-only refinement of passage requests, preserving the frozen encoder.

Goal entry/release, motors, masks and all released V1/V2 files are unchanged.
V2 final passages motivated this exploratory iteration; only new seeds train it.
"""
import argparse,gzip,json,hashlib,time
from pathlib import Path
from dataclasses import replace
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from experiments.continual_snn.data import save_records
from experiments.continual_snn.observations import teacher,SensorConfig
from experiments.continual_snn.scenes import make_case,FAMILIES,STRESS_FAMILIES
from experiments.continual_snn.stress_scenes import make_stress_case
from experiments.continual_snn.transfer import terrain_cases
from .assessment import primary_config
from .contract import AdaptationConfig
from .goal_cases import goal_cases
from .loading_v2 import load_any
from .batch import run
from .encoder import TaskEncoder,add_task
from .models import Selector
from .factored import FactoredGoalSelector
from .hysteresis import HystereticGoalSelector

RAW=ROOT/'output/terrain_goal'
FIT=RAW/'request-refinement-fit-v3b'


def fit(output):
    output.mkdir(parents=True,exist_ok=False)
    with gzip.open(ROOT/'docs/agents/terrain_goal_v1/base_training_records.jsonl.gz','rt') as f:
        original=[json.loads(s) for s in f]
    added=[]
    source=RAW/'request-refinement-training-v3'
    for line in (source/'results.jsonl').read_text().splitlines():
        row=json.loads(line);s=row['summary'];folder=source/s['case']/row['arm']
        for d in map(json.loads,(folder/'decisions.jsonl').read_text().splitlines()):
            # Base request labels ignore goal-arrival teacher state: stopping
            # is owned by the separate proximity population and hold readout.
            label=teacher(d['task'],d['request_mask'])
            if label is None:continue
            features=add_task(d['features'],d['task'])
            features.update({n:d['geometry'][n] for n in ('predicted_run_clearance','predicted_jump_clearance')})
            added.append(dict(group=f'new_request_training/{s["seed"]}/{s["case"]}',
                split='training',features=features,label=label,time=d['time'],
                source='new_training_learner_and_teacher_visited_states',
                label_source='causal_raw_mission_and_masks_not_goal_hold_or_counterfactual_outcome',
                behavior_model_checksum=s.get('model_checksum'),arm=row['arm']))
    if not added:raise ValueError('No new declared training data')
    records=original+added;save_records(output/'training_records.jsonl',records)
    base=Selector(TaskEncoder('factor32'),'ridge',alpha=.001).fit(records)
    reference=FactoredGoalSelector.load(ROOT/'docs/agents/terrain_goal_v1/primary.npz')
    factored=FactoredGoalSelector(base,reference.threshold,training_groups=reference.training_groups,
                                training_rows=reference.training_rows)
    factored.save(output/'factored.npz')
    model=HystereticGoalSelector.from_path(output/'factored.npz');model.save(output/'candidate.npz')
    save(output/'fit_protocol.json',dict(created_epoch=time.time(),original_rows=len(original),added_rows=len(added),
        added_groups=len({r['group'] for r in added}),model_checksum=model.checksum,
        encoder_checksum=model.encoder.checksum,previous_encoder_checksum=reference.encoder.checksum,
        unchanged_goal_threshold=reference.threshold,unchanged_release_threshold=model.release_threshold,
        alpha=.001,motor_training=False,neuron_training=False,
        training_batches=[source.name],no_validation_or_test_rows=True,
        fitting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='One new-data candidate after V2 clean passage errors; V2 final scenes are development evidence for this iteration'))
    print('Fitted request candidate',len(added),'new rows',model.checksum,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('train','fit','validation','regression','benchmark','confirmation','noise'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,default=1792269720)
    p.add_argument('--chunk',type=int,default=0);p.add_argument('--chunks',type=int,default=1);a=p.parse_args()
    if a.phase=='fit':return fit(a.output)
    prior=load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz');cfg=primary_config();sensor=None
    if a.phase=='train':
        scenes=[c for i,c in enumerate(goal_cases(15400,100,'request_boundary_training')) if i%10 in (7,8)]
        arms=[('v2','continuous',prior),('rule','rule',None)]
    else:
        candidate=load_any(FIT/'candidate.npz')
        arms=[('v2','continuous',prior),('candidate','continuous',candidate),('rule','rule',None)]
        if a.phase=='validation':
            scenes=[c for i,c in enumerate(goal_cases(15401,100,'request_boundary_validation')) if i%10 in (7,8)]
            scenes+=goal_cases(15402,10,'request_flat_validation',flat_only=True)
            scenes+=goal_cases(15403,10,'request_terrain_validation')
        elif a.phase=='regression':
            scenes=[make_case(f,s,'goal_regression') for s in range(14500,14510) for f in FAMILIES]
            scenes += [make_stress_case(f,s,'goal_stress_regression') for s in range(14600,14610) for f in STRESS_FAMILIES]
            cfg=AdaptationConfig();arms=[('candidate','continuous',candidate)]
        elif a.phase=='benchmark':
            scenes=terrain_cases(20261117);arms=[('candidate','continuous',candidate)]
        elif a.phase=='confirmation':scenes=goal_cases(15500,100,'request_refinement_confirmation')
        else:
            scenes=goal_cases(15501,40,'request_refinement_noisy_confirmation')
            cfg=replace(cfg,selector_position_noise=.02,selector_yaw_noise=.02);sensor=SensorConfig(.03,.015,.1,9919)
    if not 0<=a.chunk<a.chunks:raise ValueError('Invalid chunk')
    run(a.output,scenes[a.chunk::a.chunks],arms,adaptation=cfg,sensor=sensor,deadline=a.deadline)


if __name__=='__main__':main()
