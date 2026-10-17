"""Replay an exact saved case at 500 Hz to adjudicate coarse-score ambiguity."""
import argparse,json
from pathlib import Path
import numpy as np
from hybrid.arena import Scenario,Obstacle
from hybrid.terrain import TerrainPatch
from experiments.continual_snn.scenes import StreamCase,Request
from experiments.continual_snn.observations import SensorConfig
from experiments.continual_snn.study import save
from .contract import AdaptationConfig
from .loading_v2 import load_any
from .batch import run
from .verify_acceptance import check
from .goal_bias import BiasedGoalFeatures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
    episode=(a.batch/a.case/a.arm).resolve()
    if not episode.is_relative_to(a.batch.resolve()):raise ValueError('Episode outside batch')
    s=json.loads((episode/'summary.json').read_text());payload=json.loads((episode/'scenario.json').read_text())
    spec=payload['scenario'];spec['obstacles']=tuple(Obstacle(**o) for o in spec['obstacles'])
    spec['terrain']=tuple(TerrainPatch(**t) for t in spec.get('terrain',()))
    payload['scenario']=Scenario(**spec);payload['requests']=tuple(Request(**r) for r in payload['requests'])
    case=StreamCase(**payload);model=None
    if s['mode']!='rule':
        model=load_any(a.batch/'models'/(a.arm+'.npz'))
        condition=a.batch/'observation_condition.json'
        if condition.exists():
            value=json.loads(condition.read_text()).get('goal_forward_additive_bias_m')
            if value is not None:model=BiasedGoalFeatures(model,value)
    manifest=json.loads((a.batch/'manifest.json').read_text())
    run(a.output,[case],[(a.arm,s['mode'],model)],adaptation=AdaptationConfig(**manifest['adaptation']),
        sensor=SensorConfig(**s['sensor_config']),full_trace=True,
        video_cases=[case.scenario.name] if a.video else ())
    replay=a.output/a.case/a.arm;report=check(replay)
    with np.load(episode/'trace.npz') as old,np.load(replay/'trace.npz') as new:
        x,y=old['values'],new['values'];indices=np.searchsorted(y[:,0],x[:,0])
        identical=bool(np.all(indices<len(y)) and np.array_equal(x,y[indices]))
    report.update(original_shared_trace_ticks_bit_identical=identical,original_runtime_success=s['task_fulfilled'],
        original_runtime_score_preserved=s['task_fulfilled']==report['runtime_success'],
        source_episode=str(episode),method='Same frozen model, case, sensor and adaptation; full-rate independent scoring. Original files unchanged.')
    save(a.output/'adjudication.json',report)
    if not identical or not report['agrees'] or not report['original_runtime_score_preserved']:
        raise AssertionError('Replay did not resolve ambiguity consistently; inspect retained evidence')
    print('Full-rate adjudication agrees; original score preserved',a.case,a.arm,flush=True)


if __name__=='__main__':main()
