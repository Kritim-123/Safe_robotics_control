"""Finite batches with a declared scene manifest and incremental honest results."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import mujoco
import numpy as np

from .models import ROOT,FrozenBankModel,load_model
from .readout import SpikeReadout
from .observations import SensorConfig
from .runner import execute
from .scenes import cases
from .supervisor import HandoffConfig


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def aggregate(rows):
    groups=[]
    for mode in dict.fromkeys(r['arm'] for r in rows):
        group=[r['summary'] for r in rows if r['arm']==mode]
        groups.append(dict(arm=mode,trials=len(group),tasks=sum(r['task_fulfilled'] for r in group),
            completed=sum(r['episode_completed'] for r in group),collisions=sum(r['collision'] for r in group),
            falls=sum(r['fall'] for r in group),reasons=dict(Counter(r['reason'] for r in group)),
            mean_progress=float(np.mean([r['progress'] for r in group])),
            total_switches=sum(r['switches'] for r in group),
            total_hops=sum(r['completed_jumps'] for r in group),
            max_torque=max(r['max_torque'] for r in group),
            hold_violations=sum(r['hold_violation_packets']>0 for r in group)))
    return dict(episodes=len(rows),groups=groups)


def run_batch(output,scene_cases,arms,*,sensor=None,handoff=None,deadline=None,video_cases=(),full_trace=False):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):
        raise ValueError('Batch output must be inside project output')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'cases.json',[c.to_dict() for c in scene_cases])
    save(output/'arms.json',[dict(name=name,mode=mode,model_name=getattr(model,'name',mode),
        model_checksum=getattr(model,'checksum',None)) for name,mode,model in arms])
    source_paths=['experiments/continual_snn/'+name+'.py' for name in (
        'scenes','observations','supervisor','runner','models','encoding','readout','study')]
    source_paths += ['skills/dynamic.py','skills/jump_profile.py','skills/avoidance.py',
        'controllers/walking.py','controllers/navigation.py','hybrid/arena.py','hybrid/terrain.py']
    for extra in ('memory','stress_scenes','prototypes','predictive','transfer'):
        path=f'experiments/continual_snn/{extra}.py'
        if (ROOT/path).exists():source_paths.append(path)
    for source in source_paths:
        destination=output/'source_snapshot'/source;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,destination)
    for name,mode,model in arms:
        if model is not None and hasattr(model,'save'):
            model.save(output/'models'/(name+'.npz'))
        elif isinstance(model,FrozenBankModel):
            model.encoder.save(output/'models'/(name+'_encoder.npz'))
            model.memory.save(output/'models'/(name+'_associations.json'))
    save(output/'manifest.json',dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,
        started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        sensor=asdict(sensor) if sensor else None,handoff=asdict(handoff) if handoff else None,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        policy='freeze cases/arms before execution; no fitting in this batch',deadline_epoch=deadline))
    rows=[]
    with (output/'results.jsonl').open('x',encoding='utf-8') as stream:
        for case in scene_cases:
            # Never start a partial matched group once the work cutoff arrives.
            if deadline is not None and time.time()>=deadline:
                save(output/'stopped.json',dict(reason='declared wall-clock cutoff',matched_groups=len(rows)//len(arms)))
                break
            for name,mode,model in arms:
                sensor_config=sensor
                if sensor is not None:
                    sensor_config=SensorConfig(sensor.position_noise,sensor.velocity_noise,sensor.delay,sensor.seed+case.seed)
                summary=execute(case,output/case.scenario.name/name,mode=mode,model=model,
                    sensor_config=sensor_config,config=handoff,
                    video=case.scenario.name in video_cases,full_trace=full_trace)
                row=dict(arm=name,summary=summary)
                rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                save(output/'aggregate.json',aggregate(rows))
                print(len(rows),case.scenario.name,name,summary['reason'],
                    'task='+str(summary['task_fulfilled']),round(summary['progress'],3),
                    'hops='+str(summary['completed_jumps']),flush=True)
    return aggregate(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('collect','evaluate'),default='evaluate')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('development','train','validation','test','confirmation'),default='development')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--arms',nargs='+',default=['old_once','old_repeat','rule'])
    parser.add_argument('--model',type=Path)
    parser.add_argument('--position-noise',type=float,default=0.)
    parser.add_argument('--velocity-noise',type=float,default=0.)
    parser.add_argument('--delay',type=float,default=0.)
    parser.add_argument('--confirmations',type=int,default=2)
    parser.add_argument('--decision-dt',type=float,default=.1)
    parser.add_argument('--full-trace',action='store_true')
    args=parser.parse_args()
    selected=cases(args.split,args.seeds)
    loaded=load_model(args.model) if args.model else None
    baseline=None
    def old():
        nonlocal baseline
        if baseline is None:baseline=FrozenBankModel()
        return baseline
    choices=dict(old_once=('one_shot',old),old_repeat=('continuous',old),
        rule=('rule',lambda:None),always_run=('always_run',lambda:None),always_wait=('always_wait',lambda:None),
        new_once=('one_shot',lambda:loaded),new_repeat=('continuous',lambda:loaded))
    names=['rule'] if args.phase=='collect' else args.arms
    if any(name not in choices for name in names):raise ValueError('Unknown model arm')
    if any(name.startswith('new_') for name in names) and loaded is None:raise ValueError('Supply --model for new arms')
    arms=[(name,choices[name][0],choices[name][1]()) for name in names]
    sensor=SensorConfig(args.position_noise,args.velocity_noise,args.delay,20261017)
    result=run_batch(args.output,selected,arms,sensor=sensor,
        handoff=HandoffConfig(decision_dt=args.decision_dt,confirmations=args.confirmations),full_trace=args.full_trace)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
