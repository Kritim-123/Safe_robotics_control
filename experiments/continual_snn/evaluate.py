"""Predeclared matched evaluation, separate from fitting and model selection."""
import argparse
import calendar
import hashlib
import json
from pathlib import Path
import time
from .models import ROOT,load_model,FrozenBankModel
from .scenes import cases
from .stress_scenes import stress_cases
from .transfer import heading_cases,terrain_cases
from .observations import SensorConfig
from .study import run_batch,save

BASE=ROOT/'output/continual_snn'
MODEL_PATHS=dict(new=BASE/'predictive-comparison-v1/models/spike.npz',
    prototype=BASE/'predictive-comparison-v1/models/prototype-k1-r12.npz',
    sensory64=BASE/'readout-comparison-v1/models/candidate-01.npz',
    analog24=BASE/'predictive-comparison-v1/models/analog.npz')
DEADLINE=calendar.timegm(time.strptime('2026-10-17 09:55:00','%Y-%m-%d %H:%M:%S'))


def arms(names):
    models={key:load_model(path) for key,path in MODEL_PATHS.items()}
    old=FrozenBankModel()
    options=dict(old_once=('one_shot',old),old_repeat=('continuous',old),
        new_once=('one_shot',models['new']),new_repeat=('continuous',models['new']),
        prototype80=('continuous',models['prototype']),sensory64=('continuous',models['sensory64']),
        analog24=('continuous',models['analog24']),rule=('rule',None),always_run=('always_run',None))
    return [(name,*options[name]) for name in names]


def plan():
    common=['new_repeat','prototype80','analog24','rule']
    return dict(
        main=(cases('test',range(6100,6120)),
              ['old_once','old_repeat','new_once','new_repeat','prototype80','sensory64','analog24','rule','always_run'],None),
        stress=(stress_cases(range(8100,8110)),['new_repeat','prototype80','sensory64','analog24','rule','always_run'],None),
        confirmation=(cases('confirmation',range(7100,7110)),common,None),
        noise=(cases('test',range(6100,6110)),common,SensorConfig(.03,.015,.1,20261017)),
        severe_noise=(cases('test',range(6100,6110)),common,SensorConfig(.08,.05,.2,20261118)),
        heading=(heading_cases(range(9200,9210)),common,None),
        terrain=(terrain_cases(),['new_repeat','prototype80','sensory64','analog24','rule'],None),
        stale=(cases('test',range(6100,6102)),common,SensorConfig(0.,0.,.5,20261017)))


def declared_plan():
    return {name:dict(cases=[c.to_dict() for c in scenes],arms=names,
        sensor=vars(sensor) if sensor else None) for name,(scenes,names,sensor) in plan().items()}


def freeze():
    path=BASE/'frozen-evaluation-v1.json'
    if path.exists():raise FileExistsError(path)
    models={k:load_model(p) for k,p in MODEL_PATHS.items()}
    report=dict(frozen_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        primary='new_repeat: 64 sensory + 16 predictive LIF, 16 steps, packet reset, ridge alpha=.0001',
        alternate='prototype80: exact conflicts abstain, nearest count vector, radius 12',
        selection='training/validation only; 8200/8201 stress engineering exposed; no 6100/7100/8100/9200 outcomes inspected',
        models={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            checksum=models[k].checksum) for k,p in MODEL_PATHS.items()},
        planned_batches=declared_plan(),
        deadline_epoch=DEADLINE,deadline_policy='stop before a new matched case group; report any unexecuted cases',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'experiments/continual_snn').glob('*.py'))},
        dependency_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'skills/dynamic.py','skills/jump_profile.py','skills/avoidance.py','controllers/walking.py',
            'controllers/navigation.py','hybrid/arena.py','hybrid/terrain.py')},
        no_model_or_supervisor_changes_after_freeze=True)
    save(path,report);return report


def verify():
    report=json.loads((BASE/'frozen-evaluation-v1.json').read_text())
    if json.dumps(report['planned_batches'],sort_keys=True)!=json.dumps(declared_plan(),sort_keys=True):
        raise ValueError('Frozen scene/arm/sensor plan changed')
    for key,item in report['models'].items():
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen model changed: '+key)
    # Analysis/demo tooling may be added later; runtime sources cannot change.
    for source in ('scenes','observations','supervisor','runner','encoding','predictive','readout','prototypes',
                   'memory','models','stress_scenes','transfer','study'):
        rel='experiments/continual_snn/'+source+'.py'
        expected=report['source_sha256'].get(rel,report['source_sha256'].get(str(Path(rel))))
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen runtime changed: '+source)
    for source,checksum in report['dependency_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=checksum:
            raise ValueError('Frozen motor/physics dependency changed: '+source)


def evaluate(phases):
    for phase in phases:
        verify();scenes,names,sensor=plan()[phase]
        run_batch(BASE/('final-'+phase+'-v1'),scenes,arms(names),sensor=sensor,deadline=DEADLINE)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--phases',nargs='+',choices=list(plan()))
    args=parser.parse_args()
    if args.freeze:freeze()
    if args.phases:evaluate(args.phases)
