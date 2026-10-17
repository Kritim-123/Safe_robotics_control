"""Explicit new executor cohorts; source/model/case manifests precede physics."""
from dataclasses import asdict
import hashlib,json,shutil,time,os,platform
import numpy as np
import mujoco
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save,aggregate
from .runner import execute
from .contract import AdaptationConfig

_SOURCE_PATHS=list((ROOT/'experiments/terrain_goal').glob('*.py'))
_SOURCE_PATHS+=list((ROOT/'experiments/continual_snn').glob('*.py'))
_SOURCE_PATHS+=[ROOT/p for p in ('controllers/navigation.py','controllers/walking.py','skills/dynamic.py','hybrid/arena.py','hybrid/terrain.py')]
_RUNTIME_SOURCES={str(p.relative_to(ROOT)):p.read_bytes() for p in _SOURCE_PATHS}
_RUNTIME_CAPTURED=time.time()


def run(output,cases,arms,*,adaptation=None,video_cases=(),full_trace=False,sensor=None,deadline=None):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'output'):raise ValueError('Output outside project')
    output.mkdir(parents=True,exist_ok=False)
    adaptation=adaptation or AdaptationConfig();adaptation.validate()
    hashes={p:hashlib.sha256(data).hexdigest() for p,data in _RUNTIME_SOURCES.items()}
    for p,data in _RUNTIME_SOURCES.items():
        dest=output/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
    save(output/'cases.json',[c.to_dict() for c in cases])
    model_artifacts={}
    for name,mode,model in arms:
        if model is not None and hasattr(model,'save'):
            path=output/'models'/(name+'.npz');model.save(path)
            model_artifacts[name]=dict(path=str(path.relative_to(output)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(output/'manifest.json',dict(started_epoch=time.time(),adaptation=asdict(adaptation),source_sha256=hashes,
        python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__,
        source_capture_epoch=_RUNTIME_CAPTURED,source_capture='one immutable in-memory bundle at process import',process_id=os.getpid(),
        model_artifacts=model_artifacts,
        deadline_epoch=deadline,
        arms=[dict(name=n,mode=m,checksum=getattr(o,'checksum',None)) for n,m,o in arms]))
    rows=[]
    with (output/'results.jsonl').open('x') as stream:
        for case in cases:
            if deadline is not None and time.time()>=deadline:
                save(output/'stopped.json',dict(reason='Declared wall-clock cutoff before next complete matched case group',
                    completed_groups=len(rows)//len(arms),planned_cases=len(cases)))
                break
            for name,mode,model in arms:
                summary=execute(case,output/case.scenario.name/name,mode=mode,model=model,
                    adaptation=adaptation,video=case.scenario.name in video_cases,full_trace=full_trace,
                    sensor_config=sensor)
                row=dict(arm=name,summary=summary);rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                save(output/'aggregate.json',aggregate(rows))
                print(len(rows),case.scenario.name,name,summary['reason'],summary['task_fulfilled'],
                    round(summary['progress'],3),round(summary['final_goal_distance'],3),flush=True)
    return aggregate(rows)
