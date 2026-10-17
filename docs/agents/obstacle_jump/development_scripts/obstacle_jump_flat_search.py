from dataclasses import replace
import itertools
import json
from pathlib import Path

from experiments.smooth_jump import run_trial, save
from hybrid.arena import Scenario
from skills.jump_profile import JumpConfig

root=Path('output/obstacle_jump/flat-development-v1')
root.mkdir(parents=True,exist_ok=False)
grid=list(itertools.product((.4,.6,.8,1.),(.08,.12,.16,.20),(36.,48.,60.),(-2.1,-2.5)))
rows=[]
with (root/'results.jsonl').open('x') as stream:
    for i,(lean,thrust,limit,tuck) in enumerate(grid):
        config=replace(JumpConfig(),forward_lean=lean,thrust=thrust,torque_limit=limit,
                       flight_calf=tuck,flight_lean=-.1,flight=.23)
        result=run_trial(root/f'{i:03d}',Scenario('long_jump_development',bounds=(-4.,6.,-4.,4.)),config=config)
        rows.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
        print(i,result['summary']['reason'],round(result['metrics']['forward_progress'],3),
              round(result['metrics']['flight_forward_progress'],3),flush=True)
save(root/'best.json',sorted([r for r in rows if r['summary']['success']],
     key=lambda r:r['metrics']['flight_forward_progress'],reverse=True)[:15])
