from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from experiments.obstacle_jump import trial, bar_scene
from experiments.smooth_jump import save
from skills.jump_profile import JumpConfig

root=Path('output/obstacle_jump/bar-development-v2')
root.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20261028)
configs=[]
for i in range(240):
    calf=float(rng.uniform(-2.5,-2.15))
    config=replace(JumpConfig(),preparation=float(rng.uniform(.4,.65)),
        thrust=float(rng.uniform(.065,.115)),forward_lean=float(rng.uniform(.65,1.15)),
        crouch_calf=calf,crouch_thigh=-calf/2,
        extension_calf=float(rng.uniform(-1.35,-.88)),
        flight=float(rng.uniform(.18,.32)),flight_calf=float(rng.uniform(-2.6,-2.15)),
        flight_lean=float(rng.uniform(-.35,.10)),absorption=float(rng.uniform(.16,.30)),
        absorption_calf=-2.06,absorption_thigh=1.03,landing_kp=float(rng.uniform(28.,48.)),
        landing_kd=float(rng.uniform(1.2,2.5)),torque_limit=float(rng.choice((48.,60.,72.))))
    configs.append(config)
save(root/'declared_configs.json',[asdict(c) for c in configs])
with (root/'results.jsonl').open('x') as stream:
    for i,config in enumerate(configs):
        summary=trial(root/f'{i:03d}',bar_scene(f'bar-development-{i:03d}',distance=.27,height=.02),config)
        stream.write(json.dumps(summary)+'\n');stream.flush()
        print(i,summary['reason'],round(summary['final_position'][0],3),
              round(summary['final_collision_geometry_margin'],3),summary['foot_crossed'],flush=True)
