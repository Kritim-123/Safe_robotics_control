from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.obstacle_jump import trial, bar_scene, PROFILE
from experiments.smooth_jump import save
from skills.jump_profile import JumpConfig

root=Path('output/obstacle_jump/axis-validation-v1')
root.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20261030)
config=JumpConfig(**json.loads(PROFILE.read_text()))
cases=[bar_scene(f'axis-{i:02d}',distance=float(rng.uniform(.23,.24)),height=float(rng.uniform(.02,.04)),
        depth=float(rng.uniform(.012,.02)),friction=float(rng.uniform(.8,.9)),
        start=tuple(float(v) for v in rng.uniform(-.18,.18,2))) for i in range(16)]
save(root/'cases.json',[s.to_dict() for s in cases]);save(root/'frozen_config.json',asdict(config))
results=[]
with (root/'results.jsonl').open('x') as stream:
    for spec in cases:
        result=trial(root/spec.name,spec,config)
        results.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
        print(spec.name,result['reason'],result['final_collision_geometry_margin'],flush=True)
save(root/'aggregate.json',dict(trials=len(results),successes=sum(r['success'] for r in results),
    collisions=sum(r['collision'] for r in results),falls=sum(r['fall'] for r in results)))
