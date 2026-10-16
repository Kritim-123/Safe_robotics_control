"""Execute output-LIF choices in the same settled MuJoCo simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hybrid.behavior_dataset import make_context, observation_features, FAMILIES
from hybrid.skill_lif import SkillLIF, SKILLS, DEFAULT_MODEL


def execute_choice(model, family, seed, output, viewer=False):
    from skills.dynamic import execute_dynamic_skill
    spec,jump_requested=make_context(family,seed)
    policy=SkillLIF(model)
    decision={}
    def select(observation,eligibility):
        features=observation_features(spec,observation,jump_requested)
        started=time.perf_counter()
        choice,reason,counts,diagnostics=policy.choose(features,[eligibility[s] for s in SKILLS])
        decision.update(time=observation['time'],features=features.tolist(),eligibility=eligibility,
                        output_counts=counts.tolist(),decode_reason=reason,
                        chosen_action=SKILLS[choice] if choice is not None else None,
                        inference_ms=(time.perf_counter()-started)*1000,
                        output_raster=diagnostics['output_spikes'].tolist())
        return decision['chosen_action']
    history,summary=execute_dynamic_skill(None,spec,seed=seed,viewer=viewer,selector=select)
    summary.update(family=family,jump_requested=jump_requested,decision=decision,
                   task_success=bool(summary['success'] and
                                     ((summary['skill']=='jump' and summary['all_feet_airborne']) if jump_requested
                                      else (summary['skill']=='run' and summary['displacement'][0]>.1))),
                   model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest())
    directory=Path(output)/f'{family}-{seed}';directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (directory/'scenario.json').write_text(json.dumps(spec.to_dict(),indent=2),encoding='utf-8')
    with (directory/'trajectory.csv').open('w',newline='',encoding='utf-8') as stream:
        if history:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    with (directory/'output_spikes.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['microstep',*SKILLS])
        writer.writerows([[i,*row] for i,row in enumerate(decision.get('output_raster',[]))])
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=Path('output/skill_lif/execution'))
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--seeds',nargs='+',type=int,default=[4100,4101,4102])
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    if any(s<0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds) or len(set(args.families)) != len(args.families):
        parser.error('Unique families and unique nonnegative seeds required')
    rows=[]
    for family in args.families:
        for seed in args.seeds:
            result=execute_choice(args.model,family,seed,args.output,args.viewer);rows.append(result)
            print(json.dumps({k:result[k] for k in ('family','seed','skill','success','task_success','reason')}),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(dict(results=rows),indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
