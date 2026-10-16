"""Collect matched actual MuJoCo skill attempts, including rejected/failed ones.

Every context has a task known before execution. Labels rank successful measured
outcomes under that task. The jump task is explicitly an in-place demonstration,
never an obstacle-crossing task. One training example per context avoids treating
three counterfactual attempts as three independent observations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import platform
import mujoco
from pathlib import Path
import numpy as np
from hybrid.arena import Scenario, Obstacle
from hybrid.skill_lif import FEATURES, SKILLS
from skills.avoidance import clearance

FAMILIES = ('clear', 'far_obstacle', 'blocked', 'crossing', 'jump_clear', 'jump_restricted')
SPLIT_SEEDS = {'train': tuple(range(1100,1108)), 'validation': tuple(range(2100,2103)),
               'test': tuple(range(3100,3103))}


def make_context(family, seed):
    if family not in FAMILIES:
        raise ValueError('Unknown context family')
    rng = np.random.default_rng(seed)
    radius = float(rng.uniform(.15,.25))
    lateral = float(rng.uniform(-.12,.12))
    obstacles = ()
    if family == 'far_obstacle':
        obstacles = (Obstacle('far', (float(rng.uniform(3.,3.5)), lateral), radius),)
    elif family in ('blocked','jump_restricted'):
        obstacles = (Obstacle('near', (float(rng.uniform(.76,.86)), lateral), radius),)
    elif family == 'crossing':
        obstacles = (Obstacle('crossing', (float(rng.uniform(.8,1.2)),-1.2),radius,
                              (0,float(rng.uniform(.13,.20))),stop_after=12),)
    spec = Scenario(f'{family}_{seed}',goal=(3.,0.),bounds=(-3.,6.,-3.,3.),
                    obstacles=obstacles,friction=float(rng.uniform(.65,1.0)))
    return spec, family.startswith('jump_')


def observation_features(spec, observation, jump_requested):
    position = np.asarray(observation['position'])
    yaw, now = observation['yaw'], observation['time']
    forward = np.array([np.cos(yaw),np.sin(yaw)])
    left = np.array([-forward[1],forward[0]])
    obstacles = [(np.asarray(o.at(now)),o.planning_radius,np.asarray(o.speed_at(now))) for o in spec.obstacles]
    gap = clearance(position,obstacles,spec.bounds)
    if obstacles:
        index = int(np.argmin([np.linalg.norm(o[0]-position) for o in obstacles]))
        center,radius,velocity = obstacles[index]
        relative = center-position
        obstacle = spec.obstacles[index]
        height = obstacle.height
        width = 2*(obstacle.half_extents[1] if obstacle.shape == 'box' else obstacle.radius)
        front, side, vf, vl = relative@forward,relative@left,velocity@forward,velocity@left
    else:
        radius=height=width=front=side=vf=vl=0.
    goal = np.asarray(spec.goal)-position
    return np.asarray([bool(obstacles),front,side,vf,vl,radius,height,width,gap,
                       goal@forward,goal@left,observation['speed'],observation['height'],
                       jump_requested],dtype=np.float32)


def outcome_utility(summary, jump_requested):
    """Only successful trials compete; fallback hold is local safety, not task completion."""
    if not summary['success'] or summary['collision'] or summary['fall']:
        return None
    action = summary['skill']
    if jump_requested:
        if action == 'jump' and summary['all_feet_airborne'] and summary['stable_landing']:
            return 10.+summary['longest_airtime']
        return 0. if action == 'wait' else None
    if action == 'run' and summary['displacement'][0] > .1:
        return 1.+summary['displacement'][0]
    return 0. if action == 'wait' else None


def collect_context(job):
    from skills.dynamic import execute_dynamic_skill
    family,seed,split,output = job
    spec,jump_requested = make_context(family,seed)
    context_id = f'{family}-{seed}'
    directory = Path(output)/'trials'/context_id
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'scenario.json').write_text(json.dumps(spec.to_dict(),indent=2),encoding='utf-8')
    rows=[]
    for action in SKILLS:
        history,summary = execute_dynamic_skill(action,spec,seed=seed)
        trace = directory/f'{action}.csv'
        with trace.open('w',newline='',encoding='utf-8') as stream:
            if history:
                writer=csv.DictWriter(stream,fieldnames=list(history[0]))
                writer.writeheader();writer.writerows(history)
        (directory/f'{action}.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        observation = summary['decision_observation']
        if observation is None:
            raise RuntimeError(f'Context {context_id} failed before decision; preserve trial and redesign development context')
        features = observation_features(spec,observation,jump_requested)
        utility = outcome_utility(summary,jump_requested)
        row=dict(context_id=context_id,episode_id=f'{context_id}-{action}',family=family,
                 scenario_seed=seed,split=split,decision_index=0,timestamp=observation['time'],
                 task='in_place_jump' if jump_requested else 'forward_progress',chosen_action=action,
                 applied_action=action if summary['reason'] != 'failed initiation' else '',
                 label_action='',label_source='matched successful measured outcome utility',
                 outcome_success=int(summary['success']),outcome_task_fulfilled=int(utility is not None and utility > 0),outcome_collision=int(summary['collision']),
                 outcome_fall=int(summary['fall']),outcome_reason=summary['reason'],
                 outcome_duration=summary['elapsed'],outcome_progress=summary['displacement'][0],
                 outcome_airtime=summary['longest_airtime'],outcome_clearance=summary['min_clearance'],
                 outcome_max_torque=summary['max_torque'],outcome_torque_limit_ratio=summary['max_torque_limit_ratio'],
                 outcome_utility='' if utility is None else utility,
                 trace_path=str(trace.relative_to(Path(output))),
                 trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        row.update({f:float(v) for f,v in zip(FEATURES,features)})
        row.update({f'allowed_{s}':int(summary['eligibility'][s]) for s in SKILLS})
        rows.append(row)
    eligible=[r for r in rows if r['outcome_utility'] != '']
    winner=max(eligible,key=lambda r:r['outcome_utility'])['chosen_action'] if eligible else ''
    for row in rows:
        row['label_action']=winner
        if winner == 'wait': row['label_source']='successful measured local-hold fallback; requested task unfulfilled'
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/skill_lif/dataset'))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=[(f,s,split,str(output)) for split,seeds in SPLIT_SEEDS.items() for s in seeds for f in FAMILIES]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(collect_context,j) for j in jobs]
        for future in as_completed(futures):
            group=future.result();rows.extend(group)
            print(json.dumps(dict(context=group[0]['context_id'],label=group[0]['label_action'],
                                  outcomes=[r['outcome_reason'] for r in group])),flush=True)
    rows.sort(key=lambda r:(r['split'],r['context_id'],SKILLS.index(r['chosen_action'])))
    path=output/'decisions.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(schema_version=1,contexts=len(jobs),attempts=len(rows),split_seeds=SPLIT_SEEDS,families=FAMILIES,
                  feature_names=FEATURES,skill_names=SKILLS,dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  failed_attempts=sum(not r['outcome_success'] for r in rows),
                  runtime=dict(python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 map(Path,['hybrid/behavior_dataset.py','hybrid/skill_lif.py','skills/dynamic.py','controllers/walking.py','controllers/navigation.py'])},
                  units={f:('boolean' if f in ('obstacle_present','jump_requested') else
                            'm/s' if f in ('obstacle_v_forward','obstacle_v_left','speed') else 'm') for f in FEATURES},
                  feature_frame='relative coordinates projected onto current robot yaw; heights world vertical; all available before action',
                  outcome_units=dict(duration='s',progress='m along world x',airtime='s',clearance='m',max_torque='Nm',torque_limit_ratio='dimensionless'),
                  feature_source='settled pre-action simulator pose and privileged scripted obstacle state; explicit task command',
                  label_protocol='max utility among successful measured candidate trials; wait fallback utility 0; forward run 1+progress if >.1m; requested jump 10+verified flight; unrelated jump/run excluded',
                  limitations='in-place jump only; one independent decision per fresh episode; no sensor noise or real perception; scripted obstacles')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__': main()
