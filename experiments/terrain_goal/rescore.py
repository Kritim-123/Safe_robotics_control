"""Additional strict goal/corridor diagnostics without rewriting old results."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.continual_snn.study import save


def check(folder):
    s=json.loads((folder/'summary.json').read_text())
    case=json.loads((folder/'scenario.json').read_text())
    with np.load(folder/'trace.npz',allow_pickle=False) as bundle:
        a=bundle['values'];c=bundle['columns'].tolist()
    def col(n):return a[:,c.index(n)]
    xy=np.column_stack((col('qpos_0'),col('qpos_1')))
    goal=np.asarray(case['scenario']['goal']);distance=np.linalg.norm(xy-goal,axis=1)
    vel=np.column_stack([col('qvel_'+str(i)) for i in range(6)])
    stable=(np.linalg.norm(vel[:,:3],axis=1)<.04)&(np.linalg.norm(vel[:,3:],axis=1)<.2)&(col('foot_contacts')==4)&(col('gait_activity')==0)&(col('upright')>np.cos(np.deg2rad(12)))&(col('qpos_2')>.23)
    times=col('time');progress=np.asarray([any(r['task']=='progress' and r['start']<=t<r['end'] for r in case['requests']) for t in times])
    near=np.flatnonzero((distance<=.12)&progress)
    first_near=float(times[near[0]]) if len(near) else None
    eligible=(distance<=.12)&stable&(col('skill')==0)&progress
    since=arrival=None
    for t,valid in zip(times,eligible):
        since=(float(t) if since is None else since) if valid else None
        if since is not None and t-since>=.8-1e-8:arrival=float(t);break
    departed=bool(arrival is not None and np.any((times>arrival)&progress&(distance>.16)))
    delay=arrival-first_near if arrival is not None and first_near is not None else None
    strong=bool(s['episode_completed'] and s['stable_recovery'] and not s['collision'] and not s['fall']
        and s['hold_violation_packets']==0 and distance[-1]<=.12 and arrival is not None and delay<=6. and not departed)
    yaw=case['scenario']['start_yaw'];left=np.array([-np.sin(yaw),np.cos(yaw)])
    lateral=(xy-np.asarray(case['scenario']['start']))@left
    return dict(case=s['case'],original_task_fulfilled=s['task_fulfilled'],strict_goal=strong,
        final_goal_distance=float(distance[-1]),first_near=first_near,first_supported_wait=arrival,
        stopping_delay=delay,departed_after_arrival=departed,
        final_lateral=float(lateral[-1]),maximum_absolute_lateral=float(np.max(np.abs(lateral))),
        traversal_with_final_lateral_corridor=bool(s['task_fulfilled'] and abs(lateral[-1])<=.35),
        trace_dt=s['trace_dt'],borderline_delay=bool(delay is not None and abs(delay-6.)<.05))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for line in (a.batch/'results.jsonl').read_text().splitlines():
        row=json.loads(line);s=row['summary']
        rows.append(dict(arm=row['arm'],**check(a.batch/s['case']/row['arm'])))
    arms={}
    for arm in dict.fromkeys(r['arm'] for r in rows):
        group=[r for r in rows if r['arm']==arm]
        arms[arm]=dict(trials=len(group),original_tasks=sum(r['original_task_fulfilled'] for r in group),
            strict_goals=sum(r['strict_goal'] for r in group),
            traversal_with_final_lateral_corridor=sum(r['traversal_with_final_lateral_corridor'] for r in group))
    save(a.output,dict(method='Separate additional diagnostic, sampled trace; original files unchanged. Goal latency borders within 50 ms flagged.',arms=arms,rows=rows))


if __name__=='__main__':main()
