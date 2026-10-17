"""Independent sampled-trace goal scoring with supervisor mode reconstruction.

This diagnostic never rewrites runtime scores. A 20 ms trace cannot exclude
brief intervening events; disagreements require full-rate replays to adjudicate.
"""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.continual_snn.study import save


def check(episode):
    s=json.loads((episode/'summary.json').read_text());case=json.loads((episode/'scenario.json').read_text())
    with np.load(episode/'trace.npz',allow_pickle=False) as bundle:
        a=bundle['values'];cols=bundle['columns'].tolist()
    def column(name):return a[:,cols.index(name)]
    times=column('time');xy=np.column_stack([column('qpos_'+str(i)) for i in range(2)])
    distance=np.linalg.norm(xy-np.asarray(case['scenario']['goal']),axis=1)
    velocity=np.column_stack([column('qvel_'+str(i)) for i in range(6)])
    stable=((np.linalg.norm(velocity[:,:3],axis=1)<.04)&(np.linalg.norm(velocity[:,3:],axis=1)<.2)
        &(column('foot_contacts')==4)&(column('gait_activity')==0)
        &(column('upright')>np.cos(np.deg2rad(12)))&(column('qpos_2')>.23))
    events=[r for r in map(json.loads,(episode/'events.jsonl').read_text().splitlines()) if 'mode' in r]
    indices=np.searchsorted([r['time'] for r in events],times,side='right')-1
    active=(indices>=0)&np.asarray([events[max(0,i)]['mode']=='active' for i in indices])
    progress=np.zeros(len(times),dtype=bool)
    for r in case['requests']:
        if r['task']=='progress':progress|=(times>=r['start'])&(times<r['end'])
    near=np.flatnonzero((distance<=.12)&progress)
    first_near=float(times[near[0]]) if len(near) else None
    eligible=(distance<=.12)&stable&active&(column('skill')==0)&progress
    since=arrival=None
    for t,valid in zip(times,eligible):
        since=(float(t) if since is None else since) if valid else None
        if since is not None and t-since>=.8:arrival=float(t);break
    delay=arrival-first_near if arrival is not None and first_near is not None else None
    departed=bool(arrival is not None and np.any((times>arrival)&progress&(distance>.16)))
    diagnostic=bool(s['episode_completed'] and s['stable_recovery'] and not s['collision'] and not s['fall']
        and s['hold_violation_packets']==0 and s['max_torque_limit_ratio']<=1.+1e-9
        and distance[-1]<=.12 and delay is not None and delay<=6. and not departed
        and s['progress']>=s['required_progress']-.01 and not s['failed_jumps']
        and s['completed_jumps']>=s['required_jumps']
        and (s['maximum_jumps']<0 or s['completed_jumps']<=s['maximum_jumps']))
    return dict(case=s['case'],runtime_success=s['task_fulfilled'],sampled_success=diagnostic,
        agrees=diagnostic==s['task_fulfilled'],sampled_arrival=arrival,
        sampled_delay=delay,runtime_arrival=s['first_goal_arrival'],runtime_delay=s['arrival_stopping_delay'],
        trace_dt=s['trace_dt'],borderline_six_second_limit=bool(delay is not None and abs(delay-6.)<2*s['trace_dt']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for batch in a.batch:
        for line in (batch/'results.jsonl').read_text().splitlines():
            r=json.loads(line);s=r['summary']
            if not s.get('adaptation_config',{}).get('goal_arrival'):continue
            rows.append(dict(batch=str(batch),arm=r['arm'],**check(batch/s['case']/r['arm'])))
    save(a.output,dict(episodes=len(rows),matching_scores=sum(r['agrees'] for r in rows),
        disagreements=[r for r in rows if not r['agrees']],rows=rows,
        limitation='Sampled diagnostic; recorded runtime uses every 2 ms physics step. No original score changed.'))
    print('Independent sampled goal scoring:',sum(r['agrees'] for r in rows),'/',len(rows),'scores agree',flush=True)


if __name__=='__main__':main()
