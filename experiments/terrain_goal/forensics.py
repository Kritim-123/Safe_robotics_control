"""Read-only reconstruction of recorded contact geometry and paired outcomes."""
import argparse,json
from collections import Counter
from pathlib import Path
import mujoco
import numpy as np
from controllers.navigation import NavigationExperiment
from hybrid.arena import Arena,Scenario,Obstacle,build_model
from hybrid.terrain import TerrainPatch
from experiments.continual_snn.study import save
from .evidence import rows_for,failure_signals


def contact_geometry(episode):
    values=json.loads((episode/'scenario.json').read_text())['scenario']
    values['obstacles']=tuple(Obstacle(**o) for o in values['obstacles'])
    values['terrain']=tuple(TerrainPatch(**p) for p in values.get('terrain',()))
    spec=Scenario(**values);robot=NavigationExperiment(build_model(spec));m,d=robot.model,robot.data
    arena=Arena(spec,m)
    with np.load(episode/'trace.npz',allow_pickle=False) as bundle:
        a=bundle['values'];cols=bundle['columns'].tolist()
    qi=[cols.index('qpos_'+str(i)) for i in range(m.nq)]
    vi=[cols.index('qvel_'+str(i)) for i in range(m.nv)]
    contacts=[]
    for row in a:
        d.time=float(row[0]);d.qpos[:]=row[qi];d.qvel[:]=row[vi]
        arena.update(d);mujoco.mj_forward(m,d)
        if not arena.collided(d):continue
        for c in d.contact:
            x,y=int(c.geom1),int(c.geom2)
            hazard=((x in arena.hazard_geoms and y in arena.robot_geoms) or
                    (y in arena.hazard_geoms and x in arena.robot_geoms) or
                    (x in arena.terrain_geoms and y in arena.nonfoot_robot_geoms) or
                    (y in arena.terrain_geoms and x in arena.nonfoot_robot_geoms))
            if hazard:
                contacts.append(dict(time=float(d.time),geom1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,x),
                    geom2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,y),
                    body1=m.body(int(m.geom_bodyid[x])).name,body2=m.body(int(m.geom_bodyid[y])).name,
                    contact_distance=float(c.dist),contact_position=c.pos.tolist(),base_position=d.qpos[:3].tolist()))
        break
    return dict(method='Recorded states set only for offline geometry reconstruction; no new controller execution',
                first_recorded_disallowed_contacts=contacts)


def paired(rows,left,right):
    groups={}
    for r in rows:groups.setdefault((r['summary']['case'],r['summary']['seed']),{})[r['arm']]=r['summary']
    complete=[g for g in groups.values() if left in g and right in g]
    counts=Counter((g[left]['task_fulfilled'],g[right]['task_fulfilled']) for g in complete)
    return dict(left=left,right=right,matched_scenes=len(complete),
        both_success=counts[(True,True)],left_only_success=counts[(True,False)],
        right_only_success=counts[(False,True)],both_failure=counts[(False,False)],
        left_regressions=[dict(case=g[left]['case'],signals=failure_signals(g[left]),
            final_goal_distance=g[left].get('final_goal_distance'),
            arrival_stopping_delay=g[left].get('arrival_stopping_delay'))
            for g in complete if not g[left]['task_fulfilled'] and g[right]['task_fulfilled']],
        left_failure_signals=dict(Counter(signal for g in complete if not g[left]['task_fulfilled']
            for signal in failure_signals(g[left]).split(';'))),
        right_failure_signals=dict(Counter(signal for g in complete if not g[right]['task_fulfilled']
            for signal in failure_signals(g[right]).split(';'))))


def paired_pose_noise(batch):
    """Compare measured position innovations against independently logged state."""
    cases=json.loads((batch/'cases.json').read_text());results=[]
    for case in cases[:3]:
        name=case['scenario']['name'];innovations={}
        for arm in ('v1','v2','rule'):
            episode=batch/name/arm
            if not (episode/'summary.json').exists():continue
            events={r['time']:r for r in map(json.loads,(episode/'events.jsonl').read_text().splitlines())
                    if r['event']=='decision_packet'}
            decisions=list(map(json.loads,(episode/'decisions.jsonl').read_text().splitlines()))
            innovations[arm]=np.asarray([np.asarray(d['observation']['position'])-np.asarray(events[d['time']]['qpos'][:2])
                for d in decisions])
        if len(innovations)!=3:continue
        n=min(map(len,innovations.values()));reference=innovations['v1'][:n]
        maximum=max(float(np.max(np.abs(reference-x[:n]))) for x in innovations.values())
        results.append(dict(case=name,common_packets=n,maximum_absolute_noise_difference=maximum,
            empirical_position_noise_std=float(np.std(reference)),same_innovations=maximum<1e-12))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('output/terrain_goal'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reports={}
    for phase in ('clean','mild','severe'):
        rows=rows_for(a.root,'hysteresis-final-'+phase+'-v2')
        reports[phase]=[paired(rows,'v2','v1'),paired(rows,'v2','rule')]
        reports[phase+'_noise_pairing']=paired_pose_noise(a.root/('hysteresis-final-'+phase+'-v2'))
    episode=a.root/'ood-contact-replay-v2'/'goal_ood-goal-14200-014'/'v2'
    if (episode/'summary.json').exists():reports['ood_contact']=contact_geometry(episode)
    for name,pattern in (('refinement_clean','request-refinement-confirmation-v3-*'),
                         ('refinement_noise','request-refinement-noise-v3')):
        reports[name]=paired(rows_for(a.root,pattern),'candidate','v2')
    save(a.output,reports)


if __name__=='__main__':main()
