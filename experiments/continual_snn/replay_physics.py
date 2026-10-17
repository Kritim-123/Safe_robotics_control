"""Independent torque-only MuJoCo replay of a complete 500 Hz recorded episode."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from controllers.navigation import NavigationExperiment
from hybrid.arena import Arena,Scenario,Obstacle,build_model
from hybrid.terrain import TerrainPatch
from .study import save


def replay(folder,output):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    if abs(summary['trace_dt']-summary['physics_dt'])>1e-12:raise ValueError('Need a full physics-rate trace')
    values=json.loads((folder/'scenario.json').read_text())['scenario']
    values['obstacles']=tuple(Obstacle(**o) for o in values['obstacles'])
    values['terrain']=tuple(TerrainPatch(**p) for p in values.get('terrain',()))
    spec=Scenario(**values);robot=NavigationExperiment(build_model(spec));m,d=robot.model,robot.data
    arena=Arena(spec,m);mujoco.mj_resetDataKeyframe(m,d,robot.home_id)
    d.qpos[:2]=spec.start;d.qpos[3:7]=[np.cos(spec.start_yaw/2),0,0,np.sin(spec.start_yaw/2)]
    # Recorded commands already obey the original per-skill limits. Widening
    # the replay's clamp to the maximum admits both 24 and 28 N m commands.
    m.actuator_ctrlrange[:]=[-28.,28.];m.actuator_forcerange[:]=[-28.,28.]
    arena.update(d);mujoco.mj_forward(m,d)
    with np.load(folder/'trace.npz',allow_pickle=False) as trace:
        recorded=trace['values'];columns=trace['columns'].tolist()
    qi=[columns.index('qpos_'+str(i)) for i in range(19)]
    vi=[columns.index('qvel_'+str(i)) for i in range(18)]
    ti=[columns.index('torque_'+str(i)) for i in range(12)]
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    handoffs=[e for e in events if e['event']=='handoff'];support=[]
    base_id=m.body('base').id;robot_bodies=set()
    for body in range(m.nbody):
        ancestor=body
        while ancestor and ancestor!=base_id:ancestor=int(m.body_parentid[ancestor])
        if ancestor==base_id:robot_bodies.add(body)
    robot_geoms={i for i,b in enumerate(m.geom_bodyid) if int(b) in robot_bodies}
    feet={leg['geom'] for leg in robot.legs.values()}
    max_q=max_v=0.;force=np.zeros(6)
    for row in recorded:
        arena.update(d);mujoco.mj_forward(m,d)
        for event in handoffs:
            if abs(d.time-event['time'])<1e-8:
                supported=set();normal=0.
                for index,contact in enumerate(d.contact):
                    a,b=int(contact.geom1),int(contact.geom2)
                    foot=a if a in feet and b not in robot_geoms else b if b in feet and a not in robot_geoms else None
                    if foot is not None:
                        mujoco.mj_contactForce(m,d,index,force)
                        if force[0]>1e-3:supported.add(foot);normal+=float(force[0])
                support.append(dict(time=float(d.time),previous=event['previous'],next=event['next'],
                    loaded_feet=len(supported),total_normal_force=normal))
        if abs(row[0]-d.time-m.opt.timestep)>1e-8:raise ValueError('Noncontiguous recorded physics time')
        d.ctrl[:]=row[ti]
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        max_q=max(max_q,float(np.max(np.abs(d.qpos-row[qi]))))
        max_v=max(max_v,float(np.max(np.abs(d.qvel-row[vi]))))
        if max_q>1e-6 or max_v>1e-5:raise AssertionError(dict(time=float(d.time),qpos=max_q,qvel=max_v))
    report=dict(episode=summary['case'],model_checksum=summary['model_checksum'],steps=len(recorded),physics_dt=float(m.opt.timestep),
        maximum_absolute_qpos_error=max_q,maximum_absolute_qvel_error=max_v,
        method='Initial placement once; replay recorded motor commands and obstacle mocap only; no PID, SNN, state correction or external force',
        supported_handoffs=support,expected_handoffs=len(handoffs),
        all_handoffs_four_loaded_feet=len(support)==len(handoffs) and all(s['loaded_feet']==4 for s in support))
    save(output,report)
    if not report['all_handoffs_four_loaded_feet']:raise AssertionError(report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(replay(args.episode,args.output))
