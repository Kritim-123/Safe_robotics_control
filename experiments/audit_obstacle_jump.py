"""Recompute low-bar crossing evidence from retained 500 Hz foot trajectories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mujoco
import numpy as np

from experiments.smooth_jump import ROOT, save
from hybrid.arena import Scenario, Obstacle, build_model


def audit(root):
    root = Path(root).resolve()
    evidence = []
    for batch in ('validation-v1','axis-validation-v1'):
        for line in (root/batch/'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            summary = row.get('summary',row)
            evidence.append((batch,row.get('split','axis_held_out'),summary))
    nominal = json.loads((root/'release-demo-v2/summary.json').read_text())
    evidence.append(('release-demo-v2','demo',nominal))
    config = json.loads((ROOT/'docs/agents/obstacle_jump/config.json').read_text())
    hashes = {}
    for batch, split, s in evidence:
        path = root/batch if split=='demo' else root/batch/s['scenario']['name']
        trace = np.load(path/'trace.npz')
        hashes[str((path/'trace.npz').relative_to(root))] = hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest()
        assert s['config']==config
        for key in ('time','qpos','qvel','torque','feet','body_rear','upright'):
            assert np.isfinite(trace[key]).all()
        assert abs(trace['time'][-1]-s['elapsed'])<1e-10
        assert np.all(np.diff(trace['time'])>0)
        assert np.max(abs(trace['torque']))<=72.+1e-10
        spec = s['scenario'];bar=next(o for o in spec['obstacles'] if o['name']=='bar')
        heading = np.array([np.cos(spec['start_yaw']),np.sin(spec['start_yaw']),0.])
        origin = np.r_[bar['center'],0.]
        feet = trace['feet']; x=(feet-origin)@heading
        active=trace['time']>s['observation']['time']
        for i in range(4):
            overlap=active & (x[:,i]+.0175>=-bar['half_extents'][0]) & (x[:,i]-.0175<=bar['half_extents'][0])
            margin=float(np.min(feet[overlap,i,2]-.0175-bar['height'])) if overlap.any() else None
            assert margin is None and s['min_foot_bar_clearance'][i] is None or abs(margin-s['min_foot_bar_clearance'][i])<1e-10
        streak=longest=0
        for flight in (active & (trace['foot_contacts']==0) & (np.min(feet[:,:,2]-.0175,axis=1)>.005)):
            streak=streak+1 if flight else 0
            longest=max(streak,longest)
        assert abs(longest*.002-s['longest_airtime'])<1e-10
        if s['success']:
            assert s['stable_landing'] and not s['collision'] and not s['fall'] and not s['target_contacts']
            assert np.all(x[-1]-.0175>bar['half_extents'][0]+.003)
            assert min(s['min_foot_bar_clearance'])>.002
            assert s['final_collision_geometry_margin']>.003
            assert np.linalg.norm(trace['qvel'][-1,:3])<.04
            assert np.linalg.norm(trace['qvel'][-1,3:6])<.2
    # Independently reconstruct contacts in the actual model for EVERY demo
    # frame, rather than accepting the runner's collision flag on its own.
    spec_values=nominal['scenario'].copy()
    spec_values['obstacles']=tuple(Obstacle(**o) for o in spec_values['obstacles'])
    spec=Scenario(**spec_values);model=build_model(spec);data=mujoco.MjData(model)
    trace=np.load(root/'release-demo-v2/trace.npz')
    bar_id=model.geom('obstacle_geom_0').id
    for qpos,qvel in zip(trace['qpos'],trace['qvel']):
        data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data)
        assert not any(bar_id in (c.geom1,c.geom2) for c in data.contact)
    aggregate={}
    for split in dict.fromkeys(split for _,split,_ in evidence):
        group=[s for _,group_split,s in evidence if group_split==split]
        aggregate[split]=dict(trials=len(group),successes=sum(s['success'] for s in group),
            reasons=dict(Counter(s['reason'] for s in group)),
            collisions=sum(s['collision'] for s in group),falls=sum(s['fall'] for s in group))
    result=dict(audited_traces=len(evidence),groups=aggregate,trace_sha256=hashes,
        demo=dict(forward_travel=nominal['final_position'][0]-nominal['observation']['position'][0],
                  longest_airtime=nominal['longest_airtime'],min_foot_bar_clearance=min(nominal['min_foot_bar_clearance']),
                  full_robot_margin=nominal['final_collision_geometry_margin']),
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'skills/obstacle_jump.py','skills/jump_profile.py','experiments/obstacle_jump.py',
            'experiments/audit_obstacle_jump.py','tests/test_obstacle_jump.py')})
    save(root/'audit.json',result)
    times=trace['time'];feet=trace['feet'];selected=(times>=2.3)&(times<=3.5)
    fig,ax=plt.subplots(figsize=(9,4.6))
    for i,name in enumerate(('FL','FR','RL','RR')):
        ax.plot(feet[selected,i,0],feet[selected,i,2]-.0175,label=name,linewidth=2,alpha=.8)
    ax.add_patch(Rectangle((.23,0),.02,.04,color='#2585ad',label='Actual bar cross-section'))
    ax.axhline(0,color='gray',linewidth=.8);ax.set_ylim(-.015,.46)
    ax.set_xlabel('World forward position (m)');ax.set_ylabel('Foot sphere bottom height (m)')
    ax.set_title('Recorded 4 cm-bar jump: every foot passes above the obstacle')
    ax.legend(ncol=3,frameon=False);fig.tight_layout();fig.savefig(root/'foot-clearance.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(groups=aggregate,demo=result['demo']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'output/obstacle_jump')
    audit(parser.parse_args().root)
