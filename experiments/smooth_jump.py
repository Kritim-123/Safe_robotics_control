"""Measured torque-driven jump development, paired baselines and finite video demos."""
import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import uuid

import mujoco
import numpy as np

from controllers.navigation import NavigationExperiment
from hybrid.arena import Scenario, build_model
from skills.dynamic import DynamicSkill, execute_dynamic_skill
from skills.jump_profile import JumpConfig

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'output/smooth_jump/development-v1'


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def command_metrics(spec,profile,config):
    robot=NavigationExperiment(build_model(spec))
    skill=DynamicSkill(robot,'jump',0.,jump_config=config if profile=='smooth_forward' else None)
    positions,velocities=[],[]
    previous=robot.home.copy()
    for time in np.arange(0.,2.01,.01):
        target=skill.targets(float(time))
        velocity=skill.analytic_velocity if skill.analytic_velocity is not None else (target-previous)/.01
        positions.append(target.copy());velocities.append(velocity.copy());previous=target.copy()
    return dict(max_command_step=float(np.max(np.abs(np.diff(positions,axis=0)))),
        max_desired_velocity=float(np.max(np.abs(velocities))),
        max_command_velocity_step=float(np.max(np.abs(np.diff(velocities,axis=0)))))


def run_trial(folder,spec,profile='smooth_forward',config=None,*,viewer=False,video=False,interrupt_after=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    config=config or JumpConfig()
    rows=[];renderer=movie=None;next_video=0.
    if video:
        import imageio.v2 as imageio
        movie=imageio.get_writer(str(folder/'recording.mp4'),fps=30)
    def frame(model,data,name):
        nonlocal renderer,next_video
        feet={model.geom(leg).id for leg in ('FL','FR','RL','RR')}
        foot_contacts=set();normal_force=0.
        force=np.zeros(6)
        for i,contact in enumerate(data.contact):
            found=feet & {int(contact.geom1),int(contact.geom2)}
            if found:
                foot_contacts.update(found)
                mujoco.mj_contactForce(model,data,i,force)
                normal_force+=max(0.,float(force[0]))
        foot_height=min(float(data.site(leg+'_foot').xpos[2]) for leg in ('FL','FR','RL','RR'))
        upright=float(data.body('base').xmat.reshape(3,3)[2,2])
        rows.append([float(data.time),*data.qpos[:3],*data.qvel[:3],*data.qacc[:3],
            upright,len(foot_contacts),normal_force,float(not foot_contacts and foot_height>.025),*data.ctrl])
        if movie is not None and data.time>=next_video:
            from PIL import Image,ImageDraw
            if renderer is None:renderer=mujoco.Renderer(model,height=480,width=640)
            cam=mujoco.MjvCamera();cam.lookat[:]=[spec.start[0]+.12*np.cos(spec.start_yaw),spec.start[1]+.12*np.sin(spec.start_yaw),.22]
            cam.distance,cam.azimuth,cam.elevation=2.3,np.rad2deg(spec.start_yaw)+115,-25
            renderer.update_scene(data,camera=cam)
            image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,640,55),fill='black')
            draw.text((8,6),f'{profile} | {spec.name} | t={data.time:.2f}s\n'
                f'Height {data.qpos[2]:.3f} m | foot contacts {len(foot_contacts)} | torque {max(abs(data.ctrl)):.1f} Nm',fill='white')
            movie.append_data(np.asarray(image));next_video+=1/30
    try:
        history,summary=execute_dynamic_skill('jump',spec,jump_profile=profile,
            jump_config=config if profile=='smooth_forward' else None,frame_callback=frame,viewer=viewer,
            interrupt_after=interrupt_after)
    finally:
        if renderer is not None:renderer.close()
        if movie is not None:movie.close()
    matrix=np.asarray(rows)
    np.savez_compressed(folder/'trace.npz',values=matrix,columns=np.asarray(
        ['time','x','y','z','vx','vy','vz','ax','ay','az','upright','foot_contacts','normal_force','airborne']+
        [f'torque_{i}' for i in range(12)]))
    active=matrix[(matrix[:,0]>=2.) & (matrix[:,0]<4.5)]
    yaw=summary['decision_observation']['yaw'] if summary['decision_observation'] else spec.start_yaw
    forward=np.array([np.cos(yaw),np.sin(yaw)])
    progress=float(np.asarray(summary['displacement'])@forward)
    flight_progress=flight_length=0.;start=None;landing_forces=[];landing_velocity=None
    for i,row in enumerate(active):
        if row[13] and start is None:start=i
        if not row[13] and start is not None:
            duration=row[0]-active[start,0]
            if duration>flight_length:
                flight_length=duration
                flight_progress=float((row[1:3]-active[start,1:3])@forward)
                landing_velocity=float(active[max(i-1,0),6])
                landing_forces=active[i:min(len(active),i+150),12].tolist()
            start=None
    metrics=dict(forward_progress=progress,flight_forward_progress=flight_progress,
        measured_flight=flight_length,landing_vertical_velocity=landing_velocity,
        peak_landing_normal_force=max(landing_forces) if landing_forces else None,
        peak_base_acceleration=float(np.max(np.linalg.norm(active[:,7:10],axis=1))) if len(active) else None,
        max_tilt_deg=float(np.rad2deg(np.arccos(np.clip(matrix[:,10].min(),-1.,1.)))),
        torque_step_rms=float(np.sqrt(np.mean(np.diff(active[:,14:],axis=0)**2))) if len(active)>1 else None,
        max_torque_step=float(np.max(np.abs(np.diff(active[:,14:],axis=0)))) if len(active)>1 else None)
    metrics.update(command_metrics(spec,profile,config))
    metrics['qualifies']=bool(summary['success'] and summary['stable_landing'] and not summary['collision']
        and not summary['fall'] and summary['longest_airtime']>=.06 and progress>=.10 and flight_progress>0)
    result=dict(profile=profile,config=asdict(config) if profile=='smooth_forward' else None,
        scenario=spec.to_dict(),summary=summary,metrics=metrics)
    save(folder/'result.json',result)
    return result


def tune(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spec=Scenario('jump_tuning_flat',bounds=(-3.,6.,-3.,3.))
    baseline=run_trial(output/'legacy',spec,'legacy');save(output/'baseline.json',baseline)
    results=[]
    grid=itertools.product((.10,.14,.18,.22,.26),(.10,.14,.18,.22),(28.,32.,36.,40.,48.))
    with (output/'candidates.jsonl').open('x',encoding='utf-8') as stream:
        for index,(lean,thrust,limit) in enumerate(grid):
            config=replace(JumpConfig(),forward_lean=lean,thrust=thrust,torque_limit=limit)
            result=run_trial(output/f'candidate-{index:03d}',spec,config=config)
            row=dict(index=index,config=asdict(config),summary=result['summary'],metrics=result['metrics'])
            results.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            print(index,lean,thrust,limit,result['summary']['reason'],
                  round(result['metrics']['forward_progress'],3),round(result['metrics']['flight_forward_progress'],3),
                  result['metrics']['qualifies'],flush=True)
    qualified=[r for r in results if r['metrics']['qualifies']]
    save(output/'selection.json',dict(qualified=len(qualified),
        candidates=sorted(qualified,key=lambda r:(abs(r['metrics']['forward_progress']-.22),
            r['metrics']['peak_landing_normal_force'] or float('inf')))[:10]))
    return results


def evaluate(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save(output/'frozen_config.json',asdict(config))
    rng=np.random.default_rng(20261025)
    cases=[]
    for index in range(16):
        yaw=float(rng.uniform(-np.pi,np.pi));start=rng.uniform(-.2,.2,2)
        goal=start+3*np.array([np.cos(yaw),np.sin(yaw)])
        cases.append(('held_out',Scenario(f'jump-held-{index:02d}',start=tuple(start),goal=tuple(goal),
            start_yaw=yaw,friction=float(rng.uniform(.65,1.05)),bounds=(-6.,6.,-6.,6.))))
    from hybrid.terrain import TerrainPatch
    for index,friction in enumerate((.35,.5,1.1,1.2)):
        cases.append(('stress',Scenario(f'jump-friction-{index:02d}',start_yaw=.4,
            friction=friction,bounds=(-3.,6.,-3.,3.))))
    for index,height in enumerate((.01,.02)):
        cases.append(('stress',Scenario(f'jump-platform-{index:02d}',bounds=(-3.,6.,-3.,3.),
            terrain=(TerrainPatch('platform',(.5,0.),(1.4,1.5),height),))))
    rows=[]
    with (output/'results.jsonl').open('x',encoding='utf-8') as stream:
        for split,spec in cases:
            for profile in ('legacy','smooth_forward'):
                result=run_trial(output/(spec.name+'-'+profile),spec,profile,config)
                row=dict(split=split,scenario=spec.name,profile=profile,
                         summary=result['summary'],metrics=result['metrics'])
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(split,spec.name,profile,result['summary']['reason'],
                      round(result['metrics']['forward_progress'],3),result['metrics']['qualifies'],flush=True)
    groups=[]
    for split in ('held_out','stress'):
        for profile in ('legacy','smooth_forward'):
            group=[r for r in rows if r['split']==split and r['profile']==profile]
            groups.append(dict(split=split,profile=profile,trials=len(group),
                completed=sum(r['summary']['success'] for r in group),
                forward_qualified=sum(r['metrics']['qualifies'] for r in group),
                collisions=sum(r['summary']['collision'] for r in group),
                falls=sum(r['summary']['fall'] for r in group),
                mean_forward_progress=float(np.mean([r['metrics']['forward_progress'] for r in group])),
                min_forward_progress=min(r['metrics']['forward_progress'] for r in group),
                max_forward_progress=max(r['metrics']['forward_progress'] for r in group),
                mean_flight_progress=float(np.mean([r['metrics']['flight_forward_progress'] for r in group])),
                mean_peak_acceleration=float(np.mean([r['metrics']['peak_base_acceleration'] for r in group])),
                mean_landing_force=float(np.mean([r['metrics']['peak_landing_normal_force'] for r in group
                                                 if r['metrics']['peak_landing_normal_force'] is not None])),
                mean_torque_step_rms=float(np.mean([r['metrics']['torque_step_rms'] for r in group])),
                max_torque=max(r['summary']['max_torque'] for r in group),
                max_tilt=max(r['metrics']['max_tilt_deg'] for r in group)))
    save(output/'aggregate.json',dict(seed=20261025,config=asdict(config),groups=groups))
    return groups


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('tune','demo','evaluate'),default='demo')
    parser.add_argument('--profile',choices=('legacy','smooth_forward'),default='smooth_forward')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--viewer',action='store_true');parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    output=(args.output if args.output.is_absolute() else ROOT/args.output) if args.output else ROOT/'output/smooth_jump'/f'{args.phase}-{uuid.uuid4().hex}'
    if args.phase=='tune':tune(output)
    else:
        config=JumpConfig(**json.loads(args.config.read_text())) if args.config else JumpConfig()
        if args.phase=='evaluate':evaluate(output,config)
        else:
            result=run_trial(output,Scenario('smooth_jump_demo',bounds=(-3.,6.,-3.,3.)),args.profile,config=config,
                             viewer=args.viewer,video=args.video)
            print(json.dumps(result['metrics'],indent=2),flush=True)
