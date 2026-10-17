"""Persistent physics with repeated SNN requests, checked handoffs and telemetry."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.navigation import NavigationExperiment
from controllers.walking import JointPID
from hybrid.arena import Arena,build_model
from skills.avoidance import clearance
from skills.dynamic import DynamicSkill,_release_torques
from skills.jump_profile import JumpConfig
from .models import ROOT,FrozenBankModel
from .observations import SensorStream,SensorConfig,snapshot,feature_packet,request_masks,teacher
from .supervisor import HandoffConfig,HandoffSupervisor


def execute(case,output,*,model=None,mode='continuous',config=None,sensor_config=None,
            run_speed=.35,predict_horizon=1.6,viewer=False,video=False,full_trace=False,
            frame_callback=None):
    """One fresh episode, with no intermediate reset and one sole PID writer.

    one_shot freezes the first neural preference but retains the SAME ongoing
    eligibility/braking supervisor. rule uses a causal geometry/task teacher.
    These task streams assess request fulfillment, not general navigation.
    """
    if mode not in ('continuous','one_shot','rule','always_run','always_wait'):
        raise ValueError('Unknown selection mode')
    if not np.isfinite([run_speed,predict_horizon]).all() or not 0<run_speed<=.5 or not .5<=predict_horizon<=4:
        raise ValueError('Invalid run speed/prediction horizon')
    case.validate()
    folder=Path(output).resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Episode output must remain inside project output')
    folder.mkdir(parents=True,exist_ok=False)
    model=model or (FrozenBankModel() if mode in ('continuous','one_shot') else None)
    if model is not None:model.reset()
    handoffs=HandoffSupervisor(config)
    sensors=SensorStream(sensor_config or SensorConfig(seed=case.seed))
    robot=NavigationExperiment(build_model(case.scenario));m,d=robot.model,robot.data
    arena=Arena(case.scenario,m)
    mujoco.mj_resetDataKeyframe(m,d,robot.home_id)
    d.qpos[:2]=case.scenario.start
    d.qpos[3:7]=[np.cos(case.scenario.start_yaw/2),0.,0.,np.sin(case.scenario.start_yaw/2)]
    arena.update(d);mujoco.mj_forward(m,d)
    pid=JointPID(robot.kp,robot.ki,robot.kd,robot.limits)
    dt=float(m.opt.timestep)
    jump_config=JumpConfig()
    target=robot.home.copy();target_velocity=np.zeros(12)
    last_torque=np.zeros(12)
    next_control=next_observation=next_trace=next_video=0.
    skill=None;first_proposal=None;has_first=False
    masks=dict(wait=True,run=False,jump=False)
    true_start=np.asarray(case.scenario.start).copy()
    final_stable_since=jump_stable_since=None
    jump_air=jump_longest=0.;jump_reported=False
    completed_jumps=failed_jumps=cancelled_jumps=hold_violations=0
    peak_torque=limit_ratio=peak_target_step=0.
    minimum_gap=clearance(d.qpos[:2],arena.sensed_obstacles(d),case.scenario.bounds);max_tilt=0.
    collision=fall=False;reason='timeout'
    trajectories=[];latencies=[];decisions=[];skill_outcomes=[]
    renderer=movie=None
    if video:
        import imageio.v2 as imageio
        movie=imageio.get_writer(str(folder/'recording.mp4'),fps=30)
    def set_limits(bound):
        robot.limits=np.full(12,bound);pid.limits=robot.limits.copy()
        m.actuator_ctrlrange[:]=np.column_stack((-robot.limits,robot.limits))
        m.actuator_forcerange[:]=m.actuator_ctrlrange
    def record_event(item):
        if item is None:return
        events.write(json.dumps(dict(item,qpos=d.qpos.tolist(),qvel=d.qvel.tolist()),allow_nan=False)+'\n')
        events.flush()
    context=mujoco.viewer.launch_passive(m,d) if viewer else nullcontext(None)
    try:
        with (folder/'decisions.jsonl').open('x',encoding='utf-8') as packet_log, \
                (folder/'events.jsonl').open('x',encoding='utf-8') as events, \
                _release_torques(d),context as view:
            wall_started=time.perf_counter();next_view=0.
            while d.time<case.timeout:
                if view and not view.is_running():reason='viewer closed';break
                arena.update(d);mujoco.mj_forward(m,d)
                current=snapshot(robot,arena)
                if d.time>=next_observation-dt/2:
                    captured,observed_obstacles=sensors.observe(float(d.time),current['obstacles'])
                    observed=dict(current,obstacles=observed_obstacles,capture_time=captured)
                    task=case.request_at(float(d.time))
                    features=feature_packet(case.scenario,observed,task)
                    masks,geometry=request_masks(case.scenario,observed,run_speed=run_speed,
                                                 predict_horizon=predict_horizon)
                    from .predictive import PREDICTIVE_FEATURES
                    if model is not None and all(n in model.encoder.feature_names for n in PREDICTIVE_FEATURES):
                        features.update({name:geometry[name] for name in PREDICTIVE_FEATURES})
                    if d.time>=2.-dt/2:
                        if handoffs.entered_at is None:
                            handoffs.start(float(d.time));record_event(handoffs.events[-1])
                            true_start=d.qpos[:2].copy()
                            skill=DynamicSkill(robot,'wait',float(d.time),duration=case.timeout+1.,run_speed=run_speed)
                        recommended=teacher(task,masks)
                        inference_start=time.perf_counter();proposal={};evaluated=False
                        if mode=='rule':
                            proposal=dict(proposed_skill=recommended,reason='causal_geometry_teacher')
                        elif mode=='always_run':
                            proposal=dict(proposed_skill='run',reason='fixed run request')
                        elif mode=='always_wait':
                            proposal=dict(proposed_skill='wait',reason='fixed wait request')
                        elif mode=='one_shot' and has_first:
                            proposal=first_proposal
                        else:
                            proposal=model.predict(features,now=float(d.time));evaluated=True
                            if mode=='one_shot':first_proposal=proposal;has_first=True
                        latency=time.perf_counter()-inference_start
                        if evaluated:latencies.append(latency)
                        event=handoffs.observe(float(d.time),proposal['proposed_skill'],masks,
                            jump_requested=task=='jump',observation_time=captured)
                        record_event(event)
                        decision=dict(time=float(d.time),capture_time=captured,task=task,features=features,
                            observation=observed,request_mask=masks,geometry=geometry,
                            teacher_request=recommended,label_source='causal geometry/task teacher; not outcome label',
                            proposal=proposal,desired=handoffs.desired,current=handoffs.current,
                            mode=handoffs.mode,model_evaluated=evaluated,inference_seconds=latency,
                            packet_counts_cleared=bool(evaluated and proposal.get('spike_counts') is not None))
                        decisions.append(decision)
                        packet_log.write(json.dumps(decision,allow_nan=False)+'\n');packet_log.flush()
                        for request in case.requests:
                            if request.task=='hold' and request.start+case.hold_grace<=d.time<min(request.end,case.finish_after):
                                hold_violations+=int(current['linear_speed']>=.04 or current['angular_speed']>=.2)
                    next_observation+=handoffs.config.decision_dt
                if handoffs.entered_at is not None and d.time>=next_control-dt/2:
                    if d.time>=case.finish_after:
                        handoffs.desired='wait'  # Common finite-episode termination.
                    stable=current['stable']
                    entry={name:bool(allowed and stable) for name,allowed in masks.items()}
                    jump_finished=False
                    if skill is not None and skill.name=='jump':
                        elapsed=float(d.time-skill.started)
                        if (elapsed<jump_config.preparation and skill.status!='interrupted'
                                and (handoffs.desired!='jump' or not masks['jump']
                                     or handoffs.last_observation is None
                                     or d.time-handoffs.last_observation>handoffs.config.stale_after)):
                            skill.interrupt()
                            record_event(dict(time=float(d.time),event='jump_preparation_cancelled',
                                              pending_request=handoffs.desired,jump_mask=masks['jump']))
                        cancelled=(skill.status=='interrupted' and
                                   skill.forward_jump.interrupted_at<jump_config.preparation)
                        finished_time=(jump_config.preparation+jump_config.recovery if cancelled
                                       else sum(skill.forward_jump.durations))
                        jump_stable_since=(float(d.time) if jump_stable_since is None else jump_stable_since) if stable else None
                        jump_finished=(elapsed>=finished_time and jump_stable_since is not None
                                       and d.time-jump_stable_since>=.8)
                        if jump_finished and not jump_reported:
                            success=jump_longest>=.04 and not cancelled
                            completed_jumps+=int(success);failed_jumps+=int(not success and not cancelled)
                            cancelled_jumps+=int(cancelled)
                            if cancelled:handoffs.jump_consumed=False
                            result=dict(time=float(d.time),skill='jump',success=success,longest_airtime=jump_longest,
                                        stable_recovery=True,started=skill.started,cancelled_before_thrust=cancelled)
                            skill_outcomes.append(result);record_event(dict(event='skill_outcome',**result))
                            jump_reported=True
                    event=handoffs.advance(float(d.time),stable=stable,entry_mask=entry,
                        continuation_mask=masks,jump_finished=jump_finished)
                    if event:
                        record_event(event)
                        if event['event']=='no_eligible_fallback':reason='no eligible fallback';break
                        if event['event']=='braking' and skill is not None:
                            skill.interrupt()
                        elif event['event']=='handoff':
                            skill=DynamicSkill(robot,event['next'],float(d.time),duration=case.timeout+1.,run_speed=run_speed,
                                jump_config=jump_config if event['next']=='jump' else None)
                            set_limits(jump_config.torque_limit if event['next']=='jump' else 24.)
                            pid.integral[:]=0.
                            jump_air=jump_longest=0.;jump_stable_since=None;jump_reported=False
                    previous=target.copy()
                    if skill is None:
                        target=robot.home.copy();target_velocity=np.zeros(12)
                    else:
                        target=skill.targets(float(d.time))
                        target_velocity=(skill.analytic_velocity.copy() if skill.analytic_velocity is not None
                                         else (target-previous)/.01)
                    blend=(skill.forward_jump.gain_blend(skill.last_elapsed)
                           if skill is not None and skill.forward_jump is not None else 0.)
                    pid.kp=robot.kp+(jump_config.landing_kp-robot.kp)*blend
                    pid.kd=robot.kd+(jump_config.landing_kd-robot.kd)*blend
                    peak_target_step=max(peak_target_step,float(np.max(np.abs(target-previous))))
                    next_control+=.01
                elif handoffs.entered_at is None and d.time>=next_control-dt/2:
                    next_control+=.01
                d.ctrl[:]=pid.update(target,target_velocity,d.qpos[robot.qa],d.qvel[robot.va],dt)
                last_torque=d.ctrl.copy()
                peak_torque=max(peak_torque,float(np.max(np.abs(d.ctrl))))
                limit_ratio=max(limit_ratio,float(np.max(np.abs(d.ctrl)/robot.limits)))
                mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                terminal=snapshot(robot,arena)
                gap=clearance(d.qpos[:2],arena.sensed_obstacles(d),case.scenario.bounds)
                minimum_gap=min(minimum_gap,gap)
                tilt=float(np.rad2deg(np.arccos(np.clip(terminal['upright'],-1,1))))
                max_tilt=max(max_tilt,tilt)
                if skill is not None and skill.name=='jump':
                    feet_high=min(float(d.site_xpos[v['site'],2]) for v in robot.legs.values())>.025
                    jump_air=jump_air+dt if terminal['foot_contacts']==0 and feet_high else 0.
                    jump_longest=max(jump_longest,jump_air)
                if full_trace or d.time>=next_trace-dt/2:
                    trajectories.append([float(d.time),*d.qpos,*d.qvel,*d.ctrl,
                        ('wait','run','jump').index(handoffs.current),terminal['foot_contacts'],gap,
                        terminal['upright'],float(robot.activity),*robot.commanded_velocity])
                    next_trace+=.02
                if frame_callback:frame_callback(m,d,handoffs)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():reason='nonfinite state';break
                if arena.collided(d):collision=True;reason='obstacle contact';break
                if d.qpos[2]<.15 or terminal['upright']<np.cos(np.deg2rad(35)):
                    fall=True;reason='height/tilt limit';break
                if gap<.035:reason='clearance stop';break
                if handoffs.switches>32:reason='switch limit';break
                if d.time>=case.finish_after:
                    final_stable_since=(float(d.time) if final_stable_since is None else final_stable_since) if terminal['stable'] else None
                    if (final_stable_since is not None and d.time-final_stable_since>=.8
                            and handoffs.current=='wait' and handoffs.mode=='active'):
                        reason='completed';break
                if movie is not None and d.time>=next_video:
                    from PIL import Image,ImageDraw
                    if renderer is None:renderer=mujoco.Renderer(m,height=480,width=800)
                    cam=mujoco.MjvCamera();cam.lookat[:]=[*d.qpos[:2],.22]
                    cam.distance,cam.azimuth,cam.elevation=3.5,np.rad2deg(case.scenario.start_yaw)+110,-28
                    renderer.update_scene(d,camera=cam)
                    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,0,800,62),fill='black')
                    draw.text((10,6),f'{case.family} | {mode} | t={d.time:.1f}s | '
                        f'{handoffs.current}/{handoffs.mode}\n'
                        f'Task {case.request_at(float(d.time))} | desired {handoffs.desired} | '
                        f'completed hops {completed_jumps} | switches {handoffs.switches}\n'
                        'Fresh spike-count packets; persistent robot physics',fill='white')
                    panel=Image.new('RGB',(800,640),(18,22,29));panel.paste(frame,(0,0))
                    draw=ImageDraw.Draw(panel)
                    latest=next((x for x in reversed(decisions) if x['model_evaluated']),None)
                    counts=latest['proposal'].get('spike_counts') if latest else None
                    draw.text((10,490),'Output spike counts in this packet (0-16); cleared before next inference',fill='white')
                    if counts:
                        width=760/len(counts)
                        for index,count in enumerate(counts):
                            x=20+index*width
                            color=(64,166,242) if index<64 else (93,213,143)
                            draw.rectangle((x,590-count*4,x+max(1,width-2),590),fill=color)
                        active=latest['proposal'].get('active_neuron_ids',[])
                        draw.text((10,603),f'packet t={latest["time"]:.2f}s | active IDs {active[:24]}',fill='white')
                        draw.text((10,620),'Blue: sensory neurons | Green: predictive-clearance neurons',fill='white')
                    movie.append_data(np.asarray(panel));next_video+=1/30
                if view and d.time>=next_view:
                    view.cam.lookat[:]=[*d.qpos[:2],.22];view.cam.distance=3.5
                    view.sync();next_view=float(d.time)+1/60
                    time.sleep(max(0.,d.time-(time.perf_counter()-wall_started)))
    except BaseException as error:
        (folder/'error.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),
            time=float(d.time)),indent=2)+'\n',encoding='utf-8')
        raise
    finally:
        if renderer is not None:renderer.close()
        if movie is not None:movie.close()
    terminal=snapshot(robot,arena)
    # Retain the terminal state even when a contact occurs between trace ticks.
    if not trajectories or abs(trajectories[-1][0]-d.time)>1e-9:
        trajectories.append([float(d.time),*d.qpos,*d.qvel,*last_torque,
            ('wait','run','jump').index(handoffs.current),terminal['foot_contacts'],
            clearance(d.qpos[:2],arena.sensed_obstacles(d),case.scenario.bounds),terminal['upright'],
            float(robot.activity),*robot.commanded_velocity])
    forward=np.array([np.cos(case.scenario.start_yaw),np.sin(case.scenario.start_yaw)])
    progress=float((d.qpos[:2]-true_start)@forward)
    stable_recovery=bool(final_stable_since is not None and d.time-final_stable_since>=.8)
    fulfilled=bool(reason=='completed' and stable_recovery and progress>=case.required_progress-.01
        and completed_jumps>=case.required_jumps and hold_violations==0 and failed_jumps==0
        and (case.maximum_jumps is None or completed_jumps<=case.maximum_jumps))
    summary=dict(case=case.scenario.name,family=case.family,split=case.split,seed=case.seed,
        mode=mode,model_name=getattr(model,'name',mode),model_checksum=getattr(model,'checksum',None),
        episode_completed=reason=='completed',task_fulfilled=fulfilled,reason=reason,elapsed=float(d.time),
        progress=progress,required_progress=case.required_progress,completed_jumps=completed_jumps,
        required_jumps=case.required_jumps,maximum_jumps=case.maximum_jumps,
        failed_jumps=failed_jumps,cancelled_jumps=cancelled_jumps,hold_violation_packets=hold_violations,
        stable_recovery=stable_recovery,collision=collision,fall=fall,minimum_clearance=minimum_gap,
        max_tilt=max_tilt,max_torque=peak_torque,max_torque_limit_ratio=limit_ratio,
        max_joint_target_step=peak_target_step,switches=handoffs.switches,
        handoff_count=sum(e['event']=='handoff' for e in handoffs.events),decision_packets=len(decisions),
        neural_evaluations=len(latencies),latency_median=float(np.median(latencies)) if latencies else None,
        latency_p95=float(np.percentile(latencies,95)) if latencies else None,
        sensor_config=asdict(sensors.config),handoff_config=asdict(handoffs.config),
        run_speed=run_speed,predict_horizon=predict_horizon,physics_dt=dt,control_dt=.01,
        trace_dt=dt if full_trace else .02,skill_outcomes=skill_outcomes,
        final_position=d.qpos[:3].tolist(),final_linear_speed=terminal['linear_speed'],
        final_angular_speed=terminal['angular_speed'],reset_policy='initial placement only',executor_contract_version=2)
    for name,value in (('scenario.json',case.to_dict()),('summary.json',summary)):
        (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(folder/'trace.npz',values=np.asarray(trajectories),columns=np.asarray(
        ['time']+[f'qpos_{i}' for i in range(19)]+[f'qvel_{i}' for i in range(18)]+
        [f'torque_{i}' for i in range(12)]+['skill','foot_contacts','clearance','upright','gait_activity',
                                        'command_vx','command_vy','command_yaw']))
    return summary
