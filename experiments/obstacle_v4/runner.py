"""One persistent physics episode: observed counts -> jump -> V3 navigation."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.walking import JointPID
from hybrid.arena import Arena, build_model
from skills.avoidance import path_clearance, clearance
from skills.dynamic import DynamicSkill, _release_torques
from skills.jump_profile import ForwardJumpTrajectory, JumpConfig
from skills.obstacle_jump import PROFILE, projection_radius, target_contract
from experiments.continual_snn.observations import snapshot, feature_packet
from experiments.terrain_goal.geometry import shape_request_masks
from experiments.terrain_goal.loading_v2 import load_any
from experiments.terrain_goal.assessment import primary_config
from experiments.terrain_goal.terrain import TerrainGait
from .model import JumpHead, ROOT, V3, observe


def eligibility(spec, target_index, robot, arena):
    """Veto only. Never supplies the neural jump decision."""
    if target_index is None:
        return False, 'no target'
    target = spec.obstacles[target_index]
    try:
        target_contract(spec, target.name)
    except ValueError:
        return False, 'outside aligned static low-bar contract'
    delta = np.array(target.center)-spec.start
    if (abs(spec.start_yaw) > 1e-8 or abs(delta[1]) > 1e-8
            or not .23-1e-10 <= delta[0] <= .24+1e-10
            or not .02 <= target.height <= .04 or not .006 <= target.half_extents[0] <= .01
            or target.half_extents[1] < .6 or not .8 <= spec.friction <= .9
            or spec.push_time >= 0 or np.linalg.norm(spec.start) > .3):
        return False, 'outside assessed +X entry envelope'
    d, m = robot.data, robot.model
    current = snapshot(robot, arena)
    if not current['stable']:
        return False, 'not stopped on four supported feet'
    if abs(current['yaw']) > .02:
        return False, 'heading drift'
    feet = [m.geom(name).id for name in ('FL', 'FR', 'RL', 'RR')]
    near = target.center[0]-target.half_extents[0]
    if not np.all(d.geom_xpos[feet, 0]+m.geom_size[feet, 0] < near-.003):
        return False, 'feet not entirely behind target'
    others = [(o.at(d.time), o.planning_radius, o.speed_at(d.time))
              for i, o in enumerate(spec.obstacles) if i != target_index]
    predicted_gap = min(path_clearance(d.qpos[:2], [d.qpos[:2]+np.array([1.2, 0.])],
                                      [(c+v*t, r, v) for c, r, v in others], spec.bounds)
                        for t in np.linspace(0., 3., 17))
    if predicted_gap <= .15:
        return False, 'landing corridor blocked'
    return True, 'eligible'


def execute(spec, output, *, viewer=False, video=False, timeout=30., head=None, interrupt_after=None, viewer_speed=1.):
    spec.validate()
    if not np.isfinite(timeout) or not 5 <= timeout <= 90:
        raise ValueError('Invalid finite episode timeout')
    if interrupt_after is not None and (not np.isfinite(interrupt_after) or interrupt_after < 0):
        raise ValueError('Invalid interruption time')
    if not np.isfinite(viewer_speed) or not .1 <= viewer_speed <= 2.:
        raise ValueError('Invalid viewer playback speed')
    folder = Path(output).resolve()
    if not folder.is_relative_to(ROOT/'output'):
        raise ValueError('Keep episode evidence inside output/')
    folder.mkdir(parents=True, exist_ok=False)
    head = head or JumpHead.load()
    if hashlib.sha256(PROFILE.read_bytes()).hexdigest() != head.metadata['jump_profile_sha256']:
        raise ValueError('Jump profile changed')
    v3 = load_any(V3); v3.reset()
    config = primary_config()
    jump_config = JumpConfig(**json.loads(PROFILE.read_text()))
    robot = TerrainGait(build_model(spec), spec, config)
    m, d = robot.model, robot.data
    arena = Arena(spec, m)
    mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
    d.qpos[:2] = spec.start
    d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0., 0., np.sin(spec.start_yaw/2)]
    arena.update(d); mujoco.mj_forward(m, d)
    pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
    target, velocity = robot.home.copy(), np.zeros(12)
    dt = float(m.opt.timestep)
    feet = [m.geom(name).id for name in ('FL', 'FR', 'RL', 'RR')]
    radii = m.geom_size[feet, 0].copy()
    collision_geoms = [g for g in arena.robot_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
    phase, skill = 'settling', None
    curve = None; jumped_target = None; passed = set()
    confirmations = 0; last_candidate = None; stable_since = goal_since = None
    jump_started = None; jump_completed = False; cancelled = False
    overlap_seen = np.zeros(4, dtype=bool); crossed = overlap_seen.copy()
    foot_min = np.full(4, np.inf); air = longest_air = 0.; stayed_in_lane = True
    rear_margin = None; peak = ratio = 0.; max_target_step = 0.
    next_control = next_packet = next_view = next_video = 0.
    events = []; decisions = []; trace = []; last_head = {}; last_v3 = {}
    reason = 'timeout'; collision = fall = False; contact_details = []
    distance = float(np.linalg.norm(np.array(spec.goal)-d.qpos[:2]))
    renderer = movie = None
    if video:
        import imageio.v2 as imageio
        movie = imageio.get_writer(str(folder/'recording.mp4'), fps=30)

    def event(name, **detail):
        row = dict(time=float(d.time), event=name, **detail)
        events.append(row); print(json.dumps(row), flush=True)

    def limits(bound):
        robot.limits = np.full(12, bound); pid.limits = robot.limits.copy()
        m.actuator_ctrlrange[:] = np.column_stack((-robot.limits, robot.limits))
        m.actuator_forcerange[:] = m.actuator_ctrlrange
        pid.integral[:] = 0.

    limits(24.)
    context = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
    try:
        with _release_torques(d), context as view:
            wall_start = time.perf_counter()
            while d.time < timeout:
                if view and not view.is_running():
                    reason = 'viewer closed'; break
                arena.update(d); mujoco.mj_forward(m, d)
                state = snapshot(robot, arena)
                now = float(d.time)
                if now >= next_packet-dt/2:
                    features, candidate = observe(spec, d.qpos[:2], state['yaw'], state['speed'], state['stable'], now)
                    last_head = head.predict(features)
                    allowed, veto = eligibility(spec, candidate, robot, arena)
                    if now >= 2. and phase == 'settling':
                        phase = 'wait'; event('settled', next='wait')
                    pending_jump = (phase == 'wait' and candidate not in passed
                                    and last_head['proposed_skill'] == 'low_bar_jump' and allowed)
                    confirmations = confirmations+1 if pending_jump and candidate == last_candidate else int(pending_jump)
                    last_candidate = candidate
                    # Goal remains progress throughout; no explicit jump task exists.
                    observed = dict(state, obstacles=[o for i, o in enumerate(state['obstacles']) if i not in passed])
                    active_spec = replace(spec, obstacles=tuple(o for i, o in enumerate(spec.obstacles) if i not in passed))
                    masks, geometry = shape_request_masks(active_spec, observed, run_speed=.35, predict_horizon=1.6,
                                                          goal=spec.goal)
                    v3_features = feature_packet(active_spec, observed, 'progress')
                    v3_features['footprint_clearance'] = geometry['current_clearance']
                    from experiments.continual_snn.predictive import PREDICTIVE_FEATURES
                    v3_features.update({name: geometry[name] for name in PREDICTIVE_FEATURES})
                    v3_features = v3.augment_features(v3_features, task='progress', geometry=geometry, observed=observed)
                    v3_features.update(goal_mode=1., run_active=float(phase == 'run'),
                                       braking_active=float(phase == 'braking'))
                    last_v3 = v3.predict(v3_features, now=now)
                    decisions.append(dict(time=now, mission='progress', jump_features=features,
                                          target_index=candidate, jump_proposal=last_head, eligible=allowed,
                                          veto=veto, phase=phase, confirmations=confirmations,
                                          v3_features=v3_features, v3_proposal=last_v3, request_mask=masks))
                    if confirmations >= 2 and phase == 'wait':
                        curve = ForwardJumpTrajectory(robot, jump_config)
                        jump_started = now; jumped_target = candidate; stable_since = None
                        phase = 'low_bar_jump'; limits(jump_config.torque_limit)
                        event('handoff', previous='wait', next=phase,
                              source='learned_obstacle_spike_counts', target=spec.obstacles[candidate].name)
                    elif phase in ('wait', 'goal_hold') and (candidate is None or candidate in passed) and not jump_completed:
                        if last_v3['proposed_skill'] == 'run' and masks['run'] and state['stable']:
                            phase = 'run'; skill = DynamicSkill(robot, 'run', now, duration=timeout+1., run_speed=.35)
                            event('handoff', previous='wait', next='run', source='V3')
                    elif phase == 'run' and (last_v3['proposed_skill'] != 'run' or not masks['run']):
                        phase = 'braking'; skill.interrupt(); event('braking', source='V3_or_eligibility')
                    next_packet += .1
                if now >= next_control-dt/2:
                    previous = target.copy()
                    if phase == 'low_bar_jump':
                        elapsed = now-jump_started
                        if interrupt_after is not None and elapsed >= interrupt_after and not cancelled:
                            curve.interrupt(elapsed); cancelled = True; event('jump_interrupted')
                        target, velocity, _, _ = curve.sample(elapsed)
                        blend = curve.gain_blend(elapsed)
                        pid.kp = robot.kp+(jump_config.landing_kp-robot.kp)*blend
                        pid.kd = robot.kd+(jump_config.landing_kd-robot.kd)*blend
                    elif skill is not None and phase in ('run', 'braking'):
                        target = skill.targets(now)
                        velocity = (target-previous)/.01
                    else:
                        target = robot.home.copy(); velocity = np.zeros(12)
                    max_target_step = max(max_target_step, float(np.max(np.abs(target-previous))))
                    next_control += .01
                d.ctrl[:] = pid.update(target, velocity, d.qpos[robot.qa], d.qvel[robot.va], dt)
                peak = max(peak, float(np.max(np.abs(d.ctrl))))
                ratio = max(ratio, float(np.max(np.abs(d.ctrl)/robot.limits)))
                mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
                state = snapshot(robot, arena)
                trace.append([float(d.time), *d.qpos, *d.qvel, *d.ctrl])
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    reason = 'nonfinite state'; break
                if arena.collided(d):
                    collision = True; reason = 'obstacle contact'
                    for contact in d.contact:
                        pair = {int(contact.geom1), int(contact.geom2)}
                        if pair & arena.robot_geoms and pair & arena.hazard_geoms:
                            contact_details.append(dict(time=float(d.time),
                                                        geom1=m.geom(contact.geom1).name,
                                                        geom2=m.geom(contact.geom2).name))
                    break
                if d.qpos[2] < .15 or state['upright'] < np.cos(np.deg2rad(35)):
                    fall = True; reason = 'height/tilt limit'; break
                if jump_started is not None and not jump_completed:
                    bar = spec.obstacles[jumped_target]
                    feet_x = d.geom_xpos[feet, 0]-bar.center[0]
                    feet_y = d.geom_xpos[feet, 1]-bar.center[1]
                    bottoms = d.geom_xpos[feet, 2]-radii
                    near, far = -bar.half_extents[0], bar.half_extents[0]
                    overlap = (feet_x+radii >= near) & (feet_x-radii <= far)
                    overlap_seen |= overlap; crossed |= feet_x-radii > far+.003
                    foot_min[overlap] = np.minimum(foot_min[overlap], bottoms[overlap]-bar.height)
                    stayed_in_lane &= bool(np.all(np.abs(feet_y)+radii < bar.half_extents[1]))
                    air = air+dt if state['foot_contacts'] == 0 and min(bottoms) > .005 else 0.
                    longest_air = max(longest_air, air)
                    rear_margin = min(float(d.geom_xpos[g, 0]-bar.center[0])
                                      -projection_radius(m, d, g, np.array([1., 0., 0.]))
                                      for g in collision_geoms)-far
                    if d.time-jump_started >= max(2., sum(curve.durations)):
                        stable_since = (float(d.time) if stable_since is None else stable_since) if state['stable'] else None
                        if stable_since is not None and d.time-stable_since >= .8:
                            cleared = (not cancelled and np.all(overlap_seen) and np.all(crossed)
                                       and np.all(foot_min > .002) and np.all(feet_x-radii > far+.003)
                                       and rear_margin > .003 and stayed_in_lane and longest_air >= .04)
                            event('jump_outcome', success=bool(cleared), rear_margin=rear_margin,
                                  longest_airtime=longest_air)
                            if not cleared:
                                reason = 'interrupted' if cancelled else 'not fully cleared'; break
                            passed.add(jumped_target); jump_completed = True; phase = 'run'
                            limits(24.); pid.kp = robot.kp; pid.kd = robot.kd
                            skill = DynamicSkill(robot, 'run', float(d.time), duration=timeout+1., run_speed=.35)
                            event('handoff', previous='low_bar_jump', next='run', source='verified full crossing')
                if phase == 'braking' and state['stable']:
                    phase = 'goal_hold'; skill = None; event('handoff', previous='run', next='wait')
                distance = float(np.linalg.norm(np.array(spec.goal)-d.qpos[:2]))
                arrived = phase == 'goal_hold' and distance <= .12 and state['stable']
                goal_since = (float(d.time) if goal_since is None else goal_since) if arrived else None
                if goal_since is not None and d.time-goal_since >= .8:
                    reason = 'completed'; break
                if (not jump_started and d.time >= 6. and phase == 'wait' and state['stable']):
                    reason = 'refused'; break
                # The stationary target may lie inside the conservative planar
                # robot disc while the actual foot/body geometry is clear.
                # Exemption permits holding/committed crossing only, not walking
                # through an unpassed target. Physical collisions always abort.
                exempt = jumped_target if jumped_target is not None else candidate if phase in ('settling', 'wait') else None
                others = [(o.at(d.time), o.planning_radius, o.speed_at(d.time))
                          for i, o in enumerate(spec.obstacles) if i not in passed and i != exempt]
                if clearance(d.qpos[:2], others, spec.bounds) < .035:
                    reason = 'clearance stop'; break
                if movie and d.time >= next_video:
                    from PIL import Image, ImageDraw
                    if renderer is None:
                        renderer = mujoco.Renderer(m, height=480, width=896)
                    camera = mujoco.MjvCamera(); camera.lookat[:] = [*d.qpos[:2], .22]
                    camera.distance, camera.azimuth, camera.elevation = 2.8, 110., -24.
                    renderer.update_scene(d, camera=camera)
                    image = Image.new('RGB', (896, 640), (18, 22, 29))
                    image.paste(Image.fromarray(renderer.render()), (0, 0))
                    draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 896, 56), fill='black')
                    draw.text((10, 7), f'V4 | t={d.time:.1f}s | {phase} | mission: reach goal (no jump request)', fill='white')
                    draw.text((10, 28), f'Head: {last_head.get("proposed_skill")} | crossed: {jump_completed} | goal {distance:.2f} m', fill='white')
                    counts = last_head.get('spike_counts', [])
                    for i, count in enumerate(counts):
                        x = 15+i*865/max(1, len(counts))
                        draw.rectangle((x, 600-count*5, x+3, 600), fill=(80, 180, 240))
                    draw.text((10, 615), 'Fresh observation LIF counts (0-16); frozen V3 handles walking/goal stopping', fill='white')
                    movie.append_data(np.asarray(image)); next_video += 1/30
                if view and d.time >= next_view:
                    view.cam.lookat[:] = [*d.qpos[:2], .22]
                    view.cam.distance, view.cam.azimuth, view.cam.elevation = 2.8, 110., -24.
                    view.sync(); next_view = float(d.time)+1/60
                    time.sleep(max(0., d.time/viewer_speed-(time.perf_counter()-wall_start)))
    finally:
        if renderer is not None:
            renderer.close()
        if movie is not None:
            movie.close()
        (folder/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        (folder/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
        np.savez_compressed(folder/'trace.npz', values=np.array(trace))
    summary = dict(case=spec.name, success=reason == 'completed', reason=reason, elapsed=float(d.time),
                   jump_initiated=jump_started is not None, jump_completed=jump_completed,
                   collision=collision, fall=fall, contacts=contact_details, final_goal_distance=distance,
                   terminal_stable=bool(snapshot(robot, arena)['stable']), torques_released=bool(np.all(d.ctrl == 0)),
                   longest_airtime=longest_air, final_collision_geometry_margin=rear_margin,
                   min_foot_bar_clearance=[float(v) if np.isfinite(v) else None for v in foot_min],
                   max_torque=peak, max_torque_limit_ratio=ratio, max_joint_target_step=max_target_step,
                   reset_policy='initial placement only', mission='progress', explicit_jump_requests=0,
                   head_checksum=head.checksum, v3_checksum=v3.checksum,
                   jump_config=asdict(jump_config), scenario=spec.to_dict(),
                   neural_jump_proposals=sum(r['jump_proposal']['proposed_skill'] == 'low_bar_jump' for r in decisions),
                   vetoed_jump_proposals=sum(r['jump_proposal']['proposed_skill'] == 'low_bar_jump' and not r['eligible'] for r in decisions))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
