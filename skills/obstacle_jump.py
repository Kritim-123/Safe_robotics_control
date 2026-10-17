"""Explicit static low-bar jump, with full physical contact and crossing checks.

This standalone motor-skill runner does not change any SNN output schema.
"""
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
from hybrid.arena import Arena, Scenario, Obstacle, build_model
from skills.avoidance import clearance, path_clearance
from skills.dynamic import _release_torques
from skills.jump_profile import ForwardJumpTrajectory, JumpConfig

FEET = ('FL', 'FR', 'RL', 'RR')
PROFILE = Path(__file__).resolve().parents[1]/'docs/agents/obstacle_jump/config.json'


def default_scene():
    return Scenario('low_bar_jump',goal=(1.5,0.),bounds=(-4.,5.,-4.,4.),obstacles=(
        Obstacle('bar',(.24,0.),shape='box',half_extents=(.01,.6),height=.04),))


def execute_low_bar_jump(*, scenario=None, target_name='bar', **kwargs):
    """Experimental frozen profile, restricted to assessed +world-X low bars.

    Nearby tests include landing-contact failures; entry is not a guarantee of
    crossing. Arbitrary headings failed evaluation and are refused here. Use
    the explicit-config development executor for other diagnostic scenes.
    """
    spec = scenario or default_scene()
    target, heading, lateral = target_contract(spec,target_name)
    delta = np.r_[np.asarray(target.center)-spec.start,0.]
    if (abs(np.arctan2(heading[1],heading[0])) > 1e-8 or abs(delta@lateral) > 1e-8
            or not .23-1e-10 <= delta@heading <= .24+1e-10
            or not .02 <= target.height <= .04 or not .006 <= target.half_extents[0] <= .01
            or target.half_extents[1] < .6 or not .8 <= spec.friction <= .9
            or spec.push_time >= 0 or np.linalg.norm(spec.start) > .3):
        raise ValueError('Outside assessed +X low-bar envelope; use a different skill')
    config = JumpConfig(**json.loads(PROFILE.read_text(encoding='utf-8')))
    return execute_obstacle_jump(spec,target_name,config,**kwargs)


def projection_radius(model, data, geom, direction):
    """Support radius of a collision primitive along a world-space direction."""
    local = data.geom_xmat[geom].reshape(3, 3).T @ direction
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[0]+size[1]*abs(local[2]))
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[0]*np.linalg.norm(local[:2])+size[1]*abs(local[2]))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(local) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(local*size))
    return float(model.geom_rbound[geom])  # Conservative for other shapes.


def target_contract(spec, target_name):
    spec.validate()
    targets = [o for o in spec.obstacles if o.name == target_name]
    if len(targets) != 1:
        raise ValueError('Expected one explicitly named low-bar obstacle')
    target = targets[0]
    heading = np.array([np.cos(spec.start_yaw), np.sin(spec.start_yaw), 0.])
    lateral = np.array([-heading[1], heading[0], 0.])
    delta = np.r_[np.asarray(target.center)-spec.start, 0.]
    angle = np.arctan2(np.sin(target.yaw-spec.start_yaw), np.cos(target.yaw-spec.start_yaw))
    if (target.shape != 'box' or target.motion != 'linear' or np.linalg.norm(target.velocity) > 0
            or abs(angle) > 1e-6 or abs(delta@lateral) > .015 or spec.terrain
            or not .23 <= delta@heading <= .36 or not .005 <= target.height <= .04
            or not .005 <= target.half_extents[0] <= .025 or target.half_extents[1] < .5):
        raise ValueError('Target outside static aligned low-bar development contract')
    return target, heading, lateral


def execute_obstacle_jump(spec, target_name, config, *, timeout=10., interrupt_after=None,
                          viewer=False, frame_callback=None):
    """Place once, settle, launch under PID, and verify every foot and collision geom.

    The named low target is exempt only from planar clearance. Its MuJoCo
    contacts still abort immediately. Other hazards retain planar guards.
    ``config`` is explicit to keep development distinct from the default hop.
    """
    if not isinstance(config, JumpConfig):
        raise ValueError('Supply an explicit JumpConfig')
    config.validate()
    if (not np.isfinite(timeout) or not 4 < timeout <= 20 or
            (interrupt_after is not None and (not np.isfinite(interrupt_after) or interrupt_after < 0))):
        raise ValueError('Invalid timeout or interruption')
    target_spec, heading, lateral = target_contract(spec, target_name)
    robot = NavigationExperiment(build_model(spec))
    m, d = robot.model, robot.data
    arena = Arena(spec, m)
    target_index = next(i for i, o in enumerate(spec.obstacles) if o.name == target_name)
    target_geom = m.geom(f'obstacle_geom_{target_index}').id
    foot_ids = [m.geom(leg).id for leg in FEET]
    radii = m.geom_size[foot_ids, 0].copy()
    collision_geoms = [g for g in arena.robot_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
    origin = np.r_[target_spec.center, 0.]
    near, far = -target_spec.half_extents[0], target_spec.half_extents[0]
    mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
    d.qpos[:2] = spec.start
    d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0., 0., np.sin(spec.start_yaw/2)]
    arena.update(d); mujoco.mj_forward(m, d)
    pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
    curve = ForwardJumpTrajectory(robot, config)
    target, velocity = robot.home.copy(), np.zeros(12)
    next_control = 0.; dt = float(m.opt.timestep)
    started = stable_since = None
    observation = None
    interrupted = collision = fall = False
    streak = longest = peak = 0.
    foot_min = np.full(4, np.inf)
    crossed = np.zeros(4, dtype=bool)
    overlap_seen = np.zeros(4, dtype=bool)
    stayed_in_lane = True
    foot_positions = d.geom_xpos[foot_ids].copy()
    feet_x = (foot_positions-origin)@heading
    body_rear = min(float((d.geom_xpos[g]-origin)@heading)-projection_radius(m,d,g,heading)
                    for g in collision_geoms)
    rows, contacts = [], []
    reason = 'timeout'
    context = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
    with _release_torques(d), context as view:
        wall_start = time.perf_counter(); next_view = 0.
        if view:
            view.cam.lookat[:] = [*(np.asarray(spec.start)+heading[:2]*.3), .20]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 2.1, np.rad2deg(spec.start_yaw)+90., -18.
        while d.time < timeout:
            if view and not view.is_running():
                reason = 'viewer closed'; break
            arena.update(d); mujoco.mj_forward(m, d)
            others = [o for i, o in enumerate(arena.sensed_obstacles(d)) if i != target_index]
            foot_positions = d.geom_xpos[foot_ids].copy()
            feet_x = (foot_positions-origin)@heading
            supported = {g for c in d.contact for g in (int(c.geom1), int(c.geom2)) if g in foot_ids}
            upright = float(d.body('base').xmat.reshape(3, 3)[2, 2])
            if started is None and d.time >= 2.:
                stable = (len(supported) == 4 and np.linalg.norm(d.qvel[:3]) < .04 and
                          np.linalg.norm(d.qvel[3:6]) < .2 and upright > np.cos(np.deg2rad(12)))
                corridor = path_clearance(d.qpos[:2], [d.qpos[:2]+heading[:2]*1.2], others, spec.bounds)
                observation = dict(time=float(d.time), position=d.qpos[:3].tolist(),
                    foot_positions=foot_positions.tolist(), stable=bool(stable), corridor=float(corridor))
                if not stable or not np.all(feet_x+radii < near-.003) or corridor <= .15:
                    reason = 'failed initiation'; break
                started = float(d.time)
                robot.limits = np.full(12, config.torque_limit)
                pid.limits = robot.limits.copy()
                m.actuator_ctrlrange[:] = np.column_stack((-robot.limits, robot.limits))
                m.actuator_forcerange[:] = m.actuator_ctrlrange
            elapsed = float(d.time-started) if started is not None else None
            if elapsed is not None and interrupt_after is not None and elapsed >= interrupt_after:
                curve.interrupt(elapsed); interrupted = True
            if d.time >= next_control-dt/2:
                if elapsed is not None:
                    target, velocity, _, phase = curve.sample(elapsed)
                    blend = curve.gain_blend(elapsed)
                    pid.kp = robot.kp+(config.landing_kp-robot.kp)*blend
                    pid.kd = robot.kd+(config.landing_kd-robot.kd)*blend
                else:
                    phase = 'settling'
                next_control += .01
            d.ctrl[:] = pid.update(target, velocity, d.qpos[robot.qa], d.qvel[robot.va], dt)
            peak = max(peak, float(np.max(np.abs(d.ctrl))))
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            foot_positions = d.geom_xpos[foot_ids].copy()
            feet_x = (foot_positions-origin)@heading
            feet_y = (foot_positions-origin)@lateral
            bottoms = foot_positions[:, 2]-radii
            supported = {g for c in d.contact for g in (int(c.geom1), int(c.geom2)) if g in foot_ids}
            upright = float(d.body('base').xmat.reshape(3, 3)[2, 2])
            if started is not None:
                overlap = (feet_x+radii >= near) & (feet_x-radii <= far)
                foot_min[overlap] = np.minimum(foot_min[overlap], bottoms[overlap]-target_spec.height)
                overlap_seen |= overlap
                crossed |= feet_x-radii > far+.003
                stayed_in_lane &= bool(np.all(np.abs(feet_y)+radii < target_spec.half_extents[1]))
                streak = streak+dt if not supported and min(bottoms) > .005 else 0.
                longest = max(longest, streak)
            body_rear = min(float((d.geom_xpos[g]-origin)@heading)-projection_radius(m, d, g, heading)
                            for g in collision_geoms)
            gap = clearance(d.qpos[:2], others, spec.bounds)
            for c in d.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if target_geom in pair and pair & arena.robot_geoms:
                    contacts.append(dict(time=float(d.time), geom1=m.geom(c.geom1).name,
                                         geom2=m.geom(c.geom2).name,
                                         body1=m.body(m.geom_bodyid[c.geom1]).name,
                                         body2=m.body(m.geom_bodyid[c.geom2]).name,
                                         distance=float(c.dist)))
            rows.append(dict(time=float(d.time), phase=phase, qpos=d.qpos.copy().tolist(),
                qvel=d.qvel.copy().tolist(), torque=d.ctrl.copy().tolist(),
                feet=foot_positions.tolist(), foot_contacts=len(supported), body_rear=body_rear,
                upright=upright, normal_clearance=gap))
            if frame_callback:
                frame_callback(m, d, rows[-1])
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                reason = 'nonfinite state'; break
            if arena.collided(d):
                collision = True; reason = 'obstacle contact'; break
            if d.qpos[2] < .15 or upright < np.cos(np.deg2rad(35)):
                fall = True; reason = 'height/tilt limit'; break
            if gap < .035:
                reason = 'clearance stop'; break
            stable = (len(supported) == 4 and np.linalg.norm(d.qvel[:3]) < .04 and
                      np.linalg.norm(d.qvel[3:6]) < .2 and upright > np.cos(np.deg2rad(12)))
            if elapsed is not None and elapsed >= 2.:
                stable_since = (float(d.time) if stable_since is None else stable_since) if stable else None
                if stable_since is not None and d.time-stable_since >= .8:
                    cleared = (np.all(overlap_seen) and np.all(crossed) and np.all(foot_min > .002)
                               and np.all(feet_x-radii > far+.003) and body_rear > far+.003
                               and stayed_in_lane and longest >= .04)
                    reason = 'interrupted' if interrupted else 'completed' if cleared else 'not cleared'
                    break
            if view and d.time >= next_view:
                view.sync(); next_view = float(d.time)+1/60
                time.sleep(max(0., d.time-(time.perf_counter()-wall_start)))
    summary = dict(success=reason == 'completed', reason=reason, elapsed=float(d.time),
        initiated=started is not None, observation=observation, config=asdict(config),
        target=target_name, scenario=spec.to_dict(), interrupted=interrupted,
        stable_landing=stable_since is not None and d.time-stable_since >= .8,
        collision=collision, fall=fall, target_contacts=contacts, longest_airtime=longest,
        foot_crossed=crossed.tolist(), foot_overlap_seen=overlap_seen.tolist(),
        min_foot_bar_clearance=[float(v) if np.isfinite(v) else None for v in foot_min],
        final_feet_beyond=(feet_x-radii > far+.003).tolist(),
        final_collision_geometry_margin=body_rear-far, stayed_in_lane=stayed_in_lane,
        max_torque=peak, actuator_limit=float(max(robot.limits)),
        physics_dt=dt, control_dt=.01, final_position=d.qpos[:3].tolist())
    return rows, summary
