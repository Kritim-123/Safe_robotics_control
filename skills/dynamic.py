"""Bounded classical dynamic primitives in torque-driven MuJoCo physics."""
from contextlib import nullcontext, contextmanager
import time

import mujoco
import numpy as np

from controllers.navigation import NavigationExperiment
from controllers.walking import JointPID
from hybrid.arena import Arena, Scenario, build_model
from skills.avoidance import clearance, path_clearance

DYNAMIC_SKILL_NAMES = ('wait', 'run', 'jump')


@contextmanager
def _release_torques(data):
    try:
        yield
    finally:
        data.ctrl[:] = 0


def dynamic_eligibility(robot, obstacles, bounds, *, duration=4.0, run_speed=.5):
    """Current-state eligibility; caller must keep monitoring moving obstacles."""
    d = robot.data
    position = d.body('base').xpos[:2]
    rot = d.body('base').xmat.reshape(3, 3)
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    stable = (np.linalg.norm(d.qvel[:3]) < .04 and rot[2, 2] > np.cos(np.deg2rad(12))
              and np.linalg.norm(d.qvel[3:6]) < .2 and d.qpos[2] > .23)
    gap = clearance(position, obstacles, bounds)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    run_gap = path_clearance(position, [position+forward*run_speed*duration], obstacles, bounds)
    return dict(wait=bool(stable and gap > .08), run=bool(stable and run_gap > .12),
                jump=bool(stable and gap > .35))


class DynamicSkill:
    """Joint-target generator only; simulator state and PID remain caller-owned."""
    def __init__(self, robot, name, start_time, *, duration=4.0, run_speed=.5):
        if (name not in DYNAMIC_SKILL_NAMES or not np.all(np.isfinite([duration, start_time, run_speed]))
                or duration <= 0 or not 0 < run_speed <= .5 or (name == 'jump' and duration < 1.5)):
            raise ValueError('Expected wait/run/jump and positive finite duration')
        self.robot, self.name, self.started, self.duration = robot, name, start_time, duration
        self.run_speed = run_speed
        self.status = 'running'

    def targets(self, now):
        if not np.isfinite(now) or now < self.started:
            raise ValueError('Target time must be finite and at or after initiation')
        r, elapsed = self.robot, now-self.started
        if self.name == 'run' and elapsed < self.duration and self.status == 'running':
            return r.velocity_targets(np.array([self.run_speed, 0., 0.]), .01)
        if self.name == 'jump' and self.status == 'running':
            home = r.home.reshape(4, 3)
            crouch = np.tile([0., 1.1, -2.2], (4, 1))
            extension = np.tile([0., .45, -.9], (4, 1))
            if elapsed < .5:
                blend = np.clip(elapsed/.5, 0, 1)
                blend = blend*blend*(3-2*blend)
                return (home+(crouch-home)*blend).ravel()
            if elapsed < .62:
                return extension.ravel().copy()
            if elapsed < .68:
                blend = (elapsed-.62)/.06
                return (extension+(home-extension)*blend).ravel()
        return r.velocity_targets(np.zeros(3), .01)

    def interrupt(self):
        self.status = 'interrupted'


def execute_dynamic_skill(name=None, scenario=None, *, duration=4.0, seed=0, viewer=False,
                          timeout=12.0, interrupt_after=None, selector=None, run_speed=.5,
                          frame_callback=None):
    """Execute one episode; return history and honest measured outcome summary.

    Scenario observations use metres/seconds/radians. ``seed`` is recorded for
    provenance; the physical episode is deterministic for a fixed scenario.
    Initial placement is the only qpos assignment. No external forces are used.
    """
    if name not in DYNAMIC_SKILL_NAMES and not (name is None and callable(selector)):
        raise ValueError('Unknown dynamic skill')
    if (not np.all(np.isfinite([duration, timeout, run_speed])) or duration < 1.5 or timeout <= 2
            or not 0 < run_speed <= .5
            or (interrupt_after is not None and (not np.isfinite(interrupt_after) or interrupt_after < 0))):
        raise ValueError('Invalid duration, timeout or interruption time')
    spec = scenario or Scenario('dynamic_'+str(name), goal=(3., 0), bounds=(-3, 6, -3, 3))
    robot = NavigationExperiment(build_model(spec))
    m, d = robot.model, robot.data
    arena = Arena(spec, m)
    mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
    d.qpos[:2] = spec.start
    d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0, 0, np.sin(spec.start_yaw/2)]
    arena.update(d)
    mujoco.mj_forward(m, d)
    pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
    dt = float(m.opt.timestep)
    target, target_velocity = robot.home.copy(), np.zeros(m.nu)
    next_control = next_log = 0.
    skill, observation, eligibility = None, None, None
    history = []
    reason, collision, fall = 'timeout', False, False
    peak_torque = limit_ratio = airtime = longest_airtime = air_streak = 0.
    max_height, min_clearance, stable_since = 0., float('inf'), None
    start_position = d.qpos[:2].copy()
    foot_geoms = {v['geom'] for v in robot.legs.values()}
    ctx = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
    with _release_torques(d), ctx as view:
        if view is not None:
            view.cam.lookat[:] = [.7, 0, .2]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 5.5, 120, -35
        while d.time < timeout:
            wall = time.perf_counter()
            if view is not None and not view.is_running():
                reason = 'viewer closed'
                break
            arena.update(d)
            mujoco.mj_forward(m, d)
            position = d.body('base').xpos[:2].copy()
            rot = d.body('base').xmat.reshape(3, 3)
            yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
            speed = float(np.linalg.norm(d.qvel[:2]))
            obstacles = arena.sensed_obstacles(d)
            gap = clearance(position, obstacles, spec.bounds)
            min_clearance = min(min_clearance, gap)
            if d.time >= 2 and skill is None:
                observation = dict(time=float(d.time), position=position.tolist(), speed=speed,
                                   yaw=yaw, height=float(d.qpos[2]), upright=float(rot[2, 2]),
                                   obstacles=[dict(center=c.tolist(), radius=float(r), velocity=v.tolist())
                                              for c, r, v in obstacles])
                eligibility = dynamic_eligibility(robot, obstacles, spec.bounds, duration=duration, run_speed=run_speed)
                if selector is not None:
                    name = selector(observation, eligibility.copy())
                    if name is None:
                        reason = 'no eligible skill'
                        break
                    if name not in DYNAMIC_SKILL_NAMES:
                        raise ValueError('Selector must return wait, run, or jump')
                if not eligibility[name]:
                    reason = 'failed initiation'
                    break
                start_position = position.copy()
                skill = DynamicSkill(robot, name, float(d.time), duration=duration, run_speed=run_speed)
            if skill and interrupt_after is not None and d.time-skill.started >= interrupt_after:
                skill.interrupt()
            if d.time >= next_control-dt/2:
                previous = target.copy()
                target = robot.home.copy() if skill is None else skill.targets(float(d.time))
                target_velocity = (target-previous)/.01
                next_control += .01
            torque = pid.update(target, target_velocity, d.qpos[robot.qa], d.qvel[robot.va], dt)
            d.ctrl[:] = torque
            peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
            limit_ratio = max(limit_ratio, float(np.max(np.abs(torque)/robot.limits)))
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            gap = clearance(d.body('base').xpos[:2], arena.sensed_obstacles(d), spec.bounds)
            min_clearance = min(min_clearance, gap)
            contact_feet = {g for c in d.contact for g in (c.geom1, c.geom2) if g in foot_geoms}
            foot_z = min(float(d.site_xpos[v['site'], 2]) for v in robot.legs.values())
            airborne = not contact_feet and foot_z > .025
            if skill and airborne:
                airtime += dt
                air_streak += dt
                longest_airtime = max(longest_airtime, air_streak)
            else:
                air_streak = 0.
            height = float(d.qpos[2])
            upright = float(d.body('base').xmat.reshape(3, 3)[2, 2])
            if frame_callback is not None:
                frame_callback(m, d, name if skill is not None else 'settling')
            max_height = max(max_height, height)
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                reason = 'nonfinite state'
                break
            if arena.collided(d):
                collision, reason = True, 'obstacle contact'
                break
            if height < .15 or upright < np.cos(np.deg2rad(35)):
                fall, reason = True, 'height/tilt limit'
                break
            if gap < .035:
                reason = 'clearance stop'
                break
            if skill and (d.time-skill.started >= duration or skill.status == 'interrupted'):
                stable = (robot.activity == 0 and np.linalg.norm(d.qvel[:3]) < .04
                          and np.linalg.norm(d.qvel[3:6]) < .2 and upright > np.cos(np.deg2rad(12))
                          and len(contact_feet) == 4)
                stable_since = (float(d.time) if stable_since is None else stable_since) if stable else None
                if stable_since is not None and d.time-stable_since >= .8:
                    reason = ('interrupted' if skill.status == 'interrupted' else
                              'completed' if name != 'jump' or longest_airtime >= .04 else 'no verified flight')
                    break
            if d.time >= next_log-dt/2:
                history.append(dict(time=float(d.time), x=float(d.qpos[0]), y=float(d.qpos[1]),
                                    height=height, upright=upright, speed=float(np.linalg.norm(d.qvel[:2])),
                                    foot_contacts=len(contact_feet), min_foot_height=foot_z, airborne=airborne,
                                    peak_torque=float(np.max(np.abs(torque))), clearance=gap))
                next_log += .02
            if view is not None:
                view.sync()
                time.sleep(max(0, dt-(time.perf_counter()-wall)))
    # Always retain the terminal state, including failed initiation/early stop.
    contact_feet = {g for c in d.contact for g in (c.geom1, c.geom2) if g in foot_geoms}
    foot_z = min(float(d.site_xpos[v['site'], 2]) for v in robot.legs.values())
    terminal = dict(time=float(d.time), x=float(d.qpos[0]), y=float(d.qpos[1]),
                    height=float(d.qpos[2]), upright=float(d.body('base').xmat.reshape(3, 3)[2, 2]),
                    speed=float(np.linalg.norm(d.qvel[:2])), foot_contacts=len(contact_feet),
                    min_foot_height=foot_z, airborne=not contact_feet and foot_z > .025,
                    peak_torque=float(np.max(np.abs(torque))) if 'torque' in locals() else 0.,
                    clearance=clearance(d.qpos[:2], arena.sensed_obstacles(d), spec.bounds))
    if history and history[-1]['time'] == terminal['time']:
        history[-1] = terminal
    else:
        history.append(terminal)
    delta = d.qpos[:2]-start_position
    summary = dict(skill=name, seed=seed, success=reason == 'completed', reason=reason,
                   elapsed=float(d.time), displacement=delta.tolist(), distance=float(np.linalg.norm(delta)),
                   airtime=airtime, longest_airtime=longest_airtime, all_feet_airborne=longest_airtime >= .04,
                   max_height=max_height, stable_landing=stable_since is not None and d.time-stable_since >= .8,
                   max_torque=peak_torque, max_torque_limit_ratio=limit_ratio, collision=collision, fall=fall,
                   min_clearance=min_clearance, decision_observation=observation, eligibility=eligibility,
                   physics_dt=dt, control_dt=.01, duration=duration, run_speed=run_speed)
    return history, summary
