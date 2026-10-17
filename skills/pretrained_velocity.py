"""Bounded, persistent MuJoCo adapter for two frozen rl_sar Go2 policies.

The networks supply joint targets; the existing JointPID remains the sole
torque writer. No Torch or YAML dependency is needed for runtime inference.
"""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.walking import WalkingExperiment, JointPID
from hybrid.arena import Arena, Scenario, build_model
from skills.avoidance import clearance, path_clearance
from skills.dynamic import _release_torques

POLICY_ROOT = Path(__file__).resolve().parent/'policies'/'rl_sar'
COMMANDS = {'forward': (.5, 0., 0.), 'reverse': (-.3, 0., 0.),
            'sidestep_left': (0., .3, 0.), 'sidestep_right': (0., -.3, 0.),
            'turn_left': (0., 0., .5), 'turn_right': (0., 0., -.5)}


def _mlp(weights, prefix, x, layers):
    for i in range(layers):
        x = weights[f'{prefix}.{2*i}.weight'] @ x + weights[f'{prefix}.{2*i}.bias']
        if i != layers-1:
            x = np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
    return x


def numeric_forward(weights, policy, observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.shape != ((45,) if policy == 'robot_lab' else (270,)) or not np.isfinite(x).all():
        raise ValueError('Invalid pretrained policy observation')
    if policy == 'himloco':
        latent = _mlp(weights, 'estimator', x, 3)
        z = latent[3:] / max(float(np.linalg.norm(latent[3:])), 1e-12)
        x = np.r_[x[:45], latent[:3], z]
    elif policy != 'robot_lab':
        raise ValueError('Unknown pretrained policy')
    result = _mlp(weights, 'actor', x, 4)
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError('Nonfinite or invalid pretrained output')
    return result


class PretrainedVelocityPolicy:
    """Caller supplies 50 Hz ticks; reset only at supported skill entry.

    step() never resets/steps MuJoCo or writes data.ctrl. Histories persist
    across changes in command, including the stop/recovery command.
    """
    def __init__(self, model, policy='robot_lab', root=POLICY_ROOT):
        if policy not in ('robot_lab', 'himloco'):
            raise ValueError('Expected robot_lab or himloco')
        self.policy, self.model = policy, model
        folder = Path(root)/policy
        self.manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        self.config = c = self.manifest['config']
        for file, expected in self.manifest['artifacts'].items():
            if hashlib.sha256((folder/file).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Policy artifact checksum mismatch: {file}')
        with np.load(folder/'model.npz', allow_pickle=False) as values:
            self.weights = {k: values[k].copy() for k in values.files}
        if not all(np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError('Nonfinite policy parameters')
        actual_names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        if model.nu != 12 or set(actual_names) != set(c['joint_names']):
            raise ValueError('Expected twelve named Go2 joint actuators')
        self.actuators = np.array([actual_names.index(n) for n in c['joint_names']])
        joints = model.actuator_trnid[self.actuators, 0]
        self.qa, self.va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.ranges = model.jnt_range[joints]
        self.base_id = model.body('base').id
        self.default = np.asarray(c['default_dof_pos'], dtype=np.float32)
        self.scale = np.asarray(c['action_scale'], dtype=np.float32)
        self.reset()

    def reset(self):
        self.history = np.zeros((6 if self.policy == 'himloco' else 1, 45), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_input = self.history.ravel().copy()
        self.ticks = 0

    def frame(self, data, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all() or np.any(np.abs(command) > np.array([.5, .3, .5])+1e-6):
            raise ValueError('Command exceeds tested component envelope')
        velocity = np.zeros(6)
        # mjOBJ_BODY local velocity uses the inertial frame (ximat), which is
        # rotated on this Go2. Transform world angular velocity with xmat to
        # obtain the link/IMU frame expected by the policy.
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        rotation = data.body('base').xmat.reshape(3, 3)
        c = self.config
        terms = dict(ang_vel=(rotation.T @ velocity[:3])*c['ang_vel_scale'],
                     gravity_vec=rotation.T @ np.array([0., 0., -1.]),
                     commands=command*np.asarray(c['commands_scale']),
                     dof_pos=(data.qpos[self.qa]-self.default)*c['dof_pos_scale'],
                     dof_vel=data.qvel[self.va]*c['dof_vel_scale'], actions=self.last_action)
        frame = np.concatenate([terms[k] for k in c['observations']]).astype(np.float32)
        if not np.isfinite(frame).all():
            raise ValueError('Nonfinite robot observation')
        return np.clip(frame, -c['clip_obs'], c['clip_obs'])

    def step(self, data, command):
        frame = self.frame(data, command)
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = frame
        self.last_input = self.history.ravel().copy()
        action = numeric_forward(self.weights, self.policy, self.last_input)
        action = np.clip(action, self.config['clip_actions_lower'], self.config['clip_actions_upper'])
        target = np.clip(self.default+self.scale*action, self.ranges[:, 0], self.ranges[:, 1])
        # Feed back the command actually applied after joint-limit clipping.
        self.last_action = ((target-self.default)/self.scale).astype(np.float32)
        result = np.empty(12)
        result[self.actuators] = target
        self.ticks += 1
        return result


def can_enter(robot, policy, skill, obstacles, bounds, duration=4.):
    """Conservative flat-ground/current-state gate, not a moving-path proof."""
    if skill not in COMMANDS or not np.isfinite(duration) or not 0 < duration <= 4:
        raise ValueError('Unknown skill or duration outside (0, 4] seconds')
    d = robot.data
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        return False
    rot = d.body('base').xmat.reshape(3, 3)
    supported = len(({int(c.geom1) for c in d.contact} | {int(c.geom2) for c in d.contact})
                    & {v['geom'] for v in robot.legs.values()}) == 4
    stable = (supported and d.qpos[2] > .23 and rot[2, 2] > np.cos(np.deg2rad(12))
              and np.linalg.norm(d.qvel[:3]) < .04 and np.linalg.norm(d.qvel[3:6]) < .2)
    command = np.asarray(COMMANDS[skill])
    target = d.qpos[:2] + rot[:2, :2] @ command[:2]*duration
    return bool(stable and clearance(d.qpos[:2], obstacles, bounds) > .15
                and path_clearance(d.qpos[:2], [target], obstacles, bounds) > .15)


def execute_pretrained_skill(policy='himloco', skill='forward', *, scenario=None,
                             duration=4., timeout=12., interrupt_after=None,
                             viewer=False, output=None, frame_callback=None):
    """Run one experimental command option; report motion AND stable recovery.

    Initial placement happens once. Command changes and recovery use the same
    policy, history, PID and physics. This standalone runner does not assert
    that the legacy gait -> learned policy -> legacy gait handoff is validated.
    """
    if skill not in COMMANDS or policy not in ('robot_lab', 'himloco'):
        raise ValueError('Unknown learned policy or command skill')
    if (not np.isfinite([duration, timeout]).all() or not 0 < duration <= 4 or timeout <= 2
            or (interrupt_after is not None and (not np.isfinite(interrupt_after) or interrupt_after < 0))):
        raise ValueError('Invalid duration, timeout or interruption')
    spec = scenario or Scenario('pretrained_'+skill, goal=(3., 0.), bounds=(-4., 6., -4., 4.))
    spec.validate()
    robot = WalkingExperiment(build_model(spec))
    m, d = robot.model, robot.data
    arena = Arena(spec, m)
    adapter = PretrainedVelocityPolicy(m, policy)
    mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
    d.qpos[:2] = spec.start
    d.qpos[2] = .32
    d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0, 0, np.sin(spec.start_yaw/2)]
    d.qpos[adapter.qa] = adapter.default
    arena.update(d)
    mujoco.mj_forward(m, d)
    target = np.empty(12)
    target[adapter.actuators] = adapter.default
    c = adapter.config
    kp, kd, limits = [np.empty(12) for _ in range(3)]
    kp[adapter.actuators], kd[adapter.actuators] = c['rl_kp'], c['rl_kd']
    limits[adapter.actuators] = c['torque_limits']
    limits = np.minimum(limits, robot.limits)
    standing_kp, standing_kd = np.empty(12), np.empty(12)
    standing_kp[adapter.actuators], standing_kd[adapter.actuators] = c['fixed_kp'], c['fixed_kd']
    pid = JointPID(standing_kp, np.zeros(12), standing_kd, limits)
    dt = float(m.opt.timestep)
    feet = {v['geom'] for v in robot.legs.values()}
    floor = m.geom('arena_floor').id
    command = np.zeros(3)
    next_tick, peak, max_ratio, max_tilt = 0., 0., 0., 0.
    min_gap, saturation_steps = float('inf'), 0
    started, stopped, stable_since = None, None, None
    interrupted, collision, fall, recovered = False, False, False, False
    start_xy, start_yaw = d.qpos[:2].copy(), spec.start_yaw
    prev_yaw, yaw_delta = spec.start_yaw, 0.
    rows, inputs, actions = [], [], []
    reason = 'timeout'
    def sample(phase):
        rows.append(dict(time=float(d.time), phase=phase, x=float(d.qpos[0]), y=float(d.qpos[1]),
                         z=float(d.qpos[2]), yaw_delta=yaw_delta, speed=float(np.linalg.norm(d.qvel[:3])),
                         angular_speed=float(np.linalg.norm(d.qvel[3:6])), peak_torque=peak))
    context = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
    with _release_torques(d), context as view:
        if view:
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 4., 110., -25.
        while d.time < timeout:
            wall = time.perf_counter()
            if view and not view.is_running():
                reason = 'viewer closed'; break
            arena.update(d)
            mujoco.mj_forward(m, d)
            rot = d.body('base').xmat.reshape(3, 3)
            yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
            yaw_delta += float(np.arctan2(np.sin(yaw-prev_yaw), np.cos(yaw-prev_yaw)))
            prev_yaw = yaw
            tilt = float(np.arccos(np.clip(rot[2, 2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
            observed = arena.sensed_obstacles(d)
            gap = clearance(d.qpos[:2], observed, spec.bounds)
            min_gap = min(min_gap, gap)
            collision = arena.collided(d)
            floor_feet = set()
            body_contact = False
            for contact in d.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if floor in pair:
                    floor_feet |= pair & feet
                    body_contact |= bool(pair & arena.robot_geoms) and not bool(pair & feet)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                reason = 'nonfinite state'; break
            fall = bool(d.qpos[2] < .15 or tilt > np.deg2rad(35) or body_contact)
            if collision or fall or gap < .035:
                reason = 'contact' if collision else ('posture guard' if fall else 'clearance guard'); break
            if d.time >= next_tick-dt/2:
                if started is None and d.time >= 2:
                    if not can_enter(robot, adapter, skill, observed, spec.bounds, duration):
                        reason = 'failed initiation'; break
                    started = float(d.time)
                    pid.kp, pid.kd = kp, kd
                    start_xy, start_yaw = d.qpos[:2].copy(), yaw
                    yaw_delta = 0.
                if started is not None:
                    elapsed = d.time-started
                    interrupted |= interrupt_after is not None and elapsed >= interrupt_after
                    if elapsed >= duration or interrupted:
                        stopped = float(d.time) if stopped is None else stopped
                        command[:] = 0
                    else:
                        command = np.asarray(COMMANDS[skill])*min(1., elapsed/.5)
                    try:
                        target = adapter.step(d, command)
                    except (ValueError, FloatingPointError) as error:
                        reason = 'policy failure: '+str(error); break
                    inputs.append(adapter.last_input.copy())
                    actions.append(adapter.last_action.copy())
                sample('stand' if started is None else ('recover' if stopped is not None else 'execute'))
                next_tick += .02
            if stopped is not None:
                stable = (np.linalg.norm(d.qvel[:3]) < .04 and np.linalg.norm(d.qvel[3:6]) < .2
                          and tilt < np.deg2rad(12) and len(floor_feet) == 4)
                stable_since = (float(d.time) if stable_since is None else stable_since) if stable else None
                if stable_since is not None and d.time-stable_since >= .8:
                    recovered, reason = True, 'interrupted' if interrupted else 'completed'; break
                if d.time-stopped >= 6:
                    reason = 'recovery timeout'; break
            d.ctrl[:] = pid.update(target, np.zeros(12), d.qpos[robot.qa], d.qvel[robot.va], dt)
            peak = max(peak, float(np.max(np.abs(d.ctrl))))
            ratio = float(np.max(np.abs(d.ctrl)/limits))
            max_ratio = max(max_ratio, ratio)
            saturation_steps += int(ratio >= .999)
            if frame_callback is not None:
                frame_callback(m, d, skill)
            mujoco.mj_step(m, d)
            if view:
                view.cam.lookat[:] = [*d.qpos[:2], .2]
                view.sync()
                time.sleep(max(0., dt-(time.perf_counter()-wall)))
    mujoco.mj_forward(m, d)
    sample('terminal')
    delta = d.qpos[:2]-start_xy if started is not None else np.zeros(2)
    cs, sn = np.cos(start_yaw), np.sin(start_yaw)
    local = np.array([[cs, sn], [-sn, cs]]) @ delta
    vx, vy, wz = COMMANDS[skill]
    progress = float(np.sign(wz)*yaw_delta if wz else np.sign(vx or vy)*local[0 if vx else 1])
    drift = float(np.linalg.norm(local) if wz else abs(local[1 if vx else 0]))
    motion_ok = progress >= (.4 if wz else (.25 if vx else .2)) and drift <= .3
    success = reason == 'completed' and motion_ok and recovered
    if reason == 'completed' and not motion_ok:
        reason = 'insufficient motion or excessive drift'
    summary = dict(policy=policy, skill=skill, success=bool(success), reason=reason,
                   recovered=recovered, interrupted=bool(interrupted), collision=collision, fall=fall,
                   elapsed=float(d.time), started=started, stopped=stopped, duration=duration,
                   signed_progress=progress, cross_axis_drift=drift, local_displacement=local.tolist(),
                   yaw_change=yaw_delta, final_speed=float(np.linalg.norm(d.qvel[:3])),
                   final_angular_speed=float(np.linalg.norm(d.qvel[3:6])),
                   max_tilt_deg=float(np.rad2deg(max_tilt)), peak_torque=peak,
                   max_torque_limit_ratio=max_ratio, saturation_fraction=saturation_steps/max(1, round(d.time/dt)),
                   minimum_clearance=min_gap, policy_ticks=adapter.ticks, physics_dt=dt, policy_dt=.02,
                   checkpoint_sha256=adapter.manifest['artifacts']['source_policy.pt'],
                   export_sha256=adapter.manifest['artifacts']['model.npz'],
                   scenario=spec.to_dict(), experimental=True, snn_eligible=False)
    if output is not None:
        folder = Path(output).resolve(); folder.mkdir(parents=True, exist_ok=True)
        (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        with (folder/'trajectory.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(folder/'inference.npz', inputs=np.asarray(inputs), applied_actions=np.asarray(actions))
    return rows, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('robot_lab', 'himloco'), default='himloco')
    parser.add_argument('--skill', choices=tuple(COMMANDS), default='forward')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--output', default='output/pretrained_velocity/latest')
    parser.add_argument('--allow-unvalidated', action='store_true', help='Explicitly run an unavailable experimental option')
    args = vars(parser.parse_args())
    allow_unvalidated = args.pop('allow_unvalidated')
    if not allow_unvalidated:
        from skills.registry import get_skill
        item = get_skill(f"pretrained/{args['policy']}/{args['skill']}")
        if not item['available']:
            parser.error('This option failed validation; use --allow-unvalidated only for an explicit experiment')
    _, result = execute_pretrained_skill(**args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['success'] else 1)
