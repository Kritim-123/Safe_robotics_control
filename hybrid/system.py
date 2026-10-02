"""A persistent MuJoCo episode with baseline -> skill -> baseline handoffs."""
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import platform
import time

import mujoco
import numpy as np

from controllers.navigation import NavigationExperiment, WaypointController
from controllers.walking import JointPID
from hybrid.arena import Arena, build_model
from hybrid.perception import choose_masked, observe_encounter, teacher_scores, threatening_obstacle
from hybrid.snn import DEFAULT_MODEL, SpikingSelector
from hybrid.sensing import ObstacleSensor
from skills.avoidance import clearance


@dataclass
class RunResult:
    summary: dict
    history: list
    events: list
    scenario: dict


class HybridSystem:
    """Classical skill execution; a rule or SNN selector chooses the option.

    Only this loop writes motor torques. No skill resets or steps the simulator.
    Safety masks/supervision are deterministic and independent of SNN scores.
    """

    def __init__(self, scenario, mode='snn', model_path=DEFAULT_MODEL, *, replanning=False, max_speed=0.18,
                 reaction='standard'):
        if mode not in ('baseline', 'rules', 'snn'):
            raise ValueError('mode must be baseline, rules, or snn')
        self.scenario, self.mode = scenario, mode
        self.replanning = bool(replanning)
        if reaction not in ('standard', 'fast'):
            raise ValueError('Reaction must be standard or fast')
        self.reaction = reaction
        self.sensing_dt = .02 if reaction == 'fast' else .1
        self.confirmation_dt = .08 if reaction == 'fast' else .2
        self.max_speed = WaypointController(max_speed=max_speed).max_speed
        self.robot = NavigationExperiment(build_model(scenario))
        self.arena = Arena(scenario, self.robot.model)
        self.selector = SpikingSelector(model_path) if mode == 'snn' else None

    def run(self, *, viewer=False, record_callback=None, playback_speed=1.0,
            event_callback=None, should_stop=None):
        if not np.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError('Playback speed must be finite and positive')
        spec, robot = self.scenario, self.robot
        m, d = robot.model, robot.data
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        d.qpos[:2] = spec.start
        d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0, 0, np.sin(spec.start_yaw/2)]
        self.arena.update(d)
        mujoco.mj_forward(m, d)
        pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
        tracker = WaypointController(max_speed=self.max_speed)
        sensor = ObstacleSensor(position_noise=spec.position_noise, velocity_noise=spec.velocity_noise,
                                delay=spec.sensor_delay, filter_tau=spec.sensor_filter_tau, seed=spec.sensing_seed)
        robot.commanded_velocity[:] = 0
        robot.activity = robot.gait_time = 0.0
        goal = np.asarray(spec.goal)
        base_id = m.body('base').id
        dt, control_dt, sensing_dt = float(m.opt.timestep), 0.01, self.sensing_dt
        next_control = next_sense = next_log = 0.0
        target, target_velocity = robot.home.copy(), np.zeros(m.nu)
        command = np.zeros(3)
        state, skill = 'BASELINE', None
        events, history, latencies = [], [], []
        decision_latencies = []
        recovery_until, recovery_deadline, clear_since, settled = 0.0, 0.0, None, 0.0
        intervention_count = returns = overrides = total_spikes = total_opportunities = 0
        stopping, collision = False, False
        minimum_clearance = clearance(d.qpos[:2], self.arena.sensed_obstacles(d), spec.bounds)
        max_torque, path_length = 0.0, 0.0
        previous_position = d.qpos[:2].copy()
        reason = 'timeout'
        selected_target = None
        selected_obstacle, replans = None, 0
        conflict_since, conflict_obstacle = None, None

        def transition(new_state, reason_text, **details):
            nonlocal state
            events.append(dict(time=float(d.time), source=state, target=new_state,
                               reason=reason_text, **details))
            if event_callback is not None:
                event_callback(events[-1])
            state = new_state
            # The trajectory and command slews remain continuous. Only stale
            # accumulated error is removed; physics and gait phase are untouched.
            pid.integral[:] = 0

        def activate_skill(obstacle_index, reason_text):
            nonlocal skill, selected_obstacle, reason, clear_since, replans
            nonlocal conflict_since, conflict_obstacle
            nonlocal intervention_count, overrides, total_spikes, total_opportunities
            if self.replanning and intervention_count >= 12:
                reason = 'intervention limit'
                transition('FAILED', reason)
                return False
            decision_started = time.perf_counter()
            encounter = observe_encounter(position, yaw, goal, observed, spec.bounds, obstacle_index, d.time)
            if self.selector is None:
                scores, diagnostics = teacher_scores(encounter.features), dict(spikes=0, opportunities=0)
            else:
                started = time.perf_counter()
                scores, diagnostics = self.selector.predict(encounter.features)
                latencies.append((time.perf_counter()-started)*1000)
            choice = choose_masked(scores, encounter.allowed)
            decision_ms = (time.perf_counter()-decision_started)*1000
            decision_latencies.append(decision_ms)
            if choice is None:
                reason = 'no applicable skill'
                transition('FAILED', reason)
                return False
            overrides += int(choice != int(np.argmax(scores)))
            total_spikes += diagnostics['spikes']
            total_opportunities += diagnostics['opportunities']
            interrupted = skill.name if state == 'SKILL' else None
            replans += int(interrupted is not None)
            skill = encounter.skills[choice]
            if self.replanning and skill.name == 'wait':
                skill.timeout = 60.0
            selected_obstacle = obstacle_index
            intervention_count += 1
            clear_since = None
            conflict_since, conflict_obstacle = None, None
            transition('SKILL', reason_text, skill=skill.name, interrupted_skill=interrupted,
                       obstacle_index=obstacle_index, scores=np.asarray(scores).tolist(),
                       decision_ms=decision_ms, inference_ms=latencies[-1] if self.selector else None,
                       selection_position=position.tolist(), goal=goal.tolist(),
                       candidate_paths=[[p.tolist() for p in s.points] for s in encounter.skills],
                       obstacle_name=spec.obstacles[obstacle_index].name,
                       obstacle_shape=spec.obstacles[obstacle_index].shape,
                       confirmation_dt=self.confirmation_dt,
                       allowed=encounter.allowed.tolist(), features=encounter.features.tolist(), **diagnostics)
            return True

        ctx = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
        try:
            with ctx as view:
                if view is not None:
                    x0, x1, y0, y1 = spec.bounds
                    view.cam.lookat[:] = [(x0+x1)/2, (y0+y1)/2, 0.25]
                    view.cam.distance, view.cam.azimuth, view.cam.elevation = max(7.0, 1.2*max(x1-x0, y1-y0)), 110, -60
                while d.time < spec.timeout:
                    if should_stop is not None and should_stop():
                        reason = 'user stopped'
                        break
                    wall_start = time.perf_counter()
                    if view is not None and not view.is_running():
                        reason = 'viewer closed'
                        break
                    self.arena.update(d)
                    # Refresh after kinematic obstacle movement before sensing.
                    mujoco.mj_forward(m, d)
                    position = d.body('base').xpos[:2].copy()
                    rot = d.body('base').xmat.reshape(3, 3)
                    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                    distance = float(np.linalg.norm(goal-position))
                    obstacles = self.arena.sensed_obstacles(d)
                    if d.time >= next_sense - dt/2:
                        next_sense += sensing_dt
                        observed = sensor.observe(float(d.time), obstacles)
                        gap = clearance(position, observed, spec.bounds)
                        minimum_clearance = min(minimum_clearance, clearance(position, obstacles, spec.bounds))
                        threat = threatening_obstacle(position, goal, observed)
                        if d.time >= 2 and self.mode != 'baseline':
                            if state == 'BASELINE' and threat is not None and not stopping:
                                if not activate_skill(threat, 'obstacle predicted on route'):
                                    break
                            elif state == 'SKILL':
                                if self.replanning and skill.name != 'wait' and d.time-skill.started >= 1.0:
                                    moving = [(i, obstacle) for i, obstacle in enumerate(observed)
                                              if np.linalg.norm(obstacle[2]) > .025]
                                    conflict = threatening_obstacle(position, skill.target(position),
                                                                     [o for _, o in moving])
                                    if conflict is not None:
                                        conflict_index = moving[conflict][0]
                                        if conflict_index != conflict_obstacle:
                                            conflict_since, conflict_obstacle = float(d.time), conflict_index
                                        if d.time-conflict_since >= self.confirmation_dt-1e-9:
                                            if not activate_skill(conflict_index, 'moving obstacle invalidated active route'):
                                                break
                                    else:
                                        conflict_since, conflict_obstacle = None, None
                                if skill.name == 'wait':
                                    wait_threat = (threatening_obstacle(position, goal, [observed[selected_obstacle]])
                                                   if self.replanning else threat)
                                    clear_since = (float(d.time) if clear_since is None else clear_since) if wait_threat is None else None
                                    completed = clear_since is not None and d.time-clear_since >= 0.7
                                else:
                                    skill.target(position)
                                    completed = skill.complete(position, d.time)
                                if completed:
                                    recovery_until = d.time + 0.8
                                    recovery_deadline = d.time + 5.0
                                    transition('RECOVER', 'skill completed', skill=skill.name)
                                elif skill.expired(d.time):
                                    # Do not silently call a timed-out skill a success.
                                    reason = 'skill timeout'
                                    transition('FAILED', reason, skill=skill.name)
                                    break
                            elif state == 'RECOVER' and d.time >= recovery_until:
                                stable = (robot.activity == 0 and np.linalg.norm(d.qvel[:2]) < .04
                                          and np.linalg.norm(d.qvel[3:6]) < .2
                                          and rot[2, 2] > np.cos(np.deg2rad(12)))
                                if stable:
                                    returns += 1
                                    transition('BASELINE', 'stable handoff to goal tracking',
                                               speed=float(np.linalg.norm(d.qvel[:2])),
                                               angular_speed=float(np.linalg.norm(d.qvel[3:6])))
                                    skill = None
                                elif d.time >= recovery_deadline:
                                    reason = 'recovery timeout'
                                    transition('FAILED', reason)
                                    break
                        command[:] = 0
                        selected_target = None
                        if d.time >= 2:
                            if state == 'BASELINE':
                                if distance < .06:
                                    stopping = True
                                elif distance > .12:
                                    stopping = False
                                if not stopping:
                                    selected_target = goal
                                    command = tracker.command(position, yaw, goal)
                            elif state == 'SKILL' and skill.name != 'wait':
                                selected_target = skill.target(position)
                                command = tracker.command(position, yaw, selected_target)
                        m.site_pos[m.site('skill_target').id] = ([*selected_target, .08]
                                                                 if selected_target is not None else [0, 0, -1])
                        # Conservative emergency stop. It is not a formal safety
                        # guarantee: moving obstacles can still strike a stopped robot.
                        if self.mode != 'baseline' and gap < 0.035:
                            reason = 'clearance stop'
                            transition('FAILED', reason, observed_clearance=gap,
                                       true_clearance=clearance(position, obstacles, spec.bounds))
                            break
                    if d.time >= next_control - dt/2:
                        previous_target = target.copy()
                        target = robot.velocity_targets(command, control_dt)
                        target_velocity = (target-previous_target)/control_dt
                        next_control += control_dt
                    torque = pid.update(target, target_velocity, d.qpos[robot.qa], d.qvel[robot.va], dt)
                    d.ctrl[:] = torque
                    max_torque = max(max_torque, float(np.max(np.abs(torque))))
                    d.xfrc_applied[base_id, :] = 0
                    if spec.push_time <= d.time < spec.push_time + spec.push_duration:
                        d.xfrc_applied[base_id, :2] = spec.push_force
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    position = d.body('base').xpos[:2].copy()
                    path_length += float(np.linalg.norm(position-previous_position))
                    previous_position = position
                    distance = float(np.linalg.norm(goal-position))
                    speed = float(np.linalg.norm(d.qvel[:2]))
                    upright = float(d.body('base').xmat.reshape(3, 3)[2, 2])
                    if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                        reason = 'nonfinite state'
                        break
                    if self.arena.collided(d):
                        reason, collision = 'obstacle contact', True
                        break
                    if d.qpos[2] < .15 or upright < np.cos(np.deg2rad(35)):
                        reason = 'height/tilt limit'
                        break
                    arrived = (state == 'BASELINE' and distance <= .12 and speed < .04
                               and robot.activity == 0 and d.time > 2)
                    settled = settled + dt if arrived else 0.0
                    if d.time >= next_log - dt/2:
                        history.append(dict(time=float(d.time), x=float(position[0]), y=float(position[1]),
                                            yaw=yaw, height=float(d.qpos[2]), upright=upright,
                                            distance=distance, speed=speed, state=state,
                                            skill=skill.name if skill is not None else '',
                                            clearance=clearance(position, obstacles, spec.bounds),
                                            observed_clearance=gap,
                                            torque=float(np.max(np.abs(torque))),
                                            command_vx=float(robot.commanded_velocity[0]),
                                            command_vy=float(robot.commanded_velocity[1]),
                                            command_yaw_rate=float(robot.commanded_velocity[2]),
                                            selected_target=selected_target.tolist() if selected_target is not None else None,
                                            obstacle_positions=[c.tolist() for c, _, _ in obstacles]))
                        next_log += 0.1
                        if record_callback is not None:
                            record_callback(m, d, state, history[-1])
                    if settled >= 1.0:
                        reason = 'goal reached'
                        transition('COMPLETE', reason)
                        break
                    if view is not None:
                        view.sync()
                        time.sleep(max(0, dt/playback_speed-(time.perf_counter()-wall_start)))
        except KeyboardInterrupt:
            reason = 'interrupted'
        finally:
            d.ctrl[:] = 0
            d.xfrc_applied[:] = 0
        if state not in ('COMPLETE', 'FAILED'):
            transition('FAILED', reason)
        summary = dict(mode=self.mode, scenario=spec.name, max_command_speed=self.max_speed,
                       playback_speed=playback_speed, success=reason == 'goal reached',
                       reason=reason, elapsed=float(d.time), final_position=d.qpos[:2].tolist(),
                       final_distance=float(np.linalg.norm(goal-d.qpos[:2])), collision=collision,
                       fall=reason == 'height/tilt limit', path_length=path_length,
                       min_footprint_clearance=minimum_clearance, max_torque=max_torque,
                       interventions=intervention_count, returns_to_baseline=returns,
                       replanning_enabled=self.replanning, replans=replans,
                       safety_mask_overrides=overrides, snn_spikes=total_spikes,
                       snn_spike_fraction=total_spikes/max(total_opportunities, 1),
                       inference_mean_ms=float(np.mean(latencies)) if latencies else None,
                       inference_max_ms=max(latencies) if latencies else None,
                       decision_mean_ms=float(np.mean(decision_latencies)) if decision_latencies else None,
                       decision_max_ms=max(decision_latencies) if decision_latencies else None,
                       reaction_profile=self.reaction, confirmation_dt=self.confirmation_dt,
                       model_sha256=(hashlib.sha256(self.selector.path.read_bytes()).hexdigest()
                                     if self.selector is not None else None),
                       python_version=platform.python_version(), mujoco_version=mujoco.__version__,
                       numpy_version=np.__version__, physics_dt=dt, control_dt=control_dt,
                       sensing_dt=sensing_dt)
        return RunResult(summary, history, events, spec.to_dict())
