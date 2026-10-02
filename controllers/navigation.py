"""Classical waypoint tracking for the project's torque-controlled MuJoCo Go2."""
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np

from controllers.walking import JointPID, WalkingExperiment

SCENE = Path(__file__).resolve().parents[1] / 'unitree_go2' / 'scene_indoor.xml'


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class WaypointController:
    """Body-frame position feedback and heading regulation; no path planner."""

    def __init__(self, max_speed=0.18, max_yaw_rate=0.5):
        if not np.all(np.isfinite([max_speed, max_yaw_rate])) or min(max_speed, max_yaw_rate) <= 0:
            raise ValueError('Speed limits must be finite and positive')
        self.max_speed, self.max_yaw_rate = max_speed, max_yaw_rate

    def command(self, position, yaw, goal):
        error = np.asarray(goal) - np.asarray(position)
        distance = np.linalg.norm(error)
        heading = np.arctan2(error[1], error[0]) if distance > 0.1 else yaw
        heading_error = wrap_angle(heading - yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        local_error = np.array([[c, s], [-s, c]]) @ error
        # This simple gait loses net progress at very small stride lengths.
        # Keep a minimum moving stride; the supervisor separately stops at goal.
        speed = min(self.max_speed, max(0.12, 0.8 * distance))
        velocity = speed * local_error / max(distance, 1e-12)
        # Slow translation while turning, retaining a small lateral correction.
        velocity *= max(0.15, np.cos(heading_error))
        return np.r_[velocity, np.clip(1.5 * heading_error, -self.max_yaw_rate, self.max_yaw_rate)]


class NavigationExperiment(WalkingExperiment):
    """Persistent gait/physics state across waypoints, with one torque writer."""

    def __init__(self, path=SCENE):
        super().__init__(path)
        self.commanded_velocity = np.zeros(3)
        self.gait_time = 0.0
        self.activity = 0.0

    def velocity_targets(self, command, dt):
        # Slew commands to avoid abrupt foot target changes at waypoint switches.
        rate = np.array([0.25, 0.25, 0.8])
        self.commanded_velocity += np.clip(command - self.commanded_velocity, -rate * dt, rate * dt)
        moving = np.linalg.norm(self.commanded_velocity) > 0.005
        self.activity = np.clip(self.activity + (dt / 0.6 if moving else -dt / 0.6), 0, 1)
        self.gait_time += dt
        result = self.home.copy()
        origin = self.ik.body('base').xpos[:2].copy()
        for leg, info in self.legs.items():
            phase = (self.gait_time / self.period + self.phase[leg]) % 1
            if phase < self.stance:
                fraction = phase / self.stance
                travel, z = 0.5 - fraction, 0.0
            else:
                fraction = (phase - self.stance) / (1 - self.stance)
                travel = 3 * fraction**2 - 2 * fraction**3 - 0.5
                z = self.lift * np.sin(np.pi * fraction)**2
            radius = info['foot'][:2] - origin
            vx, vy, wz = self.commanded_velocity
            foot_velocity = np.array([vx - wz * radius[1], vy + wz * radius[0]])
            displacement = foot_velocity * self.period * self.stance * travel
            offset = self.activity * np.r_[displacement, z]
            result[info['acts']] = self.solve_leg(leg, info['foot'] + offset)
        return result

    def navigate(self, waypoints, *, start=(0.0, 0.0), start_yaw=0.0,
                 timeout=120.0, tolerance=0.12, viewer=False):
        goals = np.asarray(waypoints, dtype=float)
        start = np.asarray(start, dtype=float)
        if (goals.ndim != 2 or goals.shape[1] != 2 or len(goals) == 0
                or start.shape != (2,) or not np.all(np.isfinite(goals))
                or not np.all(np.isfinite(start))
                or not np.all(np.isfinite([start_yaw, timeout, tolerance]))
                or timeout <= 0 or tolerance <= 0):
            raise ValueError('Supply finite (x,y) waypoints/start and positive timeout/tolerance')
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home_id)
        d.qpos[:2] = start
        d.qpos[3:7] = [np.cos(start_yaw / 2), 0, 0, np.sin(start_yaw / 2)]
        mujoco.mj_forward(m, d)
        self.commanded_velocity[:] = 0
        self.gait_time = self.activity = 0.0
        pid = JointPID(self.kp, self.ki, self.kd, self.limits)
        navigator = WaypointController()
        dt = float(m.opt.timestep)
        target_dt = max(dt, round(0.01 / dt) * dt)
        next_target = 0.0
        target = self.home.copy()
        target_velocity = np.zeros(m.nu)
        index, settled = 0, 0.0
        stopping = False
        history = []
        reason = 'timeout'
        # Fixed indoor obstacles only. Floor and robot self-contacts are separate.
        layout = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'indoor_layout')
        obstacles = set(np.flatnonzero(m.geom_bodyid == layout)) if layout >= 0 else set()
        ctx = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
        with ctx as view:
            if view is not None:
                view.cam.distance, view.cam.elevation, view.cam.azimuth = 3.0, -35, 135
            while d.time < timeout:
                wall = time.perf_counter()
                if view is not None and not view.is_running():
                    reason = 'viewer closed'
                    break
                base = d.body('base')
                rotation = base.xmat.reshape(3, 3)
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                distance = float(np.linalg.norm(goals[index] - base.xpos[:2]))
                if d.time >= next_target - dt / 2:
                    command = np.zeros(3)
                    if d.time >= 2.0:
                        if distance <= tolerance and index < len(goals) - 1:
                            index += 1
                            distance = float(np.linalg.norm(goals[index] - base.xpos[:2]))
                        if index == len(goals) - 1:
                            if distance < tolerance * 0.5:
                                stopping = True
                            elif distance > tolerance:
                                stopping = False
                        if not stopping:
                            command = navigator.command(base.xpos[:2], yaw, goals[index])
                    previous = target.copy()
                    target = self.velocity_targets(command, target_dt)
                    target_velocity = (target - previous) / target_dt
                    next_target += target_dt
                d.ctrl[:] = pid.update(target, target_velocity, d.qpos[self.qa], d.qvel[self.va], dt)
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                base = d.body('base')
                distance = float(np.linalg.norm(goals[index] - base.xpos[:2]))
                upright = float(base.xmat.reshape(3, 3)[2, 2])
                if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                    reason = 'nonfinite state'
                    break
                if base.xpos[2] < 0.15 or upright < np.cos(np.deg2rad(35)):
                    reason = 'height/tilt limit'
                    break
                if any(c.geom1 in obstacles or c.geom2 in obstacles for c in d.contact):
                    reason = 'obstacle contact'
                    break
                speed = float(np.linalg.norm(d.qvel[:2]))
                arrived = (d.time > 2 and index == len(goals) - 1 and distance <= tolerance
                           and speed < 0.04 and self.activity == 0)
                settled = settled + dt if arrived else 0.0
                if not history or d.time - history[-1]['time'] >= 0.05 - dt / 2:
                    history.append(dict(time=float(d.time), x=float(base.xpos[0]),
                                        y=float(base.xpos[1]), yaw=float(yaw),
                                        distance=distance, waypoint=index, speed=speed,
                                        height=float(base.xpos[2]), upright=upright))
                if settled >= 1.0:
                    reason = 'goal reached'
                    break
                if view is not None:
                    view.cam.lookat[:] = base.xpos
                    view.sync()
                    time.sleep(max(0, dt - (time.perf_counter() - wall)))
        # Simulation is stopped here; this is not a hardware stopping command.
        d.ctrl[:] = 0
        return history, dict(success=reason == 'goal reached', reason=reason,
                            elapsed=float(d.time), position=d.qpos[:2].tolist(),
                            goal=goals[-1].tolist(),
                            final_distance=float(np.linalg.norm(goals[-1] - d.qpos[:2])),
                            waypoint_index=index)
