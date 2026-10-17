"""Watch live MuJoCo observations and the frozen SNN's actual spike counts.

python -m experiments.spike_pattern_skills.live
R restarts the scene. J toggles the explicit jump-request input, not a jump.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.walking import WalkingExperiment, JointPID
from hybrid.arena import Arena, build_model
from skills.avoidance import clearance
from skills.dynamic import _release_torques
from .encoder import RandomLIFEncoder
from .observations import observation_from_simulation
from .record import ROOT
from .situations import situations


def run(case='crossing_right_to_left-03', duration=None, headless=False):
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError('Duration must be positive and finite')
    if headless and duration is None:
        raise ValueError('Headless checks require a finite duration')
    cases = {item.scenario.name: item for item in situations()}
    if case not in cases:
        raise ValueError('Unknown study case: '+case)
    chosen = cases[case]
    spec = chosen.scenario
    encoder = RandomLIFEncoder.load(ROOT/'docs/agents/spike_pattern_100/model.npz')
    robot = WalkingExperiment(build_model(spec))
    m, d = robot.model, robot.data
    arena = Arena(spec, m)
    pid = JointPID(robot.kp, robot.ki, robot.kd, robot.limits)
    pending = deque()
    def on_key(key):
        if key in (82, 74):  # GLFW R/J; process changes on the physics thread.
            pending.append(key)
    def reset():
        mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
        d.qpos[:2] = spec.start
        d.qpos[3:7] = [np.cos(spec.start_yaw/2), 0., 0., np.sin(spec.start_yaw/2)]
        pid.integral[:] = 0
        arena.update(d)
        mujoco.mj_forward(m, d)
    reset()
    requested = chosen.jump_requested
    view = None
    peak, updates, resets, next_observation = 0., 0, 0, 0.
    guard = None
    counts = [0]*16
    active = []
    figure = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(figure)
    figure.title, figure.xlabel = 'Spikes per 16-step window', 'Output neuron ID'
    figure.range[:] = [[-.5, 15.5], [0, 16.5]]
    figure.flg_extend = 0
    dt = float(m.opt.timestep)
    began = time.perf_counter()
    try:
        if not headless:
            view = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
            view.cam.lookat[:] = [*spec.start, .15]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 4.5, np.rad2deg(spec.start_yaw)+100, -35
        with _release_torques(d):
            while (view is None or view.is_running()) and (duration is None or d.time < duration):
                wall = time.perf_counter()
                while pending:
                    key = pending.popleft()
                    if key == 82:
                        reset(); next_observation = 0.; guard = None; resets += 1
                    elif key == 74:
                        requested = not requested
                        next_observation = float(d.time)
                if guard is None:
                    arena.update(d)
                    mujoco.mj_forward(m, d)
                    rotation = d.body('base').xmat.reshape(3, 3)
                    if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()):
                        guard = 'nonfinite state'
                    elif d.qpos[2] < .15 or rotation[2, 2] < np.cos(np.deg2rad(35)):
                        guard = 'posture guard'
                    elif arena.collided(d):
                        guard = 'obstacle contact'
                    elif clearance(d.qpos[:2], arena.sensed_obstacles(d), spec.bounds) < .035:
                        guard = 'clearance guard'
                    if guard is not None:
                        d.ctrl[:] = 0
                    elif d.time >= next_observation-dt/2:
                        observation = dict(time=float(d.time), position=d.qpos[:2].tolist(),
                            yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                            speed=float(np.linalg.norm(d.qvel[:2])), height=float(d.qpos[2]))
                        item = observation_from_simulation(spec, observation, requested,
                            observation_id=f'{case}-{d.time:.3f}', context_id=case)
                        firing = encoder.encode(item['features'])
                        counts, active = firing['spike_counts'], firing['active_neuron_ids']
                        updates += 1
                        next_observation += .1
                        for neuron, count in enumerate(counts):
                            figure.linepnt[neuron] = 4
                            figure.linedata[neuron, :8] = [neuron-.3, 0, neuron-.3, count,
                                                           neuron+.3, count, neuron+.3, 0]
                            figure.linergb[neuron] = [.15, .8, .85]
                    if guard is None:
                        d.ctrl[:] = pid.update(robot.home, np.zeros(12), d.qpos[robot.qa], d.qvel[robot.va], dt)
                        peak = max(peak, float(np.max(np.abs(d.ctrl))))
                        mujoco.mj_step(m, d)
                if view is not None:
                    status = 'PHYSICS STOPPED: '+guard if guard else 'Standing; no pattern-to-skill decoder trained'
                    text = (f'Frozen SNN | {case} | t={d.time:.1f}s\n{status}\n'
                            f'Active neuron IDs: {active}\n'
                            f'IDs  0-7: {counts[:8]}\nIDs 8-15: {counts[8:]}\n'
                            f'Explicit jump request: {requested} (input only)\n'
                            'R: restart scene | J: toggle request | close window to stop')
                    view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                    mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ''))
                    view.set_figures((mujoco.MjrRect(10, 10, 440, 190), figure))
                    view.sync()
                    time.sleep(max(0., dt-(time.perf_counter()-wall)))
                elif guard is not None:
                    break
    finally:
        if view is not None:
            view.close()
    result = dict(case=case, simulated_time=float(d.time), peak_torque=peak,
                  neural_updates=updates, guard=guard, active_neuron_ids=active, spike_counts=counts,
                  wall_seconds=time.perf_counter()-began, manual_scene_resets=resets,
                  encoder_checksum=encoder.checksum, skills_selected=0)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='crossing_right_to_left-03')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--headless', action='store_true')
    run(**vars(parser.parse_args()))
