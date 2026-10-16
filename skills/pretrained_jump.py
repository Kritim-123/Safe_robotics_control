"""Experimental pretrained gap-jump adapter; not enabled in the SNN selector.

Run: python -m skills.pretrained_jump --width .3 --speed 1 --viewer
Uses the project's robot, actual terrain rays, and capped joint torque PD.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import torch

from controllers.walking import WalkingExperiment, JointPID
from skills.dynamic import _release_torques
from third_party.go2_atomic_skills.skills import WalkSkill, JumpSkill
from third_party.go2_atomic_skills.obs import DEFAULT_JOINT_POS, MJLAB_JOINT_ORDER
from third_party.go2_atomic_skills.mujoco_helper import obs_from_mujoco, raycast_height_scan

ROOT = Path(__file__).resolve().parents[1]


def gap_model(width=.3, friction=.8):
    if not np.isfinite(width) or not 0 <= width <= .4 or not np.isfinite(friction) or friction <= 0:
        raise ValueError('Expected gap width 0..0.4 m and positive friction')
    root = ET.parse(ROOT/'unitree_go2/go2_mjx.xml').getroot()
    root.find('compiler').set('meshdir', str(ROOT/'unitree_go2/assets'))
    # All robot geoms are excluded from the terrain-only scan group.
    for geom in root.iter('geom'):
        geom.set('group', '0')
        if geom.get('friction') is not None:
            geom.set('friction', f'{friction} .005 .0001')
    world = root.find('worldbody')
    for name, x, half, z in [('approach', -1.3, 2.7, -.1),
                              ('landing', 4.4+width, 3., -.1),
                              ('pit', 1.4+width/2, 10., -1.1)]:
        ET.SubElement(world, 'geom', name=name, type='box', pos=f'{x} 0 {z}',
                      size=f'{half} 3 .1', group='3', friction=f'{friction} .005 .0001',
                      rgba='.35 .45 .5 1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class PretrainedGapJump:
    """50 Hz joint-target interface with shared action history across handoff."""
    def __init__(self, model):
        actual = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        if actual != list(MJLAB_JOINT_ORDER):
            raise ValueError('Pretrained policy requires FL/FR/RL/RR hip/thigh/calf actuator order')
        self.walk = WalkSkill()
        self.jump = JumpSkill()
        self.reset()

    def reset(self):
        self.walk.reset()
        self.jump.reset()
        self.last = np.zeros(12, dtype=np.float32)

    def targets(self, model, data, command, jumping):
        inp = obs_from_mujoco(model, data, base_body='base')
        scan = raycast_height_scan(model, data, base_body='base', geomgroup=[0, 0, 0, 1, 0, 0])
        self.walk.last_action = self.last.copy()
        self.jump.last_action = self.last.copy()
        # Both clocks/history advance once per control tick, using the action
        # actually applied on the preceding tick (including at the handoff).
        walk_target = self.walk.act(inp, command)
        jump_target = self.jump.act(inp, command, scan)
        selected = self.jump if jumping else self.walk
        self.last = selected.last_action.copy()
        target = jump_target if jumping else walk_target
        if not np.all(np.isfinite(target)):
            raise ValueError('Nonfinite policy output')
        return target, scan


def evaluate(width=.3, speed=1., friction=.8, viewer=False, output=None, walk_only=False):
    if not np.isfinite(speed) or not 0 < speed <= 2.5:
        raise ValueError('Expected forward command 0..2.5 m/s')
    torch.set_num_threads(1)
    robot = WalkingExperiment(gap_model(width, friction))
    m, d = robot.model, robot.data
    policy = PretrainedGapJump(m)
    mujoco.mj_resetDataKeyframe(m, d, robot.home_id)
    d.qpos[robot.qa] = DEFAULT_JOINT_POS
    d.qpos[2] = .33
    # Match policy training armature and PD, but retain project torque ceiling.
    m.dof_armature[robot.va] = np.tile([.01, .01, .02], 4)
    limits = np.minimum(robot.limits, np.tile([23.5, 23.5, 24.], 4))
    pid = JointPID(np.tile([20., 20., 40.], 4), np.zeros(12),
                   np.tile([1., 1., 2.], 4), limits)
    mujoco.mj_forward(m, d)
    target = DEFAULT_JOINT_POS.copy()
    feet = {v['geom'] for v in robot.legs.values()}
    terrain = {m.geom(n).id for n in ('approach', 'landing', 'pit')}
    dt = float(m.opt.timestep)
    next_tick, peak, air, max_air = 0., 0., 0., 0.
    scan = np.full(187, .33)
    samples = []
    jumped, crossed, landed_at = False, False, None
    rows, reason = [], 'timeout'
    ctx = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
    with _release_torques(d), ctx as view:
        if view:
            view.cam.lookat[:] = [1.5, 0, .1]
            view.cam.distance, view.cam.azimuth, view.cam.elevation = 5., 100., -25.
        while d.time < 12:
            wall = time.perf_counter()
            if view and not view.is_running():
                reason = 'viewer closed'
                break
            if d.time >= next_tick-dt/2:
                if d.time >= 1:
                    jumped = jumped or (not walk_only and d.qpos[0] >= 1.)
                    crossed = crossed or d.qpos[0] > 1.4+width+.45
                    command = (0. if crossed else speed*min((d.time-1)/1., 1.), 0., 0.)
                    target, scan = policy.targets(m, d, command, jumped and not crossed)
                    target = np.clip(target, robot.joint_limits[:, 0], robot.joint_limits[:, 1])
                    # Reflect any joint-limit clipping in the next observation.
                    policy.last = ((target-DEFAULT_JOINT_POS)/.25).astype(np.float32)
                rows.append(dict(time=float(d.time), skill='jump' if jumped and not crossed else 'walk',
                                 x=float(d.qpos[0]), y=float(d.qpos[1]), z=float(d.qpos[2]),
                                 vx=float(d.qvel[0]), max_torque=peak))
                samples.append(np.concatenate((d.qpos.copy(), d.qvel.copy(), target.copy(), scan.copy())))
                next_tick += .02
            d.ctrl[:] = pid.update(target, np.zeros(12), d.qpos[robot.qa], d.qvel[robot.va], dt)
            peak = max(peak, float(np.max(np.abs(d.ctrl))))
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            contacts = set()
            body_contact = False
            for c in d.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if pair & terrain:
                    contacts |= pair & feet
                    body_contact |= not bool(pair & feet)
            air = air+dt if not contacts and jumped and not crossed else 0.
            max_air = max(max_air, air)
            upright = d.body('base').xmat.reshape(3, 3)[2, 2]
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                reason = 'nonfinite state'; break
            if body_contact or d.qpos[2] < .15 or upright < np.cos(np.deg2rad(35)):
                reason = 'body contact or posture guard'; break
            if abs(d.qpos[1]) > 2.5 or d.qpos[0] < -3.5 or d.qpos[0] > 6.5:
                reason = 'bounds guard'; break
            stable = (crossed and len(contacts) == 4 and np.linalg.norm(d.qvel[:3]) < .1
                      and np.linalg.norm(d.qvel[3:6]) < .2 and upright > np.cos(np.deg2rad(12)))
            landed_at = (float(d.time) if landed_at is None else landed_at) if stable else None
            if landed_at is not None and d.time-landed_at >= .8:
                reason = 'stable crossing'; break
            if view:
                view.sync()
                time.sleep(max(0., dt-(time.perf_counter()-wall)))
    summary = dict(width=width, speed=speed, friction=friction, reason=reason,
                   crossed=bool(crossed), jump_activated=bool(jumped), longest_airtime=max_air,
                   success=reason == 'stable crossing' and max_air >= .04,
                   traversal_success=reason == 'stable crossing',
                   final_x=float(d.qpos[0]), final_z=float(d.qpos[2]), peak_torque=peak,
                   time=float(d.time), checkpoint='hando_w30', experimental=True, walk_only=walk_only)
    if output:
        path = Path(output).resolve(); path.mkdir(parents=True, exist_ok=True)
        rows.append(dict(time=float(d.time), skill='terminal', x=float(d.qpos[0]),
                         y=float(d.qpos[1]), z=float(d.qpos[2]), vx=float(d.qvel[0]), max_torque=peak))
        (path/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        with (path/'trace.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(path/'observations.npz', samples=np.asarray(samples),
                            columns=np.array(['qpos[19]', 'qvel[18]', 'joint_target[12]', 'height_scan[187]']),
                            terminal_qpos=d.qpos.copy(), terminal_qvel=d.qvel.copy())
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=float, default=.3)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--friction', type=float, default=.8)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--walk-only', action='store_true', help='Disable jump handoff for a walking control')
    parser.add_argument('--output', default='output/pretrained_jump/latest')
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
