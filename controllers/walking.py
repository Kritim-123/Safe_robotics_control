"""Simulation-only joint torque PID and foot trajectory experiment."""
from pathlib import Path
import time
from contextlib import nullcontext
import numpy as np
import mujoco
import mujoco.viewer

class JointPID:
    def __init__(self, kp, ki, kd, limits, integral_limit=3.0):
        self.kp, self.ki, self.kd = map(np.asarray, (kp, ki, kd))
        self.limits = np.asarray(limits)
        if (not all(np.all(np.isfinite(x)) and np.all(x >= 0)
                    for x in (self.kp, self.ki, self.kd))
            or not np.all(np.isfinite(self.limits)) or np.any(self.limits <= 0)
            or not np.isfinite(integral_limit) or integral_limit <= 0):
            raise ValueError('PID gains must be nonnegative and limits positive/finite')
        self.integral_limit = integral_limit  # Nm, limits the I contribution
        self.integral = np.zeros(len(limits))

    def update(self, target, target_velocity, q, velocity, dt):
        if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite([target, target_velocity, q, velocity])):
            raise ValueError('Invalid PID input')
        error = target - q
        increment = self.ki * error * dt
        candidate = np.clip(self.integral + increment,
                            -self.integral_limit, self.integral_limit)
        pd = self.kp * error + self.kd * (target_velocity - velocity)
        raw = pd + candidate
        blocked = ((raw > self.limits) & (increment > 0)) | ((raw < -self.limits) & (increment < 0))
        self.integral = np.where(blocked, self.integral, candidate)
        return np.clip(pd + self.integral, -self.limits, self.limits)

class WalkingExperiment:
    def __init__(self, path, kp=90.0, ki=3.0, kd=3.0, period=0.6,
                 stride=0.08, lift=0.06, stance=0.6):
        if (not np.all(np.isfinite([period, stride, lift, stance])) or period <= 0
            or stride < 0 or lift < 0 or not 0 < stance < 1):
            raise ValueError('Invalid gait parameters')
        self.model = m = (path if isinstance(path, mujoco.MjModel)
                          else mujoco.MjModel.from_xml_path(str(path)))
        self.home_id = m.key('home').id
        self.joints = m.actuator_trnid[:, 0].copy()
        if (m.nu != 12 or np.any(m.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
            or not np.allclose(m.actuator_gear[:, 0], 1)
            or not np.allclose(m.actuator_gear[:, 1:], 0)
            or np.any(m.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError('Expected twelve unit-gear, stateless joint actuators')
        self.qa = m.jnt_qposadr[self.joints]
        self.va = m.jnt_dofadr[self.joints]
        self.joint_limits = m.jnt_range[self.joints].copy()
        self.limits = np.minimum(24.0, np.min(np.abs(m.actuator_forcerange), axis=1))
        if np.any(self.limits <= 0):
            raise ValueError('Expected finite positive model force limits')
        # Motor-equivalent force law: tau = ctrl. Remove the built-in servo.
        m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_gainprm[:] = 0
        m.actuator_gainprm[:, 0] = 1
        m.actuator_biasprm[:] = 0
        m.actuator_ctrllimited[:] = True
        m.actuator_forcelimited[:] = True
        m.actuator_ctrlrange[:] = np.column_stack((-self.limits, self.limits))
        m.actuator_forcerange[:] = m.actuator_ctrlrange
        m.key_ctrl[:] = 0  # Home keyframe angles must never become motor torques.
        # CPU contact solve: the supplied MJX model uses only one iteration.
        m.opt.iterations = 50
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(m)
        self.ik = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.ik, self.home_id)
        mujoco.mj_forward(m, self.ik)
        self.home = self.ik.qpos[self.qa].copy()
        self.legs = {}
        for leg in ('FL', 'FR', 'RL', 'RR'):
            acts = np.array([m.actuator(f'{leg}_{j}').id for j in ('hip','thigh','calf')])
            site = m.site(f'{leg}_foot').id
            self.legs[leg] = dict(acts=acts, site=site,
                                  foot=self.ik.site_xpos[site].copy(),
                                  geom=m.geom(leg).id)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.period, self.stride, self.lift, self.stance = period, stride, lift, stance
        self.phase = dict(FL=0.0, RR=0.0, FR=0.5, RL=0.5)

    def solve_leg(self, leg, target):
        m, d = self.model, self.ik
        info = self.legs[leg]
        ids = info['acts']
        jac = np.zeros((3, m.nv))
        for _ in range(20):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            error = target - d.site_xpos[info['site']]
            if np.linalg.norm(error) < 0.0003:
                break
            mujoco.mj_jacSite(m, d, jac, None, info['site'])
            j = jac[:, self.va[ids]]
            change = j.T @ np.linalg.solve(j @ j.T + 1e-5*np.eye(3), error)
            d.qpos[self.qa[ids]] = np.clip(d.qpos[self.qa[ids]] + np.clip(change, -.15, .15),
                self.joint_limits[ids, 0] + .01, self.joint_limits[ids, 1] - .01)
        mujoco.mj_kinematics(m, d)
        if np.linalg.norm(target-d.site_xpos[info['site']]) > .003:
            raise RuntimeError(f'{leg}: unreachable foot target')
        return d.qpos[self.qa[ids]].copy()

    def targets(self, t, mode):
        if mode == 'stand' or t <= 0:
            return self.home.copy()
        result = self.home.copy()
        blend = min(t / 1.5, 1.0)
        blend = blend*blend*(3-2*blend)
        for leg, info in self.legs.items():
            if mode == 'step':
                # Shift weight rearward/right before unloading front-left.
                # Feet move +X/+Y relative to the body; the body shifts -X/-Y.
                shift = np.clip(t, 0, 1) if t < 3 else np.clip(4-t, 0, 1)
                shift = shift*shift*(3-2*shift)
                s = np.clip((t-1) / 2.0, 0, 1)
                offset = np.array([.06*shift, .06*shift,
                    self.lift*np.sin(np.pi*s)**2 if leg == 'FL' else 0])
            else:
                phase = (t / self.period + self.phase[leg]) % 1
                if phase < self.stance:
                    s = phase / self.stance
                    x, z = self.stride*(.5-s), 0
                else:
                    s = (phase-self.stance)/(1-self.stance)
                    x = self.stride*(3*s*s-2*s*s*s-.5)
                    z = self.lift*np.sin(np.pi*s)**2
                offset = blend*np.array([x, 0, z])
            result[info['acts']] = self.solve_leg(leg, info['foot']+offset)
        return result

    def run(self, mode='walk', duration=8.0, viewer=False):
        if mode not in ('stand','step','walk') or duration <= 0:
            raise ValueError('Choose stand, step, or walk and a positive duration')
        m, d = self.model, self.data
        for state in (d, self.ik):
            mujoco.mj_resetDataKeyframe(m, state, self.home_id)
            state.ctrl[:] = 0
            mujoco.mj_forward(m, state)
        pid = JointPID(self.kp, self.ki, self.kd, self.limits)
        dt = float(m.opt.timestep)
        settle = 2.0
        target_dt = max(dt, round(.01/dt)*dt)
        target, previous = self.home.copy(), self.home.copy()
        desired_velocity = np.zeros(m.nu)
        next_target = 0.0
        start = None
        history = []
        reason = 'duration reached'
        ctx = mujoco.viewer.launch_passive(m, d) if viewer else nullcontext(None)
        with ctx as view:
            if view is not None:
                view.cam.distance = 1.5
                view.cam.azimuth = 135
                view.cam.elevation = -20
            try:
                while d.time < settle + duration - dt/2:
                    wall = time.perf_counter()
                    if view is not None and not view.is_running():
                        reason = 'viewer closed'
                        break
                    if start is None and d.time >= settle - dt/2:
                        start = d.body('base').xpos.copy()
                    if d.time >= next_target - dt/2:
                        previous = target.copy()
                        target = self.targets(max(0, float(d.time)-settle), mode)
                        desired_velocity = (target-previous)/target_dt
                        next_target += target_dt
                    torque = pid.update(target, desired_velocity, d.qpos[self.qa], d.qvel[self.va], dt)
                    d.ctrl[:] = torque
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    base = d.body('base')
                    upright = float(base.xmat.reshape(3,3)[2,2])
                    if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                        reason = 'nonfinite state'
                        break
                    if base.xpos[2] < .15 or upright < np.cos(np.deg2rad(35)):
                        reason = 'height/tilt limit'
                        break
                    if len(history) == 0 or d.time-history[-1]['time'] >= .02-dt/2:
                        contacts = {leg: False for leg in self.legs}
                        for contact in d.contact:
                            for leg, info in self.legs.items():
                                if info['geom'] in (contact.geom1, contact.geom2):
                                    contacts[leg] = True
                        row = dict(time=float(d.time), x=float(base.xpos[0]), y=float(base.xpos[1]),
                            height=float(base.xpos[2]), upright=upright,
                            speed=float(d.sensor('global_linvel').data[0]),
                            joint_rms=float(np.sqrt(np.mean((target-d.qpos[self.qa])**2))),
                            peak_torque=float(np.max(np.abs(torque))),
                            saturated=float(np.mean(np.abs(torque) >= self.limits-1e-8)))
                        for leg, info in self.legs.items():
                            row[f'{leg}_foot_z'] = float(d.site_xpos[info['site'],2])
                            row[f'{leg}_contact'] = int(contacts[leg])
                        history.append(row)
                    if view is not None:
                        view.cam.lookat[:] = base.xpos
                        view.sync()
                        time.sleep(max(0,dt-(time.perf_counter()-wall)))
            except KeyboardInterrupt:
                reason = 'interrupted'
            finally:
                d.ctrl[:] = 0  # Physics stops; zero torque does not hold a standing robot.
        delta = np.zeros(3) if start is None else d.body('base').xpos-start
        summary = dict(mode=mode, reason=reason, elapsed=float(d.time),
                       forward=float(delta[0]), lateral=float(delta[1]),
                       height=float(d.body('base').xpos[2]),
                       min_upright=min(r['upright'] for r in history) if history else float('nan'))
        return history, summary
