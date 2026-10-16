"""Bridge a user's own MuJoCo Go2 scene to the skill observations.

Optional module — only imported when you call it, and the only place ``mujoco``
is used. Provides:
  obs_from_mujoco     fill ObsInputs from an MjData (maps joint order)
  raycast_height_scan the real 17x11 down-scan via mujoco.mj_ray (best effort)
  fake_gap_scan       a synthetic scan encoding a forward gap (jump trigger)
"""

from __future__ import annotations

import os

import numpy as np

from .obs import (BASE_SPAWN_Z, DEFAULT_JOINT_POS, MJLAB_JOINT_ORDER,
                  SCAN_MAX_DISTANCE, SCAN_N, SCAN_X, SCAN_Y, ObsInputs,
                  scan_grid_offsets, yaw_from_quat)

MJCF_PATH = os.path.join(os.path.dirname(__file__), "assets", "go2_mjcf", "go2.xml")

# PD gains the policies trained with (go2_constants): (stiffness, damping, armature)
_GAINS = {"hip": (20.0, 1.0, 0.01), "thigh": (20.0, 1.0, 0.01),
          "calf": (40.0, 2.0, 0.02)}
# Actuator effort limits (N·m) the training sim enforces (go2_constants): the real
# Go2 torque limits. Without these the PD actuator applies unbounded torque, making
# any package-sim validation optimistic.
_EFFORT_LIMITS = {"hip": 23.5, "thigh": 23.5, "calf": 45.0}


def build_go2_mjcf_model(with_floor=True, with_actuators=True,
                         integrator="implicitfast"):
  """Compile a standalone Go2 MjModel from the bundled MJCF.

  Adds a floor plane and PD **position** actuators (kp=stiffness, kv=damping)
  plus per-joint armature matching the trained gains, in mjlab JOINT order — so
  ``data.ctrl[i]`` is the position target for joint ``MJLAB_JOINT_ORDER[i]``
  (the order the skills return targets in). ``integrator='implicitfast'`` matches
  mjlab. Returns the compiled ``mujoco.MjModel``."""
  import mujoco
  spec = mujoco.MjSpec.from_file(MJCF_PATH)
  if with_floor:
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            pos=[0, 0, 0], rgba=[0.4, 0.5, 0.6, 1])
  if with_actuators:
    for jname in MJLAB_JOINT_ORDER:
      grp = "hip" if "hip" in jname else "thigh" if "thigh" in jname else "calf"
      kp, kv, arm = _GAINS[grp]
      spec.joint(jname).armature = arm
      a = spec.add_actuator(name=jname, target=jname,
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
      a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
      a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
      a.gainprm[0] = kp
      a.biasprm[1] = -kp        # force = kp*(target - q) - kv*qvel
      a.biasprm[2] = -kv
      a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
      lim = _EFFORT_LIMITS[grp]
      a.forcerange[0] = -lim
      a.forcerange[1] = lim
  model = spec.compile()
  model.opt.timestep = 0.005
  if integrator == "implicitfast":
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  return model


def set_default_pose(mj_model, mj_data, base_z=0.33):
  """Place the robot upright at the default joint pose (mjlab order)."""
  import mujoco
  mj_data.qpos[:3] = [0, 0, base_z]
  mj_data.qpos[3:7] = [1, 0, 0, 0]
  padr, _ = _joint_qadr(mj_model, MJLAB_JOINT_ORDER)
  mj_data.qpos[padr] = DEFAULT_JOINT_POS
  mj_data.qvel[:] = 0
  mujoco.mj_forward(mj_model, mj_data)


def _joint_qadr(mj_model, joint_names):
  import mujoco
  padr, vadr = [], []
  for nm in joint_names:
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)
    if jid < 0:
      raise ValueError(f"joint '{nm}' not found in the MuJoCo model")
    padr.append(int(mj_model.jnt_qposadr[jid]))
    vadr.append(int(mj_model.jnt_dofadr[jid]))
  return np.array(padr), np.array(vadr)


def obs_from_mujoco(mj_model, mj_data, joint_names=MJLAB_JOINT_ORDER,
                    base_body="base_link", gyro_sensor="imu_ang_vel") -> ObsInputs:
  """Build ObsInputs from a live MjData.

  ``joint_names`` maps YOUR model's joints to MJLAB_JOINT_ORDER (defaults to the
  mjlab names). Base pose is read from the free-joint root; body-frame angular
  velocity from the ``imu_ang_vel`` gyro if present, else the free-joint qvel
  (already body-frame in MuJoCo)."""
  import mujoco
  padr, vadr = _joint_qadr(mj_model, joint_names)
  joint_pos = mj_data.qpos[padr].astype(np.float32)
  joint_vel = mj_data.qvel[vadr].astype(np.float32)

  bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_body)
  if bid >= 0:
    base_z = float(mj_data.xpos[bid][2])
    base_quat = np.asarray(mj_data.xquat[bid], dtype=np.float32)  # w,x,y,z
  else:                                    # fall back to the free-joint root
    base_z = float(mj_data.qpos[2])
    base_quat = np.asarray(mj_data.qpos[3:7], dtype=np.float32)

  sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_sensor)
  if sid >= 0:
    adr = int(mj_model.sensor_adr[sid])
    base_ang_vel = np.asarray(mj_data.sensordata[adr:adr + 3], dtype=np.float32)
  else:
    base_ang_vel = np.asarray(mj_data.qvel[3:6], dtype=np.float32)

  return ObsInputs(joint_pos=joint_pos, joint_vel=joint_vel,
                   base_ang_vel=base_ang_vel, base_z=base_z, base_quat=base_quat)


def raycast_height_scan(mj_model, mj_data, base_body="base_link",
                        geomgroup=None) -> np.ndarray:
  """Real 187-ray down-scan (base_z - hit_z per cell, miss -> max_distance).

  Yaw-aligned level grid at the base position, rays straight down. Because the
  origin sits at base_z and the ray points down, base_z - hit_z == the ray
  distance.

  mjlab's scan hits the TERRAIN only. ``mj_ray`` casts against the whole scene
  and can only exclude ONE body (the base), so downward rays may hit the
  robot's own legs. To match mjlab, pass ``geomgroup`` — a length-6 uint8 mask
  selecting the geom groups to test (e.g. only the terrain's group). Otherwise
  this is best effort; consider a dedicated terrain-only collision group."""
  import mujoco
  gg = None if geomgroup is None else np.asarray(geomgroup, dtype=np.uint8)
  bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_body)
  if bid >= 0:
    base_pos = np.asarray(mj_data.xpos[bid], dtype=np.float64)
    quat = np.asarray(mj_data.xquat[bid], dtype=np.float64)
  else:
    base_pos = np.asarray(mj_data.qpos[:3], dtype=np.float64)
    quat = np.asarray(mj_data.qpos[3:7], dtype=np.float64)
  yaw = yaw_from_quat(quat)
  c, s = np.cos(yaw), np.sin(yaw)
  offsets = scan_grid_offsets()                      # (187, 2) base-frame xy
  vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  geomid = np.zeros(1, dtype=np.int32)
  out = np.empty(SCAN_N, dtype=np.float32)
  for k, (dx, dy) in enumerate(offsets):
    ox = base_pos[0] + c * dx - s * dy
    oy = base_pos[1] + s * dx + c * dy
    pnt = np.array([ox, oy, base_pos[2]], dtype=np.float64)
    dist = mujoco.mj_ray(mj_model, mj_data, pnt, vec, gg, 1, bid, geomid)
    out[k] = SCAN_MAX_DISTANCE if (dist < 0 or dist > SCAN_MAX_DISTANCE) else dist
  return out


def fake_gap_scan(dist_to_gap: float, gap_width: float, base_z: float = BASE_SPAWN_Z,
                  gap_depth: float = 1.0, full_width: bool = True) -> np.ndarray:
  """Synthesize a height scan with a forward gap — the way to INVOKE the jump
  with no real terrain.

  A gap is a trench across the path: every grid cell whose FORWARD offset x lies
  in [dist_to_gap, dist_to_gap + gap_width] reads a drop (base_z + gap_depth,
  or max_distance if that exceeds the sensor range); all other cells read flat
  ground (base_z). ``dist_to_gap`` must be < 0.8 m to fall inside the scan.
  With ``full_width`` the trench spans all lateral rows (a real gap); otherwise
  only the center row drops.

  This is NOT a pure trigger button: it presents a gap in the forward scan. The
  policy originates a jump only when it also has forward command/momentum and
  judges the gap crossable (see docs/JUMP_TRIGGER.md)."""
  raw = np.full((len(SCAN_Y), len(SCAN_X)), float(base_z), dtype=np.float32)
  drop = min(base_z + gap_depth, SCAN_MAX_DISTANCE)
  x_lo, x_hi = dist_to_gap, dist_to_gap + gap_width
  for ix, xoff in enumerate(SCAN_X):
    if x_lo <= xoff <= x_hi:
      if full_width:
        raw[:, ix] = drop
      else:
        raw[len(SCAN_Y) // 2, ix] = drop
  return raw.reshape(-1).astype(np.float32)          # y-major, matches grid order
