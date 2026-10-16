"""Bit-for-bit reconstruction of the two mjlab observation groups.

Every term order, scale, and the jump's 5-frame term-major history are what the
mjlab ``observation_manager`` produces for these tasks. Verified against the live
env in ``validation/validate_obs.py`` (max|manual - mjlab| < 1e-4 for both).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# --- Go2 / mjlab constants ---------------------------------------------------

#: mjlab joint enumeration order (the order joint_pos / joint_vel / actions use).
MJLAB_JOINT_ORDER = (
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

#: default joint positions in MJLAB_JOINT_ORDER (from go2_constants.INIT_STATE).
DEFAULT_JOINT_POS = np.array(
  [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8],
  dtype=np.float32)

BASE_SPAWN_Z = 0.32            # nominal standing base height (m)
ACTION_SCALE = 0.25            # JointPositionActionCfg.scale
CONTROL_DT = 0.02             # 50 Hz (sim timestep 0.005 x decimation 4)

# The training/eval bridge (robot_safety_sandbox.base) multiplies the policy's
# [-1,1] action by ctrl_gain BEFORE the env applies it, and the env stores that
# post-gain vector as the `actions` obs term. For the go2 tasks ctrl_gain = 3.0
# (registry default). So, per control step, with a = clip(policy_out, -1, 1):
#     ctrl         = ctrl_gain * a                    # what the env receives
#     last_action  = ctrl                             # the `actions` obs term
#     joint target = default_joint_pos + ACTION_SCALE * ctrl   (effective 0.75*a)
CTRL_GAIN = 3.0

WALK_PHASE_PERIOD = 0.6        # velocity walker gait clock period (s)
JUMP_PHASE_PERIOD = 0.5        # parkour/gap gait clock period (s)
PHASE_STAND_EPS = 0.1          # phase zeroed when |command| < this

# height-scan grid (RayCastSensor GridPatternCfg size=(1.6,1.0) res=0.1)
SCAN_MAX_DISTANCE = 5.0
SCAN_SCALE = 1.0 / SCAN_MAX_DISTANCE        # 0.2
SCAN_X = np.round(np.arange(-0.8, 0.8 + 0.05, 0.1), 3).astype(np.float32)  # 17
SCAN_Y = np.round(np.arange(-0.5, 0.5 + 0.05, 0.1), 3).astype(np.float32)  # 11
SCAN_NX = len(SCAN_X)          # 17
SCAN_NY = len(SCAN_Y)          # 11
SCAN_N = SCAN_NX * SCAN_NY      # 187

JUMP_HISTORY = 5
WALK_OBS_DIM = 47
JUMP_FRAME_DIM = 235
JUMP_OBS_DIM = 1175


def scan_grid_offsets() -> np.ndarray:
  """Ray XY offsets in the base (yaw-aligned) frame, in mjlab flatten order.

  meshgrid(x, y, indexing='xy').flatten() is y-MAJOR: ray k = iy*17 + ix, with
  iy over the 11 y-rows (-0.5..+0.5) and ix over the 17 x-cols (-0.8..+0.8).
  Forward ('ahead') cells are the high-ix columns; center row is iy=5 (y=0).
  Returns (187, 2) array of (x_offset, y_offset).
  """
  gx, gy = np.meshgrid(SCAN_X, SCAN_Y)      # numpy default indexing='xy' -> (11,17)
  return np.stack([gx.flatten(), gy.flatten()], axis=1).astype(np.float32)


# --- small quaternion helpers (w, x, y, z; mujoco/mjlab convention) ----------

def quat_to_rot(q) -> np.ndarray:
  """Rotation matrix R (world<-body) from unit quaternion (w,x,y,z)."""
  w, x, y, z = [float(v) for v in q]
  n = math.sqrt(w * w + x * x + y * y + z * z)
  if n == 0:
    return np.eye(3, dtype=np.float32)
  w, x, y, z = w / n, x / n, y / n, z / n
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=np.float32)


def projected_gravity_b(q) -> np.ndarray:
  """Gravity unit vector expressed in the base frame = R(q)^T @ [0,0,-1].

  Matches mjlab Entity.projected_gravity_b (quat_apply_inverse(quat,[0,0,-1])).
  Upright -> [0,0,-1]."""
  R = quat_to_rot(q)
  return (R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)


def yaw_from_quat(q) -> float:
  w, x, y, z = [float(v) for v in q]
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --- observation inputs the user fills from their own state ------------------

@dataclass
class ObsInputs:
  """The raw state a user supplies each control step, in mjlab semantics.

  Fill ``base_ang_vel`` and ``projected_gravity`` directly, OR pass
  ``base_quat`` (w,x,y,z) and let the builders derive projected_gravity (and,
  for the raycast helper, yaw). All joint arrays are in MJLAB_JOINT_ORDER.
  """
  joint_pos: np.ndarray                       # (12,) absolute joint angles
  joint_vel: np.ndarray                       # (12,) joint velocities
  base_ang_vel: np.ndarray                    # (3,) body-frame angular velocity
  base_z: float = BASE_SPAWN_Z                # world z of the base (jump only)
  projected_gravity: np.ndarray | None = None  # (3,) body-frame gravity unit
  base_quat: np.ndarray | None = None          # (4,) w,x,y,z (fallback source)

  def proj_grav(self) -> np.ndarray:
    if self.projected_gravity is not None:
      return np.asarray(self.projected_gravity, dtype=np.float32)
    if self.base_quat is not None:
      return projected_gravity_b(self.base_quat)
    raise ValueError("ObsInputs needs projected_gravity or base_quat")


def phase_term(step: int, period: float, command) -> np.ndarray:
  """mjlab parkour.mdp.phase: [sin(2pi p), cos(2pi p)], p = (step*dt % T)/T.

  NOTE the order is (sin, cos). Zeroed when ||command|| < 0.1."""
  p = ((step * CONTROL_DT) % period) / period
  out = np.array([math.sin(p * 2 * math.pi), math.cos(p * 2 * math.pi)],
                 dtype=np.float32)
  if float(np.linalg.norm(command)) < PHASE_STAND_EPS:
    out[:] = 0.0
  return out


# --- walker single frame (47-d) ---------------------------------------------

def build_walker_obs(inp: ObsInputs, command, last_action: np.ndarray,
                     step: int) -> np.ndarray:
  """actor group: base_ang_vel(3), projected_gravity(3), command(3), phase(2),
  joint_pos(12), joint_vel(12), actions(12)  ->  47."""
  command = np.asarray(command, dtype=np.float32)
  jp = np.asarray(inp.joint_pos, dtype=np.float32) - DEFAULT_JOINT_POS
  jv = np.asarray(inp.joint_vel, dtype=np.float32)
  return np.concatenate([
    np.asarray(inp.base_ang_vel, dtype=np.float32),
    inp.proj_grav(),
    command,
    phase_term(step, WALK_PHASE_PERIOD, command),
    jp, jv,
    np.asarray(last_action, dtype=np.float32),
  ]).astype(np.float32)


# --- jump single frame (235-d) ----------------------------------------------

def build_jump_frame(inp: ObsInputs, command, last_action: np.ndarray,
                     step: int, height_scan: np.ndarray) -> np.ndarray:
  """proprioception single frame: base_ang_vel(3), projected_gravity(3),
  command(3), phase(2), base_height(1), joint_pos(12), joint_vel(12),
  actions(12), height_scan(187, x0.2)  ->  235.

  height_scan is the RAW mdp.height_scan value (base_z - hit_z per ray, miss ->
  5.0); the 0.2 scale is applied HERE (per mjlab: scale before history stack)."""
  command = np.asarray(command, dtype=np.float32)
  jp = np.asarray(inp.joint_pos, dtype=np.float32) - DEFAULT_JOINT_POS
  jv = np.asarray(inp.joint_vel, dtype=np.float32)
  scan = np.asarray(height_scan, dtype=np.float32) * SCAN_SCALE
  return np.concatenate([
    np.asarray(inp.base_ang_vel, dtype=np.float32),
    inp.proj_grav(),
    command,
    phase_term(step, JUMP_PHASE_PERIOD, command),
    np.array([inp.base_z], dtype=np.float32),
    jp, jv,
    np.asarray(last_action, dtype=np.float32),
    scan,
  ]).astype(np.float32)


class HistoryBuffer:
  """5-frame history with mjlab's backfill + term-major flatten.

  On the first append after a reset every slot is filled with that frame
  (mjlab CircularBuffer backfill). ``flatten()`` returns the 1175-vector in
  TERM-MAJOR order: for each of the 9 terms, its 5 frames oldest->newest.
  """

  # (name, dim) in single-frame concat order.
  TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3),
           ("phase", 2), ("base_height", 1), ("joint_pos", 12),
           ("joint_vel", 12), ("actions", 12), ("height_scan", SCAN_N))

  def __init__(self, length: int = JUMP_HISTORY, frame_dim: int = JUMP_FRAME_DIM):
    self.length = length
    self.frame_dim = frame_dim
    self._buf: list[np.ndarray] = []       # chronological oldest->newest

  def reset(self):
    self._buf = []

  def append(self, frame: np.ndarray):
    frame = np.asarray(frame, dtype=np.float32)
    if not self._buf:                       # backfill on first push after reset
      self._buf = [frame.copy() for _ in range(self.length)]
    else:
      self._buf.append(frame)
      self._buf = self._buf[-self.length:]

  def flatten(self) -> np.ndarray:
    frames = np.stack(self._buf, axis=0)    # (L, 235) oldest->newest
    out = []
    off = 0
    for _name, dim in self.TERMS:
      out.append(frames[:, off:off + dim].reshape(-1))  # (L*dim,) frame-major
      off += dim
    return np.concatenate(out).astype(np.float32)
