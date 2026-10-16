"""The atomic-skill API: WalkSkill, JumpSkill, and the Go2Skills facade.

Each skill OWNS its stateful bits (the gait-phase clock, the last raw action,
and — for the jump — the 5-frame history ring buffer). Call ``reset()`` at the
start of an episode.

Both policies output a 12-d action in [-1, 1], in mjlab JOINT order (the same
order as ``joint_pos`` / ``DEFAULT_JOINT_POS`` — verified against the env's
action term ``target_names``: action[i] drives joint i, NO regrouping). The
action is scaled by ``CTRL_GAIN`` (=3.0, the training/eval bridge's gain) and
the joint POSITION TARGET the PD controller tracks is
``default_joint_pos + 0.25 * (CTRL_GAIN * action)``. The skill methods return
that target; the post-gain ctrl (what the `actions` obs term holds) is kept on
``.last_action``.
"""

from __future__ import annotations

import numpy as np
import torch

from .nets import JumpNorm, WalkerNorm, load_actor, load_critic
from .obs import (ACTION_SCALE, CONTROL_DT, CTRL_GAIN, DEFAULT_JOINT_POS,
                  JUMP_FRAME_DIM, SCAN_N, HistoryBuffer, ObsInputs,
                  build_jump_frame, build_walker_obs)


def _ctrl(policy_out: np.ndarray, clip: bool = True) -> np.ndarray:
  """Post-gain control vector: apply the bridge's ctrl_gain to the policy output.
  This is BOTH the `actions` obs term and the thing the joint-target formula
  scales.

  ``clip`` controls whether the raw action is clamped to [-1,1] BEFORE the gain.
  There are two action-processing paths in the SOURCE, and the DEPLOYMENT one is
  what this package reproduces:
    * The eval/deployment safety-FILTER path (``eval.policies.safety_modules``
      ``fallback_fn``) applies ``clamp(policy._predict, -1, 1)`` before the gain
      — so the arm engaged as a value filter feeds CLAMPED post-gain actions.
      This is what the validated composed result (E098) was measured under, and
      what the package's ``validate_value_filter`` faithfulness gate matches
      (worst |ΔV| back to ~1e-6 with clip=True). Hence the jump arm CLAMPS.
    * The raw training rollout (``base.py`` step_tensor) applies
      ``ctrl = action * ctrl_gain`` with no clamp, so a direct env.step with the
      raw policy mean stores an UNCLIPPED actions term. That path is reproduced
      by ``clip=False`` and kept only as an A/B knob — it is NOT the deployment
      target and desyncs from the validated filter (|ΔV|~0.2).
  The stock SB3 walker (numpy VecEnv path) clamps for the same reason SB3 does
  before stepping."""
  a = np.clip(policy_out, -1.0, 1.0) if clip else policy_out
  return (CTRL_GAIN * a).astype(np.float32)


def _target(ctrl: np.ndarray) -> np.ndarray:
  return (DEFAULT_JOINT_POS + ACTION_SCALE * ctrl).astype(np.float32)


class WalkSkill:
  """Atomic joystick walker (47-d single-frame obs, stock SB3 PPO)."""

  def __init__(self, device: str = "cpu"):
    self.device = device
    self.net = load_actor("walker_actor.pt", 47, device)
    self.norm = WalkerNorm(device)
    self.reset()

  def reset(self):
    self.step = 0
    self.last_action = np.zeros(12, dtype=np.float32)

  def raw_obs(self, inp: ObsInputs, command) -> np.ndarray:
    """The 47-d observation the policy consumes (pre-normalization)."""
    return build_walker_obs(inp, command, self.last_action, self.step)

  def act(self, inp: ObsInputs, command=(1.0, 0.0, 0.0)) -> np.ndarray:
    """One control step. Returns 12 joint position targets (mjlab order)."""
    obs = self.raw_obs(inp, command)
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
    ctrl = _ctrl(a)                      # walker: SB3-clipped post-gain (numpy VecEnv path)
    self.last_action = ctrl
    self.step += 1
    return _target(ctrl)


class JumpSkill:
  """Reach-avoid gap-jumping arm (1175-d = 235x5 history, ReachAvoidPPO1P).

  ``width`` picks the arm:
  'hando_w30' (DEFAULT) = the handover-range finetuned reach-avoid arm (gap
  0.30, finetuned on real walker-handover states — the recommended deployment
  arm); 'w30'/'w20'/'w12' = the from-scratch reverse-curriculum width ladder
  (0.30 / 0.20 / 0.12 m) kept as selectable alternates.

  ``with_critic=True`` also loads the reach-avoid state-value head V(s) (the
  same certificate the arm was trained under), enabling ``value_and_action`` and
  the :class:`Go2ValueFilter`. It is off by default so a bare arm needs only the
  actor."""

  def __init__(self, device: str = "cpu", width: str = "hando_w30",
               with_critic: bool = False, clip_action: bool = True):
    self.device = device
    self.width = width
    # The deployment safety-filter path clamps the arm action to [-1,1] before
    # the ctrl_gain (eval.policies.safety_modules), and that is the convention
    # the validated composed result was measured under — so clip_action=True by
    # default. clip_action=False reproduces the raw (unclamped) training-rollout
    # path and is an A/B knob only; see _ctrl.
    self.clip_action = bool(clip_action)
    self.net = load_actor(f"jump_actor_{width}.pt", 1175, device)
    self.norm = JumpNorm(width, device)
    self.critic = (load_critic(f"jump_critic_{width}.pt", 1175, device)
                   if with_critic else None)
    self.history = HistoryBuffer()
    self.reset()

  def reset(self):
    self.step = 0
    self.last_action = np.zeros(12, dtype=np.float32)
    self.history.reset()

  def raw_frame(self, inp: ObsInputs, command, height_scan: np.ndarray) -> np.ndarray:
    """The 235-d single frame (pre-history, pre-normalization)."""
    return build_jump_frame(inp, command, self.last_action, self.step, height_scan)

  def act(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
          height_scan: np.ndarray | None = None) -> np.ndarray:
    """One control step. ``height_scan`` is 187 raw ray heights (base_z - hit_z,
    miss -> 5.0); if None a flat-ground scan is synthesized (no gap -> the
    policy just walks/holds). Returns 12 joint position targets (mjlab order)."""
    if height_scan is None:
      height_scan = flat_ground_scan(inp.base_z)
    frame = self.raw_frame(inp, command, height_scan)
    self.history.append(frame)
    obs = self.history.flatten()
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
    ctrl = _ctrl(a, clip=self.clip_action)   # jump arm: deployment-filter clamp (clip_action, default True)
    self.last_action = ctrl
    self.step += 1
    return _target(ctrl)

  def value_and_action(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
                       height_scan: np.ndarray | None = None):
    """One control step that returns BOTH the certificate value and the action.

    Assembles the jump frame, pushes it onto the history, and normalizes ONCE,
    then evaluates the critic V(s) and the actor on the SAME normalized obs — so
    a single call advances the history/phase clock exactly once (the env's
    once-per-control-step semantics). Returns ``(V, ctrl)`` where ``V`` is the
    scalar reach-avoid value (safe iff >= 0) and ``ctrl`` is the post-gain
    control vector (12,). Requires the skill built with ``with_critic=True``."""
    if self.critic is None:
      raise RuntimeError(
        "JumpSkill.value_and_action needs the critic; build it with "
        "JumpSkill(..., with_critic=True) (or use Go2ValueFilter).")
    if height_scan is None:
      height_scan = flat_ground_scan(inp.base_z)
    frame = self.raw_frame(inp, command, height_scan)
    self.history.append(frame)
    obs = self.history.flatten()
    o = self.norm(torch.as_tensor(obs, device=self.device).unsqueeze(0))
    with torch.no_grad():
      a = self.net(o).squeeze(0).cpu().numpy()
      v = float(self.critic(o).squeeze(-1).cpu().item())
    ctrl = _ctrl(a, clip=self.clip_action)   # jump arm: deployment-filter clamp (clip_action, default True)
    self.last_action = ctrl
    self.step += 1
    return v, ctrl


def flat_ground_scan(base_z: float) -> np.ndarray:
  """Height scan over featureless flat ground: every ray hits z=0, so the raw
  mdp.height_scan value is base_z everywhere (187,)."""
  return np.full(SCAN_N, float(base_z), dtype=np.float32)


class Go2Skills:
  """Facade owning both skills.

      skills = Go2Skills(device="cpu")
      skills.reset()
      tgt = skills.walk(obs_inputs, command=(vx, vy, wz))
      tgt = skills.jump(obs_inputs, height_scan=None)
  """

  def __init__(self, device: str = "cpu", jump_width: str = "hando_w30"):
    self.device = device
    self.walk_skill = WalkSkill(device)
    self.jump_skill = JumpSkill(device, width=jump_width)

  def reset(self):
    self.walk_skill.reset()
    self.jump_skill.reset()

  def walk(self, obs_inputs: ObsInputs, command=(1.0, 0.0, 0.0)) -> np.ndarray:
    return self.walk_skill.act(obs_inputs, command)

  def jump(self, obs_inputs: ObsInputs, command=(1.0, 0.0, 0.0),
           height_scan: np.ndarray | None = None) -> np.ndarray:
    return self.jump_skill.act(obs_inputs, command, height_scan)


class Go2ValueFilter:
  """The reach-avoid arm deployed as a LEAST-RESTRICTIVE VALUE FILTER.

  Composes the walker (the nominal being filtered) with the jumping arm's
  certificate V(s) and its fallback action. The canonical switching rule, one
  memoryless comparison per step (Hsu, Hu & Fisac, "The Safety Filter"):

      apply the walker action   iff  V(s) >  eps
      else apply the jump action (hand authority to the certified fallback)

  So the arm engages when ``V(s) <= eps``. ``eps`` is the ONLY knob for the
  canonical rule: 0.0 is the value-zero level (the certificate's own safe-set
  boundary); 0.25 is the tuned "shield earlier" value for the w30 arm; a larger
  eps engages the jump earlier / more conservatively.

  LANDING LATCH (opt-in, ``landing_latch``; default OFF -> byte-identical to the
  memoryless rule above). The memoryless switch is correct for BRAKING but wrong
  at the LANDING handover: once the robot is over/past the gap the trigger
  releases (distance -> the scan sees no forward drop so ``dist_to_gap -> inf``;
  value -> ``V(s) > eps`` again mid-flight), so authority snaps back to the
  WALKER *at or before touchdown* — a controller that has never seen the landing
  pose, still commanded 1.0 m/s forward — which head-dives / anchored-front flips
  (measured: jump-kept survival ~0.93 vs stay/walker handover 0.17/0.38 within
  1 s). The latch removes that failure mode: when the trigger first engages the
  jump it LATCHES ``engaged=True`` and holds it through flight AND landing, with
  the jump kept at its REQUESTED command the whole time (never zeroed — a zeroed
  command is OOD for an arm trained under a constant command and causes a
  backward creep into the gap). It releases to the walker per ``release``:
    * ``release="settle"`` — release once the settled-stand criterion has held
      CONTINUOUSLY for ``settle_time_s``. Precise, but can fail to fire if the
      stand creeps (never continuously "settled").
    * ``release="timed"``  — release at a fixed ``release_delay_s`` AFTER
      touchdown, no velocity gate (the sandbox composition's ≥0.90 20-s-survival
      mode: keep the arm through landing, hand over 0.5-1.0 s after touchdown).
      Touchdown = first foot contact after airborne if ``foot_contacts`` are on
      the obs, else the base-height upturn (falling -> not-falling) after descent.
  On release the WALKER's command is ramped 0 -> requested over ``release_ramp_s``
  so the walker does not lurch off a fresh stand. The latch is NOT the canonical
  least-restrictive filter (it is deliberately more restrictive on the release
  side); use OFF for the certificate study, ON for a deployed jump-then-recover
  skill.

  Settled-stand criterion (per step, evaluated only while latched):
    * if per-foot contact inputs are present on the obs (``foot_contacts``, an
      attribute hardware can supply) -> all four feet loaded AND upright;
    * else -> upright (projected-gravity z <= ``-settle_upright``) AND slow:
      base planar speed <= ``settle_speed`` if a base linear velocity is present
      on the obs (``base_lin_vel``), otherwise the joint-velocity norm below
      ``settle_jointvel`` (the hardware-agnostic fallback, since the shipped
      ``ObsInputs`` carries neither foot contact nor base velocity).

  DEPLOYMENT CONTRACT (the switching-filter gotcha this class handles for you):
  the env computes BOTH obs groups from the current sim state, and the
  ``actions`` term in EACH group is the last *applied* (selected) control — not
  each policy's own previous output. So after every step both the walker's and
  the jump's ``last_action`` are set to the SELECTED post-gain control. Getting
  this wrong silently corrupts V; see ``validation/validate_value_filter.py``.
  """

  TRIGGERS = ("value", "distance")

  RELEASES = ("settle", "timed")

  def __init__(self, device: str = "cpu", jump_width: str = "hando_w30",
               eps: float = 0.25, trigger: str = "value", D: float = 0.40,
               clip_action: bool = True, landing_latch: bool = True,
               release: str = "timed", settle_time_s: float = 1.0,
               settle_speed: float = 0.15, settle_upright: float = 0.8,
               settle_jointvel: float = 4.0, release_delay_s: float = 0.75,
               release_ramp_s: float = 0.5):
    self.device = device
    self.eps = float(eps)
    if trigger not in self.TRIGGERS:
      raise ValueError(f"trigger must be one of {self.TRIGGERS}, got {trigger!r}")
    self.trigger = trigger
    self.D = float(D)
    # --- landing-latch config (0.4.3 default ON, release="timed": the deployable
    # setting from the sim2sim landing study; pass landing_latch=False for the
    # memoryless 0.4.1 behaviour) ---
    self.landing_latch = bool(landing_latch)
    if release not in self.RELEASES:
      raise ValueError(f"release must be one of {self.RELEASES}, got {release!r}")
    self.release = release
    self.settle_time_s = float(settle_time_s)
    self.settle_speed = float(settle_speed)
    self.settle_upright = float(settle_upright)
    self.settle_jointvel = float(settle_jointvel)
    self.release_delay_s = float(release_delay_s)
    self.release_ramp_s = float(release_ramp_s)
    self.walk_skill = WalkSkill(device)
    self.jump_skill = JumpSkill(device, width=jump_width, with_critic=True,
                               clip_action=clip_action)
    self.reset()

  def reset(self):
    self.walk_skill.reset()
    self.jump_skill.reset()
    # latch state machine
    self._latched = False          # currently holding the jump past the trigger
    self._settled_s = 0.0          # continuous settled-stand time (s)
    self._releasing = False        # in the post-release command ramp
    self._release_s = 0.0          # time elapsed in the release ramp (s)
    # touchdown detection (for release="timed")
    self._prev_base_z = None       # previous base z (finite-diff vertical vel)
    self._prev_vz = None           # previous vertical velocity estimate
    self._descended = False        # base has been falling since latch (flight)
    self._airborne_seen = False    # all feet left the ground (hardware contacts)
    self._td = False               # touchdown detected
    self._t_since_td = 0.0         # seconds since touchdown

  def _touchdown_now(self, inp: ObsInputs) -> bool:
    """Update touchdown state from the obs; return True on the touchdown frame.

    If per-foot contacts are on the obs (``foot_contacts``): touchdown = the
    first foot contact after an airborne (all-feet-off) phase. Otherwise detect
    it from the base height: the first upturn of vertical velocity (falling ->
    not-falling) after the base has been descending, with the base back near
    stance (a completed landing, not a fall into the gap)."""
    if self._td:
      return False
    fc = getattr(inp, "foot_contacts", None)
    if fc is not None:
      any_contact = bool(np.any(np.asarray(fc, dtype=np.float32) > 0.5))
      if not any_contact:
        self._airborne_seen = True
      if self._airborne_seen and any_contact:
        self._td = True
      return self._td
    # base-height fallback (no foot contacts on the shipped ObsInputs)
    bz = float(inp.base_z)
    if self._prev_base_z is not None:
      vz = (bz - self._prev_base_z) / CONTROL_DT
      if vz < -0.15:
        self._descended = True
      if (self._descended and self._prev_vz is not None
          and self._prev_vz < 0.0 and vz >= 0.0 and bz > 0.18):
        self._td = True
      self._prev_vz = vz
    self._prev_base_z = bz
    return self._td

  def _settled_now(self, inp: ObsInputs, settle_speed: float,
                   settle_upright: float, settle_jointvel: float) -> bool:
    """Settled-stand test for the landing latch (see the class docstring)."""
    upright = float(inp.proj_grav()[2]) <= -float(settle_upright)
    fc = getattr(inp, "foot_contacts", None)
    if fc is not None:                              # hardware per-foot contacts
      all_loaded = bool(np.all(np.asarray(fc, dtype=np.float32) > 0.5))
      return bool(all_loaded and upright)
    blv = getattr(inp, "base_lin_vel", None)
    if blv is not None:                             # base linear velocity present
      slow = float(np.linalg.norm(np.asarray(blv, dtype=np.float32)[:2])) <= float(settle_speed)
    else:                                           # hardware-agnostic fallback
      slow = float(np.linalg.norm(np.asarray(inp.joint_vel, dtype=np.float32))) <= float(settle_jointvel)
    return bool(upright and slow)

  @staticmethod
  def dist_to_gap_from_scan(height_scan: np.ndarray, base_z: float,
                            drop_threshold: float = 0.7) -> float:
    """Estimate distance to the near edge of a forward gap from the raw 187-ray
    scan: the smallest FORWARD x-offset (high-ix column, y-center rows) whose
    drop (scan value - base_z) exceeds ``drop_threshold`` (the training gap
    definition). Returns ``inf`` if no forward drop is visible. Used by the
    distance-trigger when no explicit ``dist_to_gap`` is supplied."""
    from .obs import SCAN_NX, SCAN_NY, SCAN_X
    scan = np.asarray(height_scan, dtype=np.float32).reshape(SCAN_NY, SCAN_NX)
    cy = SCAN_NY // 2
    rows = scan[max(cy - 1, 0):cy + 2, :]          # 3 center y-rows, robust
    drop = rows - float(base_z)
    for ix in range(SCAN_NX):
      if SCAN_X[ix] > 0 and bool((drop[:, ix] > drop_threshold).any()):
        return float(SCAN_X[ix])
    return float("inf")

  def step(self, inp: ObsInputs, command=(1.0, 0.0, 0.0),
           height_scan: np.ndarray | None = None, eps: float | None = None,
           trigger: str | None = None, D: float | None = None,
           dist_to_gap: float | None = None, landing_latch: bool | None = None,
           release: str | None = None):
    """One filtered control step. Returns ``(target, info)`` where ``target`` is
    the 12 joint position targets (mjlab order) and ``info`` carries
    ``{"value", "engaged", "eps", "trigger", "dist_to_gap", "latched",
    "settled_for_s", "cmd_applied", "touchdown", "t_since_td", "release"}``.
    ``height_scan`` is the raw 187 ray heights (base_z - hit_z, miss -> 5.0);
    None synthesizes flat ground (no gap -> the arm holds, the walker passes).

    Two engage modes:
      * ``trigger="value"``    — canonical least-restrictive filter: engage when
        ``V(s) <= eps``.
      * ``trigger="distance"`` — engage when distance to the gap's near edge is
        ``<= D`` (a decision LINE at the last brakeable point — E098's validated
        deployment mode). ``dist_to_gap`` may be passed explicitly (trivial with
        the fake-gap override, where it is known); otherwise it is estimated
        from the scan's first forward drop-off column.
    Both modes always evaluate V (reported in ``info``); only the engage test
    differs.

    ``landing_latch`` (None -> the instance default): when True the raw engage
    above becomes a LATCH — once the jump engages it stays engaged through flight
    AND landing (the jump keeps its requested command the whole time; the command
    is NEVER zeroed on the arm, which is OOD for an arm trained under a constant
    command), and releases to the walker per ``release``:
      * ``release="settle"`` — release once the settled-stand criterion holds
        continuously for ``settle_time_s``. Can fail to fire if the stand creeps.
      * ``release="timed"``  — release at a fixed ``release_delay_s`` AFTER
        touchdown (no velocity gate; the sandbox composition's ≥0.90-survival
        mode). Touchdown is detected from foot contacts if present, else from the
        base-height upturn after descent.
    On release the WALKER's command is ramped 0 -> requested over
    ``release_ramp_s``. See the class docstring."""
    eps = self.eps if eps is None else float(eps)
    trigger = self.trigger if trigger is None else trigger
    D = self.D if D is None else float(D)
    latch = self.landing_latch if landing_latch is None else bool(landing_latch)
    release = self.release if release is None else release
    if release not in self.RELEASES:
      raise ValueError(f"release must be one of {self.RELEASES}, got {release!r}")

    # certificate value + the jump's fallback ctrl, one history/phase advance.
    v, jump_ctrl = self.jump_skill.value_and_action(inp, command, height_scan)
    # raw (memoryless) trigger — always the engage test the canonical filter uses.
    if trigger == "distance":
      if dist_to_gap is None:
        scan = height_scan if height_scan is not None else flat_ground_scan(inp.base_z)
        dist_to_gap = self.dist_to_gap_from_scan(scan, inp.base_z)
      raw_engaged = dist_to_gap <= D
    else:
      dist_to_gap = float("nan") if dist_to_gap is None else dist_to_gap
      raw_engaged = v <= eps

    # --- latch state machine (no-op when latch is OFF) ------------------------
    ramp_frac = 1.0
    if not latch:
      engaged = bool(raw_engaged)
    else:
      if raw_engaged and not self._latched:         # first engage -> latch on
        self._latched = True
        self._settled_s = 0.0
        self._releasing = False
      if self._latched:
        # settle bookkeeping (always tracked; drives release only in settle mode)
        if self._settled_now(inp, self.settle_speed, self.settle_upright,
                             self.settle_jointvel):
          self._settled_s += CONTROL_DT
        else:
          self._settled_s = 0.0
        # touchdown bookkeeping (drives release only in timed mode)
        self._touchdown_now(inp)
        if self._td:
          self._t_since_td += CONTROL_DT
        if release == "timed":
          do_release = self._td and self._t_since_td >= self.release_delay_s
        else:                                         # "settle"
          do_release = self._settled_s >= self.settle_time_s
        if do_release:
          self._latched = False
          self._releasing = True
          self._release_s = 0.0
        engaged = self._latched
      else:
        engaged = bool(raw_engaged)                 # not latched: fall back to raw
      if self._releasing:                           # ramp the handed-back command
        if self.release_ramp_s > 0.0:
          ramp_frac = min(self._release_s / self.release_ramp_s, 1.0)
        else:
          ramp_frac = 1.0
        self._release_s += CONTROL_DT
        if ramp_frac >= 1.0:
          self._releasing = False

    # walker command for this step: full command unless ramping back after release.
    if ramp_frac >= 1.0:
      cmd_walk = command                            # unchanged -> byte-identical
    else:
      cmd_walk = tuple(float(c) * ramp_frac for c in np.asarray(command, dtype=np.float32))
    # nominal (walker) post-gain ctrl; discard its target, keep the ctrl.
    self.walk_skill.act(inp, cmd_walk)
    a_nom = self.walk_skill.last_action

    applied_ctrl = jump_ctrl if engaged else a_nom
    cmd_applied = tuple(np.asarray(command if engaged else cmd_walk, dtype=np.float32))
    # SHARED applied last_action into BOTH groups (see the deployment contract).
    self.walk_skill.last_action = applied_ctrl
    self.jump_skill.last_action = applied_ctrl
    target = _target(applied_ctrl)
    return target, {"value": float(v), "engaged": bool(engaged), "eps": eps,
                    "trigger": trigger, "dist_to_gap": float(dist_to_gap),
                    "latched": bool(self._latched),
                    "settled_for_s": float(self._settled_s),
                    "cmd_applied": cmd_applied, "release": release,
                    "touchdown": bool(self._td),
                    "t_since_td": float(self._t_since_td)}
