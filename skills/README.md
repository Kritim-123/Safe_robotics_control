# Robot movement skills

This folder contains robot skills, not Codex agent skills.

`obstacle_jump.py` adds experimental `execute_low_bar_jump()` and registry entry
`standalone/low_bar_jump`: a recorded standing jump over a 4 cm-high bar. It uses
a separate frozen 72 N m simulation profile and checks every foot, all collision
geometry, physical contacts and stable landing. Nearby +X tests passed 14/16;
landing contacts remain. It is callable explicitly and excluded from SNN
selection. See the [obstacle-jump report](../docs/agents/obstacle_jump_validation.md).

`jump_profile.py` adds a smooth short-forward hop, using quintic target and gain
transitions. Run `python -m experiments.smooth_jump --viewer --video`, or call
`execute_dynamic_skill('jump', jump_profile='smooth_forward')`. The updated
learned-model demo selects this profile by default. The legacy API default is
retained for reproducing archived in-place-hop outcomes. See the
[jump validation report](../docs/agents/smooth_forward_jump_validation.md).
`JumpConfig` supplies explicit finite torque bounds; the tuned default is
28 N m for this simulation profile, rather than the old 24 N m cap.

The [external-policy report](../docs/agents/pretrained_velocity_validation.md)
adds two frozen Go2 motor policies (`robot_lab`, `HIMLoco`) and seven validated
command options. Use `python -m skills.registry` to discover available entries,
or `python -m skills.pretrained_velocity --policy himloco --skill forward --viewer`.
Failed options remain unavailable. These are standalone callable skills;
the existing SNN output schemas are unchanged.

The separate experiment in `dynamic.py` implements `wait`, `run` (fast trot),
and `jump` (in-place hop). The new output-spiking selector and collected dataset
are documented in `docs/agents/skill_lif_validation.md`; it does not change the
legacy four-action schema below.

* `walk.py`: standalone straight-walking simulation entry point.
* `navigate.py`: standalone world-waypoint tracking entry point.
* `avoidance.py`: persistent-episode options `bypass_left`, `bypass_right`,
  `wait`, and `back_off`, executed by `hybrid/system.py`.
* `policies/selector_snn/`: trained LIF skill selector (`model.npz`), its
  schema/provenance (`manifest.json`) and training metrics (`training.json`).

The avoidance options supply targets and termination conditions. They never
reset or advance MuJoCo themselves. Joint torque PID remains active throughout.
The network selects an option; it does not output joint torques.

Adding or reordering options changes the SNN output schema and requires
updating the teacher, dataset, masks, tests and retraining the network. Merely
copying a locomotion checkpoint here does not make it compatible. See
`docs/hybrid_system.md` and `docs/pretrained_policies_and_hybrid_control.md`.
