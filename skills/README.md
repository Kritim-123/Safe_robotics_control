# Robot movement skills

This folder contains robot skills, not Codex agent skills.

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
