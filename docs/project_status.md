# Project assessment

See the [consolidated progress summary](project_summary.md) for all milestones
through October 16, 2026, including the new-obstacle generalization failures,
pretrained-policy transfer results and which artifacts are included in Git.

October 16, 2026 pretrained-jump experiment: imported the MIT-licensed
SafeRoboticsLab `hando_w30` actor into a separate simulation adapter. Flat
walking control traversed and settled; 0/7 gap crossings succeeded under the
project's torque/posture limits. Default jump and SNN outputs are unchanged.
See [source, evidence and viewer command](agents/pretrained_jump.md).

October 16, 2026 update: fast trot and in-place jump primitives are tested;
252 actual simulation attempts across 84 conditions are recorded for training.
A new three-output LIF selector drives wait/run/jump from binary output spikes.
Final new-seed execution: 18/18 primitive completions, 9/18 task fulfillments,
no contact/falls; full suite: 75 tests passed. See
[run/jump dataset and selector report](agents/skill_lif_validation.md).

October 15, 2026 update: optional `yield_hold`/`retreat_hold` composites and an
across-observation LIF retention ablation are implemented. The fresh 87-episode
study found no changed selected actions from memory; composite hold semantics
introduced timeouts behind persistent blockers. Defaults remain unchanged.
See [measured report](agents/integration_validation.md) and
[skill catalog](agents/skill_catalog.md). The final suite passed 61 tests.

Updated October 1, 2026 (America/Denver). The original proposal describes a
drone. The user explicitly chose **Go2 in MuJoCo, with a spiking neural network**
for this implementation. The proposal PDF has not been changed.

| Component | Implemented | Remaining limitation |
|---|---|---|
| Go2 simulation | Menagerie model, torque-actuator adapter, flat arena and furnished indoor scene | No physical robot validation |
| Classical locomotion | Gait trajectories, leg IK, joint torque PID with limits and anti-windup | Slow gait; no rough terrain or recovery from arbitrary pushes |
| Destination tracking | World-frame start/goal and intermediate waypoints; arrival/timeout/fall/contact outcomes | Local tracking, not a global map planner |
| Robot skill folder | Walk, navigate, bypass left/right, wait, back-off, run, in-place hop; experimental pretrained gap adapter | Pretrained gap policy failed transfer tests and is excluded from SNN selection |
| Spiking network | Trained 12-64-32-4 LIF selector, 16-step spike encoding, NumPy deployment | Geometric-teacher imitation; no RL or event-camera input |
| Handoffs | Persistent physics; skill completion, stable recovery, return to baseline; optional moving-hazard preemption | Local preemption is not a global planner; guards are not formal safety guarantees |
| Obstacles/disturbances | Static and moving cylinders, constrained passages, lateral push, friction variation | Scripted mocap obstacles cannot be pushed out of their paths |
| Perception | Ground-truth obstacle states with optional noise, delay and smoothing | Camera/LiDAR detection and tracking not implemented |
| Evaluation | Paired baseline/rules/SNN batches; trajectories, events, videos, spike rasters | Small scenario families; no claim of broad generalization |
| Usability | CLI, editable scenario JSON, executed notebook, retraining entry point | Notebook frontend must select the Windows environment |

Start with `Go2_hybrid.ipynb`, [the hybrid guide](hybrid_system.md) and
[measured validation](hybrid_validation.md). The original `Go2_runner.ipynb`
continues to provide the standalone walk experiment.

The research in [pretrained policies](pretrained_policies_and_hybrid_control.md)
identifies published external Go2 weights. The default controller still uses our
own **skill-selection SNN**; imported motor weights now exist only in the
experimental pretrained gap-jump adapter described above.

The implementation demonstrates the requested control loop. It does not yet
establish that an SNN beats the geometric teacher, saves energy, tolerates
arbitrary moving obstacles, or transfers to hardware. The next milestones are
sensor-derived perception, larger held-out environments, outcome-based skill
learning, and a carefully validated pretrained locomotion adapter.
