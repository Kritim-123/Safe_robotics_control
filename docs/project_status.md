# Project assessment

Updated October 1, 2026 (America/Denver). The original proposal describes a
drone. The user explicitly chose **Go2 in MuJoCo, with a spiking neural network**
for this implementation. The proposal PDF has not been changed.

| Component | Implemented | Remaining limitation |
|---|---|---|
| Go2 simulation | Menagerie model, torque-actuator adapter, flat arena and furnished indoor scene | No physical robot validation |
| Classical locomotion | Gait trajectories, leg IK, joint torque PID with limits and anti-windup | Slow gait; no rough terrain or recovery from arbitrary pushes |
| Destination tracking | World-frame start/goal and intermediate waypoints; arrival/timeout/fall/contact outcomes | Local tracking, not a global map planner |
| Robot skill folder | Walk, navigate, bypass left/right, wait, back-off | Classical options; no imported RL locomotion weights |
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
identifies published external Go2 weights, but none has been imported into this
controller. The saved learned artifact is our own **skill-selection SNN**.

The implementation demonstrates the requested control loop. It does not yet
establish that an SNN beats the geometric teacher, saves energy, tolerates
arbitrary moving obstacles, or transfers to hardware. The next milestones are
sensor-derived perception, larger held-out environments, outcome-based skill
learning, and a carefully validated pretrained locomotion adapter.
