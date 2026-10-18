# Project assessment

October 17, 2026 V5 reliability refinement: all five reported failures now
complete autonomously. The prior twenty worlds are development regressions;
the frozen fresh paired assessment completes 40/40 courses versus original
V5's 31/40, gaining nine successes and losing none, with no refined contact,
fall, refusal or overshoot. All 211 tests passed. The
[report](agents/v5_refinement_validation.md) documents the separate 72 Nm
landing-profile revision, preserved encoder population plus finer cells,
new outcome bank, exact state-matched contact repair and fresh comparison.
Original V3/V4/V5 artifacts and physical guards are preserved.

October 17, 2026 V5 walk-up update: the robot walks to a static low bar,
brakes to a supported stance, selects the jump from a new spike-count outcome
bank and resumes V3 goal tracking in the same episode. Frozen assessment:
15/20 complete courses, four learned refusals, one rear-thigh jump contact,
no fall or approach overshoot; six unsuitable-target controls made no jump.
All 204 project tests passed. The [V5 report](agents/approach_low_bar_v5_validation.md)
records measured launch state, source/model freeze, exact decision replay,
remaining contact failure and the distinction between classical braking and
learned jump selection. V3, V4 and the jump motor profile remain unchanged.

October 17, 2026 V4 observation-selected low-bar update: a separate 168-cell
LIF count-association head selects the existing low-bar jump without a jump
request, verifies complete crossing/landing and resumes the unchanged V3
goal tracker. Held-out assessment: 18/20 low-bar crossings and goal stops,
two learned count-conflict refusals, no contact/fall; six unsuitable targets
caused no jump, and a clear-path control reached its goal without jumping.
The scope starts already in the tested +X launch stance. The
[V4 report](agents/autonomous_low_bar_v4_validation.md) retains limitations,
the invalid-start validation control contact and reproducible evidence.

October 17, 2026 terrain/goal update: the opt-in 64-neuron count selector and
bounded terrain gait improve the original 60-case course from 0/60 to 44/60,
matching rule control. A separately confirmed goal-hold readout passes 76/100
fresh mild-noise and 69/100 severe-noise goals, versus 46/100 and 22/100 for
the first goal readout. Neuron packets reset; goal commitment persists until
fresh release evidence. All 194 project tests passed, with motor limits and
the earlier runtime preserved. Wider-terrain and localization limitations,
fresh confirmations, actual counts and reproducible commands are in the
[terrain/goal assessment](agents/terrain_goal_validation.md).

The subsequent V3 request-data refinement passes 83/100 fresh clean and 31/40
fresh noisy goals versus V2's 78/100 and 30/40 on the same cases, with the
encoder, goal-hold code and motor profile unchanged. It preserves every prior
successful request-regression/original-course case. The report keeps these
140 new geometries separate from V2's earlier confirmations and records the
three known calf contacts that remain reproducible under V3.

October 17, 2026 continuous SNN selection: a frozen 80-neuron count encoder
and expanded association bank support repeated decisions with checked braking,
jump commitment, recovery and fresh count packets. A matched same-model control
improved from 60/80 to 70/80 fresh stress tasks after targeted association-data
coverage; regular/noisy confirmation passed 80/80 and 32/32. The original16
bank also benefits strongly from motion/task-stream data. Terrain/goal transfer
failed 0/60 for that earlier model, including flat controls; it is an experimental request-stream
selector, with privileged simulation observations. All 176 tests passed.
See the [models, evidence, limitations and replay commands](agents/continual_snn_validation.md).

October 16, 2026 low-obstacle jump: a separate 72 N m simulation profile clears
a 4 cm-high, 2 cm-deep bar in a 64.7 cm standing jump with stable recovery,
all feet and collision geometry beyond it, and no contact. Fresh assessment:
19/24 broad-heading crossings, 4/4 boundary cases, 14/16 +X cases; all six
farther/friction stress cases failed. Both interruptions recovered. This is an
experimental callable skill outside the SNN schemas. All 141 tests passed; see
[recordings, measured limits and failures](agents/obstacle_jump_validation.md).

October 16, 2026 jump stress assessment: 120 new fixed conditions, unchanged
forward-hop profile. Of 100 ordinary cases: 49 qualified forward hops, six short
completed hops, 28 setup failures, seven refused starts and ten execution
failures. All 20 interruption cases recovered. Flat heading/start variations
passed 10/10; low friction reduced travel, raised terrain exposed setup/lost-flight
problems, and approaching obstacles caused eight clearance stops. All 120 traces
were audited and eight jump tests passed. See the
[full condition table, limitations and finite replays](agents/jump_stress_validation.md).

October 16, 2026 smooth jump: after 217 development candidates, a quintic
forward-hop profile completed all 16 fresh flat-ground tests with 16-20 cm
forward travel, about 8 cm airborne travel, lower peak base acceleration and
torque-step RMS. The old 24 N m bound was relaxed; the selected simulation
profile uses 28 N m after testing bounds up to 48. Landing force is higher than
the small legacy hop. All 134 tests passed. See the
[measured tradeoffs, stress failures and replay commands](agents/smooth_forward_jump_validation.md).

October 16, 2026 expanded association learning: 204 new matched skill trials
and 256 comparison/regression episodes produced 94-example association banks.
A dense terrain scan resolves the training flat/threshold count conflict.
Fresh evaluation fulfilled 7/28 tasks versus 0/28 for the original bank;
the old regression tasks stayed at 6/12. Existing motor limits and SNN random
weights remain fixed. See [data, models, failures and commands](agents/expanded_association_validation.md).

October 16, 2026 terrain curriculum: ten ordered levels and 60 seeded scene
configurations now include collidable ramps, thresholds, steps, uneven blocks,
cross-slopes, friction transitions and mixed obstacle courses. Assessment
compares frozen pattern associations with direct run, records live outcomes,
and optionally stops progression at the first failed level. This adds terrain
and evaluation tools, rather than terrain training. See the
[actual curriculum results and limits](agents/terrain_curriculum_validation.md).

October 16, 2026 external-policy expansion: imported frozen rl_sar robot_lab
and HIMLoco networks with source configs, license, checksums and NumPy exports.
The 36-trial comparison passed 23 tasks without contacts/falls; seven command
options passed all three contexts and are available through `skills.registry`.
The SNN schemas and default navigation controller are unchanged. See
[integration, actual results and remaining handoff work](agents/pretrained_velocity_validation.md).

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
own **skill-selection SNN**; imported motor weights are used in the experimental
gap adapter and the separately validated velocity skills described above.

The implementation demonstrates the requested control loop. It does not yet
establish that an SNN beats the geometric teacher, saves energy, tolerates
arbitrary moving obstacles, or transfers to hardware. The next milestones are
sensor-derived perception, larger held-out environments, outcome-based skill
learning, and a carefully validated pretrained locomotion adapter.
