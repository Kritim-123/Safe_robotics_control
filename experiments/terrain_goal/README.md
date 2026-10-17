# Terrain and goal selection

This experiment separates skill requests, goal arrival and physical terrain
execution. The original 60-case transfer assessment improved from 0/60 to
44/60 with the first complete controller; the updated rule control also passed
44/60. See the [validation report](../../docs/agents/terrain_goal_validation.md)
for fresh confirmations, noise tests, failures and actual commands.

## Run a finite demonstration

For the retained V3 model on a four-obstacle course with repeated skill changes:

```powershell
.venv-win/Scripts/python.exe -m experiments.terrain_goal.obstacle_demo --viewer --video
```

This single episode includes clearing and crossing obstacles, two side boxes,
an explicit hop request and goal stopping. The headless development check
completed seven handoffs and one hop without collision or fall, finishing
8.2 cm from the goal. The hop is requested by the task stream, rather than
inferred from vision. Video, fresh neuron counts and handoff logs are saved
under `output/terrain_goal/live-v3-obstacles-*`.

From the project root, using the existing environment:

```powershell
.venv-win/Scripts/python.exe -m experiments.terrain_goal.demo --scene steps --viewer --video
.venv-win/Scripts/python.exe -m experiments.terrain_goal.demo --scene noisy_goal --viewer --video
.venv-win/Scripts/python.exe -m experiments.terrain_goal.demo --scene refused_slope --viewer
```

The steps demonstration uses a confirmed course. The noisy replay reuses a
confirmation geometry. The refused slope intentionally remains a failed goal:
the current terrain scan exceeds the motor profile's allowed up-step estimate.
Demonstrations are recordings, not additional independent test evidence.

The default is the frozen V2 goal readout. Launching a demo performs inference
and logs execution; it does not train weights or update associations.

The retained V3 request-data revision uses the same encoder, goal hold and
motors, and improves a separate fresh comparison from 78/100 to 83/100 clean
goals and 30/40 to 31/40 noisy goals. Select it explicitly:

```powershell
.venv-win/Scripts/python.exe -m experiments.terrain_goal.demo --scene steps --model docs/agents/terrain_goal_v3/primary.npz --viewer --video
```

It preserves the previous successful regression cases. Its three known calf
contacts remain reproducible; it does not solve terrain motor planning or
localization bias. The report distinguishes V1/V2/V3 development and fresh tests.

## Observation and skill contract

Task intent is explicit: progress, hold or jump. Goal coordinates and robot
position use metres, yaw uses radians, obstacle velocity uses metres per second.
The observations use privileged MuJoCo state and a known static terrain map.
The noise tests separately perturb selector pose and obstacle observations;
the low-level waypoint controller and physical terminal guards retain current
simulation state. These tests do not implement a camera pipeline.

The encoder has 64 frozen LIF neurons. IDs 0–31 encode task intent and predicted
run/jump/current clearance. IDs 32–63 encode monotonic goal proximity through
eight distance scales. Each 100 ms decision starts 16 numerical LIF updates
with zero membranes and counts. The seeded arrays and encoder checksum are
versioned independently of the learned readout.
Explicit hold tasks retain the older zero-goal feature convention; goal
proximity comparisons use progress packets.

A frozen three-class ridge readout selects wait/run/jump. A count-sum threshold
of 92, learned from training-only goal packets, can replace a run request with
wait. V2 commits that goal hold after two near-goal count packets and releases
it after two packets below 25 spikes, corresponding to the existing 0.16 m
resume boundary. Explicit task changes and episode reset clear commitment.
This is readout state; neuron packets still reset independently.

The existing supervisor handles label confirmation, stable initiation,
braking, pre-thrust cancellation, flight commitment and recovery. A changed
neural request never resets the physical robot or replaces its joint PID.
Wait/run use 24 Nm and the existing short hop uses 28 Nm. The separate low-bar
jump is outside this experiment.

The selected classical terrain profile uses a 0.8 s gait period, 70 percent
stance, 12 cm swing-foot lift and bounded braking. A latched eligibility check
refuses a sampled terrain up-step above 5 cm. This is a limit on the scan
estimate, not a guarantee for arbitrary terrain. Conservative robot-disc
clearance to oriented rectangles improves passage eligibility. Falls, contacts,
guard stops, unsupported surfaces and missed goals remain failed tasks.

Strict goal success requires a supported stopped wait within 0.12 m, held for
0.8 s during a progress mission, within 6 s of first entering that region, and
no later departure beyond 0.16 m. A timed terminal stop near the goal is
insufficient. The historical traversal criterion remains available separately.

## Files and reproduction

The [research notes](research.md) explain goal conditioning, aggregation of
learner-visited states, population coding and supervisory hysteresis.
The [protocol](../../docs/agents/terrain_goal_protocol.md) defines acceptance and
the [second-iteration protocol](../../docs/agents/terrain_goal_hysteresis_protocol.md)
records how V1 noise failures motivated V2 before new confirmation scenes.

```powershell
.venv-win/Scripts/python.exe -m experiments.terrain_goal.package --reproduce output/terrain_goal/my-v1-reproduction
.venv-win/Scripts/python.exe -m experiments.terrain_goal.package_v2 --reproduce output/terrain_goal/my-v2-reproduction
.venv-win/Scripts/python.exe -m experiments.terrain_goal.assessment --phase benchmark --output output/terrain_goal/my-benchmark
.venv-win/Scripts/python.exe -m experiments.terrain_goal.evaluate_release --phase clean --output output/terrain_goal/my-v3-clean
.venv-win/Scripts/python.exe -m experiments.terrain_goal.evaluate_release --phase noise --output output/terrain_goal/my-v3-noise
.venv-win/Scripts/python.exe -m experiments.terrain_goal.package_refinement --reproduce output/terrain_goal/my-v3-reproduction
```

Outputs must use new directories. Saved cases, source snapshots, models,
per-packet observations/counts, desired/actual skills, handoffs, torques and
terminal outcomes are under `output/terrain_goal/`. Portable models and training
records are under `docs/agents/terrain_goal_v1/` and `terrain_goal_v2/`.
The earlier V2 continual selector and its jump demonstration remain available
under `experiments.continual_snn`.
`evaluate_release` compares portable V3/V2/rule artifacts directly, so it does
not require the original raw fitting directory. Its default clean/noise seeds
replay the published 100/40 confirmations; supply a new `--seed` for a new
assessment. `--chunk`/`--chunks` partition complete matched case groups.

Height-offset and preview gait variants, imported HIM comparisons and wider
parameter grids are development diagnostics. Their failures do not qualify
them as alternate motor skills.
