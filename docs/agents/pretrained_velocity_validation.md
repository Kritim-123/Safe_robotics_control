# Two external locomotion policies added to the Go2 skill library

Measured October 16, 2026, sequentially in the existing Windows environment.
**Two frozen learned motor policies now provide seven available command options.**
They are callable through `skills.registry`; the SNNs still have their original
four-action and three-action schemas. No new output classes or weights were
silently added to either SNN.

## How far our current policies have developed

Our original SNN selects classical avoidance options and demonstrated 40/40
navigation successes in its recorded main holdout, matched by its geometric
teacher. The newer LIF selector emits actual output spikes for wait/run/jump
and uses a dataset of 252 simulated attempts, but fulfilled 9/18 tasks on new
conditions and 3/18 on unfamiliar obstacles. The hop has measured flight and
landing; it does not clear obstacles. Retained membranes did not change actions
in the earlier frozen-model study, and imported gap jumping succeeded in 0/7 crossings.
Those remain historical evidence, not newly rerun results in this milestone.

The new work fills a different gap: learned velocity control with measured
reverse, lateral and turning options. It does not solve the selector's
generalization problem or establish learned parkour.

## Research and source selection

The [research and contract](pretrained_velocity_research.md) compares rl_sar,
go2_rl_gym, go2-parkour and the already imported atomic-skill package. We selected
rl_sar's compact `robot_lab` and `HIMLoco` packages because weights, inference
contracts and an Apache-2.0 license are available together. More complex MoE
and parkour packages remain candidates, not integrated capabilities.

Upstream: [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar), pinned revision
`376d42c9b128f963ab08579762d5a216a976ce39`. Original TorchScript exports,
configs, base config and license are retained in `skills/policies/rl_sar/`.
Per-policy manifests record source/export hashes and exact contracts. The numeric
NumPy exports preserve the original weights; **no local policy training was done**.
Runtime uses NumPy/MuJoCo only. Torch is needed for conversion and parity tests.

## Integration and development findings

`PretrainedVelocityPolicy` binds joints by name, builds the source-specific
observation vector, maintains previous applied actions, and preserves HIM's
six newest-first history frames across command changes. It returns joint targets
without stepping or resetting physics. The standalone runner uses source fixed
PD gains for initial standing, source locomotion PD gains after entry, zero
integral gain, and the existing `JointPID` torque limiter. The allowed torque is
the minimum of source, model and project limits: at most 23.5 Nm for robot_lab
and 24 Nm for HIM, despite HIM's 33.5 Nm source setting.

The final runs use the **original project geometry, damping, friction-loss
settings and 2 ms physics step**, with 20 ms inference. No body forces or
mid-episode state placement are used. Initial nominal joint pose and .32 m base
height are set once before physics; skills start after two seconds of standing.
Contact, clearance, finite-state and posture guards run every physics step.

Development exposed two adapter errors: holding the nominal pose with locomotion
gains could sag before entry, and MuJoCo's local `mjOBJ_BODY` velocity used the
rotated inertial frame rather than the expected link/IMU frame. Standing now uses
the supplied fixed gains, and world angular velocity is transformed by the link
rotation. A regression test checks it against the actual gyro under a nontrivial
orientation/angular velocity. An intermediate damping/friction-loss comparison
also failed; its changes were discarded. None of these initial development
failures are presented as successful transfer tests.

Raw development records are retained under `output/pretrained_velocity/` in
`development`, `development-v2`, `source-damping-development`, and
`frame-corrected-development`. The first two contain the faulty adapter;
their outcomes must not be used to rank the final policies.

## Frozen-command evaluation

Six commands per policy, each with a requested four-second duration, followed
by zero command and stable recovery. Commands are sampled at 50 Hz; recorded
stop times can be one 20 ms tick late at floating-point boundaries. Three
separate contexts varied yaw/friction: `.2/.7`, `-.2/.9`,
and `.4/.65` (radians/coefficient), IDs 6200–6202. IDs are provenance, not random
perturbations. Commands and thresholds were fixed in the research contract;
no tuning on these contexts was performed. The final collection repeats those
same contexts after adding CLI availability checks; it is a verification replay,
not an additional independent holdout.

| Command option | robot_lab | HIMLoco | Available after validation |
|---|---:|---:|---|
| Forward (.5 m/s command) | 0/3 | 3/3 | HIM only |
| Reverse (-.3 m/s command) | 2/3 | 3/3 | HIM only |
| Sidestep left (.3 m/s command) | 3/3 | 3/3 | Both |
| Sidestep right (-.3 m/s command) | 0/3 | 3/3 | HIM only |
| Turn left (.5 rad/s command) | 0/3 | 0/3 | Neither |
| Turn right (-.5 rad/s command) | 3/3 | 3/3 | Both |
| **Total** | **8/18** | **15/18** | **Seven options, each 3/3** |

Overall: **23/36 successful tasks**, no obstacle contacts or falls. The seven
available options account for 21/21 evaluation successes and also passed their
single development context. This small flat-ground set is not a general success
probability. The 13 unsuccessful trials remain failures: timeouts or insufficient
motion/excessive drift, despite staying upright. Robot_lab reverse is disabled
because one context did not settle in time. The CLI refuses unavailable options
unless explicitly invoked with `--allow-unvalidated` for an experiment.

HIM forward displacement was 1.485–1.515 m including deceleration/recovery,
with .125–.156 m cross-axis drift. HIM reverse progressed .578–.624 m.
These are measured distances, not proof that the commanded velocities were
tracked exactly. All accepted trials require signed progress, bounded drift,
and .8 seconds with four foot contacts, linear speed <.04 m/s, angular speed
<.2 rad/s and tilt <12 degrees. Command duration is not equivalent to episode
duration. Torque saturation occurred, including 24 Nm in HIM forward; limits
were preserved rather than raised to obtain success.

Tracked evidence: [`validation.json`](../../skills/policies/rl_sar/validation.json)
contains all 36 outcomes, thresholds, source hashes and availability decisions.
Raw summaries, trajectories and inference inputs/actions are in
`output/pretrained_velocity/final/`. NumPy/Torch parity was independently checked
on 340 robot_lab and 380 HIM actual rollout inputs: maximum absolute action
errors were approximately 4.18e-6 and 2.15e-6, respectively. Initial export parity
used 128 separate synthetic vectors per network. Neither is a physical test.

## Use the added skills

From the repository root:

```powershell
# List available skills; --all also shows disabled experiments
.\.venv-win\Scripts\python.exe -m skills.registry
.\.venv-win\Scripts\python.exe -m skills.registry --all

# View a validated learned forward or lateral option
.\.venv-win\Scripts\python.exe -m skills.pretrained_velocity --policy himloco --skill forward --viewer
.\.venv-win\Scripts\python.exe -m skills.pretrained_velocity --policy himloco --skill sidestep_right --viewer

# Reproduce the sequential comparison and regression checks
.\.venv-win\Scripts\python.exe -m hybrid.evaluate_pretrained_velocity --output output/pretrained_velocity/final
.\.venv-win\Scripts\python.exe -m hybrid.audit_pretrained_velocity
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Python entry point:

```python
from skills.registry import run_skill
history, summary = run_skill('pretrained/himloco/reverse')
print(summary['success'], summary['reason'])
```

`run_skill` enforces registry availability. The lower-level
`execute_pretrained_skill` remains an explicit experimental runner so failed
options can be reproduced. Both create a standalone episode. The adapter class
can supply persistent targets, but switching from the legacy gait to these
policies and back has **not** been validated or wired into `HybridSystem`.
Doing that requires compatible handoffs and a new outcome dataset/SNN schema
where additional skill classes are desired. It must not be inferred from these
standalone successes. Retained HIM history is internal motor-policy state;
it is unrelated to the SNN membrane-retention experiment.

## Checks and remaining limitations

The full regression suite passed **85 tests in 51.080 seconds**; the log is
retained locally at `output/pretrained_velocity/tests-final.log`. Focused tests
cover real motion/recovery, torque limits, continuity, failed initiation,
interruption, timeout, callback cleanup, nonfinite inputs, checksum refusal,
joint binding, angular frame, history/reset, numeric parity and registry gates.
After adding checkpoint-hash gating, all six focused tests passed again in
3.345 s, including refusal to inherit availability after a checkpoint change.
The public HIM forward CLI also completed and recovered successfully.

Asset source/config hashes and license are retained for reproduction. To repeat
conversion after downloading the exact manifest-listed source files under the
same upstream paths, run:

```powershell
.\.venv-win\Scripts\python.exe -m skills.import_pretrained --source tmp/rl_sar_upstream
```

The importer checks known SHA-256 values before loading the source TorchScript.
Raw files in `output/` and downloaded scratch copies remain git-ignored; the
two policy packages and compact validation are tracked project assets. A Git
attribute preserves their original bytes across platforms.

The audit also repeats the original development context and records it in
`output/pretrained_velocity/audit-development/`; those repetitions are not
additional independent validation contexts. Availability is matched to the
exact source/export hashes rather than inherited by a replacement checkpoint.

These policies are blind velocity controllers; they do not sense obstacles,
plan routes, cross gaps, climb stairs, or establish hardware safety here. Noisy
observations, pushes, longer sequences, arbitrary terrain and legacy-policy
handoffs remain untested for these new motor skills. Source-model differences,
including the physics period, remain. Existing simulation guards are failure
detectors, not formal safety guarantees.
