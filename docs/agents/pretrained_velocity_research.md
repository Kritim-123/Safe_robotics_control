# External Go2 policies: assessment and integration contract

Research date: October 16, 2026. One agent owns this report, the new
`skills/pretrained_velocity.py`, `skills/import_pretrained.py`,
`skills/registry.py`, `skills/policies/rl_sar/`, evaluation code and tests.
Existing selector weights and action schemas are outside this milestone.

## Assessment of our policies

The avoidance selector is trained geometric-teacher imitation with classical
movement underneath. Its recorded main holdout was 40/40, matched by rules.
The separate wait/run/jump model learns actual binary output spikes from 252
attempts, but new-seed execution fulfilled only 9/18 tasks; unfamiliar-obstacle
execution fulfilled 3/18. The in-place hop is functional but does not clear
obstacles. Frozen membrane retention did not change chosen actions. Imported
gap jumping failed all seven transfer conditions. These are historical results
from the linked project reports, not fresh measurements for this milestone.

The useful missing capability is commandable learned locomotion with tested
reverse, lateral and turning behavior, rather than another unverified jump.

## Candidates checked online

| Published package | What is available | Decision |
|---|---|---|
| [rl_sar](https://github.com/fan-ziqi/rl_sar) Go2 `robot_lab` | Small TorchScript velocity actor, config and Apache-2.0 repository license | Integrate and measure on our model |
| Same repository, `HIMLoco` | Different actor with integrated estimator and six-frame observation history | Integrate as a separate candidate; retain our torque ceiling despite higher upstream limits |
| [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) | MuJoCo deployment code and linked exported MoE models; MIT additions/BSD upstream notices | Strong follow-up; defer a third locomotion adapter until the two smaller packages are measured |
| [JustinMLu/go2-parkour](https://github.com/JustinMLu/go2-parkour) | Parkour deployment and network assets | Defer: requires several auxiliary networks and a correctly reproduced perception/history contract |
| [SafeRoboticsLab atomic skills](https://github.com/SafeRoboticsLab/go2_atomic_skills) | Walker and scan-dependent reach-avoid jumping | Already imported; recorded gap-transfer failures remain failures |

Upstream claims are not local test evidence. Commands to a velocity network
create parameterized skill options; they are not additional independently
trained networks, obstacle planners or new SNN classes.

## Pinned observation and action contract

Use rl_sar revision `376d42c9b128f963ab08579762d5a216a976ce39`.
Preserve source bytes, license and SHA-256 values. Export only numeric tensors
into non-pickle NumPy archives; compare exported inference against TorchScript
before use. Runtime requires NumPy and MuJoCo, not Torch or YAML.

Sources: [robot_lab config](https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/robot_lab/config.yaml),
[HIMLoco config](https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/himloco/config.yaml),
[base joint names and timing](https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/base.yaml),
[observation construction](https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp),
[history ordering](https://raw.githubusercontent.com/fan-ziqi/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/observation_buffer/observation_buffer.cpp).

Inputs are body angular velocity (rad/s), body projected gravity, body velocity
commands `(vx, vy, yaw_rate)` (m/s, m/s, rad/s), relative joint angles (rad),
joint velocities (rad/s) and previous applied normalized actions. Joint binding
uses names and the source mapping. Each frame has 45 elements; HIM uses six
frames, newest first, zero-initialized at entry, and its estimator is included.
Feature order, scales, nominal pose, ELU layers and action scales remain specific
to each source package. Inference runs at 50 Hz. Our physics stays at 2 ms
instead of the source's 5 ms: this transfer difference must be reported.

Outputs are absolute joint angle targets, clipped to model joint limits. Exactly
one instance of existing `JointPID` writes torque. Use source PD gains with zero
integral gain inside the standalone learned-policy episode; cap torque by the
minimum of source, model and existing 24 Nm ceiling. Do not mix the legacy gait's
PID gains with policy actions or alter the legacy controller.

## Skill and validation contract fixed before rollouts

Candidate command options: `forward` (.5,0,0), `reverse` (-.3,0,0),
`sidestep_left` (0,.3,0), `sidestep_right` (0,-.3,0),
`turn_left` (0,0,.5), `turn_right` (0,0,-.5). Units are as above.
Each is a four-second command followed by zero command and measured recovery.

Entry requires supported flat ground, tilt below 12 degrees, base above .23 m,
linear speed below .04 m/s and angular speed below .2 rad/s. Check conservative
current and projected path clearance before entering. Continue clearance,
contact, posture and finite-state guards every physics step. No state reset is
permitted during execution or recovery. Interruptions command zero, then attempt
recovery; they still report failure. Timeout and ineligible entry report failure.

Acceptance: signed translation >=.25 m for forward/reverse or >=.20 m lateral;
signed yaw change >=.40 rad for turns; translation cross-axis drift <=.30 m,
turn translation drift <=.30 m. Recovery requires linear speed <.04 m/s,
angular speed <.2 rad/s, tilt <12 degrees and four foot contacts continuously for
.8 s. No contact with obstacles/non-foot floor, no falls, no torque excess.
Recovery has a six-second budget after the command; interruptions and shortened
timeouts do not count as skill successes. Failure to meet motion or recovery is
an unsuccessful skill even if the robot stays upright.

First run one development context (yaw 0, friction .8), then freeze code and
commands before evaluation on three different contexts (yaw .2/-.2/.4 rad,
friction .7/.9/.65). Context IDs are provenance, not hidden randomness. Record
every attempted option and failure. Registry availability follows measured
per-policy/per-option results; unvalidated/failed options remain explicit
experimental entries. Test initiation, masks, mapping, history, reset, invalid
inputs, interruption, timeout, cleanup and persistent physics. Run the complete
regression suite after integration. No physical deployment or automatic SNN
schema extension is part of this milestone.
