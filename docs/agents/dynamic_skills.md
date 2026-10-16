# Dynamic classical skills: measured October 16, 2026

The new `skills/dynamic.py` implements a faster trot (`run`) and a genuine
in-place airborne `jump`, alongside `wait`, using the existing torque PID.
These are hand-designed classical primitives, not trained locomotion policies.
The separate three-action spiking-selector dataset must distinguish an explicit
jump request from obstacle avoidance: this jump has **not** crossed an obstacle.

Measured development results on flat MuJoCo terrain:

| Measurement | Result |
|---|---:|
| Original walk forward distance, same 4 s active interval | .20685 m |
| Faster trot forward distance, same 4 s active interval | .60820 m |
| Original walk / faster trot mean forward speed | .05171 / .15205 m/s |
| Base jump continuous all-four-feet airborne duration | .124 s |
| Settled / peak jump base height | .26463 / .34281 m |
| Jump horizontal landing displacement | .06781 m, predominantly backward |
| Jump peak torque / maximum limit ratio | 24 Nm / 1.0 |

The original walk comparison uses `WalkingExperiment.run(duration=4)` and the
same generated arena, model, friction, initialization and 2 s standing period.
Distances are measured from the closest logged samples at simulation t=2 and
t=6. Stopping distance after the four-second interval is excluded. The faster
trotting episode moves .82742 m including deceleration and recovery. Its feet
never all leave the floor together; **aerial running is not demonstrated**.
The command .5 m/s is not the measured robot speed.

Three development configurations (seeds 0/1/2 recorded as provenance; yaw
0/.2/-.2 rad and friction .8/.7/.9) completed both run and jump without obstacle
contact or falling. These are small development checks, not a population claim.
Seed does not introduce hidden randomness: changing the supplied scenario is
what varies physical conditions. Profiles were frozen before the integrator's
training/validation/held-out data collection.

## Contracts

`execute_dynamic_skill(name, scenario, ...)` returns `(history, summary)`.
`name` is `wait`, `run`, or `jump`. With `name=None`, a
`selector(observation, eligibility)` callback chooses once after two seconds
of standing, and the same simulation continues. It returns one of those names
or `None` to refuse selection. Refusal returns failure reason `no eligible
skill`; unexpected callback exceptions propagate after torque cleanup.

`decision_observation` contains actual settled time (seconds), XY position
(metres), yaw (radians), planar speed (m/s), base height (metres), upright
rotation coefficient and all actual obstacle centres/radii/velocities at that
time. No future scripted obstacle states or trial outcome enter this record.
Eligibility is a dictionary with the same three identifiers. A forced
ineligible action is retained as `failed initiation`, never executed or silently
relabelled successful. The user/integrator chooses the task objective separately.

| Identifier | Initiation | Execution | Completion |
|---|---|---|---|
| wait | Stable posture, current footprint gap >.08 m | Zero velocity through existing gait/PID | Requested duration plus .8 s stable posture |
| run | Stable posture; full heading-aligned command-distance segment has >.12 m gap | Body-forward .5 m/s command for 4 s, original gait/IK/slew | Stop gait then .8 s stable posture |
| jump | Stable posture, current footprint gap >.35 m | .5 s smooth crouch, .12 s extension, .06 s retraction to home | Requested duration, verified >=.04 s flight and .8 s stable landing |

Stable initiation means base >.23 m, linear speed <.04 m/s, angular speed
<.2 rad/s and tilt <12 degrees. The default four-second duration is configurable
but must be >=1.5 s. Default total episode timeout is 12 s. Run speed can be
reduced but cannot exceed .5 m/s. Commands follow initial body heading; the
primitive is not a goal planner. Jump parameters are fixed: crouch joint
hip/thigh/calf angles `(0,1.1,-2.2)` rad, extension `(0,.45,-.9)` rad, then
original home `(0,.9,-1.8)` rad for each leg. No new actuator model or gains
are installed. All joint targets remain within model limits in tested runs.

`DynamicSkill` supplies joint targets for persistent execution without resetting
physics. Only the episode wrapper initializes pose, once. The existing 500 Hz
PID, 100 Hz targets, 24 Nm/model-lower limits and anti-windup remain unchanged.
No body force, qvel injection, mid-episode placement, or teleportation is used.
Flight is evaluated every 2 ms: no foot contact plus every foot site >.025 m
above the floor. Histories are sampled at 50 Hz with the terminal sample added
even on early failure. `summary.longest_airtime` uses physics cadence, not
the subsampled trace.

Interruptibility means the target source can switch to standing; it does not
imply instantaneous stopping or cancellation of momentum. Tests interrupt run
and jump during early execution and jump at .7 s while airborne. They report
`success=False`, reason `interrupted`, and verify stable recovery. Recovery
requires gait activity zero, linear speed <.04 m/s, angular speed <.2 rad/s,
tilt <12 degrees, all four feet contacting, for .8 s. Clearance <.035 m,
obstacle contact, height <.15 m, tilt >35 degrees, nonfinite state, failure to
establish flight, and timeout are unsuccessful outcomes. Dynamic obstacles can
invalidate initiation; the current-clearance guard remains active. Stopping
simulation at failure is not an active hardware safety strategy.

## Evidence and reproducibility

Run from the repository root:

```powershell
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p test_dynamic_skills.py -v
```

Final targeted result: **8 tests passed in 5.038 s**. Tests cover three real
physical jump/landing configurations, same-scene original walk comparison,
failed initiation, standing near a blocker, timeout, interruption, monotonic
physical position changes, torque bounds, settled causal selector observations,
selection refusal and exception torque cleanup. The parent owns broader suite
and integrated selector evidence.

Raw results are under `output/dynamic_skills/`, including complete traces,
`original_walk.json`, `development_tuning.json`, `failed_initial_jump.json`, and
`midflight_interruption.json`. Compact tracked results and source fingerprint
are in `docs/agents/dynamic_metrics.json`.

The first jump attempt failed: a deeper crouch and slower retraction reached
.39015 m base height but never achieved verified all-feet flight, then hit the
35-degree tilt limit at 2.914 s. Its evidence is retained. An 18-profile bounded
development sweep shortened thrust/retraction and reduced crouch depth; every
candidate in that revised sweep completed. The chosen moderate profile avoids
claiming the highest jump as the objective. All tuning outcomes remain recorded.

This work demonstrates in-place jumping and faster trotting on simple simulated
flat terrain. It does not demonstrate obstacle jumping, aerial running, learned
low-level control, hardware validity, or reliable recovery from arbitrary
mid-thrust interruption. No policy weights or licenses were imported.
