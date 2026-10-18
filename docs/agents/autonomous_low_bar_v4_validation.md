# V4 autonomous low-bar selection

Measured October 17, 2026 in this Windows MuJoCo environment. V4 selects the
low-bar jump from observed geometry and robot state without a jump request.
The nominal persistent episode clears a 4 cm bar, resumes V3 goal tracking and
stops 9.17 cm from its goal. No intermediate robot reset, external lift or
velocity injection is used. Frozen held-out assessment: **18/20** full
crossings and goal stops, **0 contacts, 0 falls**, and **2 learned refusals**.

## Architecture and provenance

The retained V3 model is unchanged. A new, independently seeded 168-cell LIF
head encodes 13 observed quantities: obstacle presence, forward/lateral
position, height, depth, span, measured target speed/alignment, landing
corridor, robot stability/speed, goal distance and friction. No task jump bit,
scene name, episode time or future schedule enters it. Opposed level-sensitive
cells and seeded mixed cells produce 0–16 spikes per neuron. Membranes and
counts start at zero for each 100 ms packet. This is numerical LIF encoding,
not a claim about biological timing. All observations are privileged simulator
measurements/geometry, not camera pixels or estimated depth.

Training seed 17401 generated the nominal trial and 31 varied low bars using
the existing frozen motor profile: 30 full crossings and 2 contact failures.
Their actual outcomes label the recorded observations. Another 80 compound
eligibility-negative examples are explicitly marked as such, rather than
misrepresented as measured robot outcomes. A nearest spike-count association
bank learns these labels; encoder weights and motor parameters are not learned.
All 112 training rows are classified correctly, which is training fit only.

Head semantic checksum:
`5b3c8c7c6c21ad4433b63cf0ca20aa79b720ec504773c665d93313a09486e980`.
Inherited V3 semantic checksum:
`b2dcc0e47c62a98896bed3e9108f5534988594fc806535c98cf5baa216d01841`.
The source/model freeze contains 162 hashes, checked with zero mismatches.
Training records and the saved head are in `docs/agents/obstacle_v4/`.

## Execution contract

The single mission is always `progress`; no request stream supplies jumps.
Two fresh matching neural proposals and supported/stopped posture initiate
`low_bar_jump`. Eligibility may veto proposals but cannot initiate one. It
checks the existing static aligned +X entry envelope, actual feet behind the
bar, friction and a landing corridor with constant-velocity predictions of
other obstacles over three seconds. This is a finite approximation.

The existing 72 Nm low-bar trajectory executes under one PID writer. During
flight the maneuver remains committed even if new observations decode wait.
All physical target collisions still abort. Finite-state, posture/height,
other-hazard clearance, interruption and torque-release checks remain active.
Full crossing requires every foot sphere to overlap and clear the bar, all feet
and collision primitives to end beyond its far face, lateral containment,
shared flight, and .8 seconds of stable four-foot support. Only then is the
crossed target removed from forward planning. Its physical geometry stays
active. A checked handoff resumes walking, and the unchanged V3 readout handles
subsequent walking/goal stopping. Walking restores its 24 Nm bound and gains.

Nominal decisions: settle until 2 s, two observation proposals select takeoff
preparation at 2.1 s, verified crossing/landing at 4.904 s, walking, braking
at 11.1 s, supported goal stop by 12.712 s. Flight lasts .484 s. At verified
landing, the rearmost collision primitive has 8.10 cm horizontal margin; foot
clearances during bar overlap exceed 22.4 cm. Peak torque is 72 Nm and the
maximum actuator-limit ratio is 1.0. These are simulation settings, not a
physical Go2 motor rating or hardware safety certification.

## Frozen evaluations and retained failures

| Cohort | Cases | Result |
|---|---:|---|
| Development motor outcome data | 32 | 30 crossings, 2 contacts; used for fitting |
| Validation low bars, seed 18401 | 6 | 6 crossings and goal stops; no contact/fall |
| Held-out low bars, seed 19401 | 20 | 18 crossings and goal stops, 2 refusals; no contact/fall |
| Held-out unsuitable targets | 6 | No jump and no contact/fall in all six |
| Held-out clear path | 1 | Goal reached, no jump |

The unsuitable controls are tall, moving, blocked landing, low friction,
distant and misaligned bars. Their learned heads made zero jump proposals;
these results therefore are not just guard vetoes. Separate tests force a
jump proposal against tall/moving/blocked scenes and verify that eligibility
still vetoes it. Forcing the head to wait on the nominal eligible bar prevents
takeoff, demonstrating that geometry guards do not secretly trigger the jump.

Cases `v4-test-09` and `v4-test-16` are eligible low bars but their count vectors
are equally close to training rows 7 (success) and 23 (failure). The decoder
abstains instead of breaking the tie. These cases are failures to complete the
mission; their unexecuted jumps cannot be credited as successful or proven
unsafe. More outcome coverage and a more discriminating observation/count
representation are follow-up work; the held-out cases were not added to the
training bank.

The original validation misalignment control used a rotated wide bar at the
near launch distance. It contacted the robot during settling before any jump
decision. That failure is retained under `output/obstacle_v4/validation-v1/`.
The held-out misalignment control moves the rotated bar to .55 m to test a
valid-start refusal. This is a control-scene correction, not a motor fix; the
validation contact remains a recorded failure.

## Verification and reproduction

The five new tests cover exact model serialization/tamper checks and fresh
count resets, an actual full crossing/goal episode with one plant reset,
neural-decision necessity, forced-proposal guard vetoes, preparation
interruption/recovery and viewer-close torque cleanup. An independent replay
recomputed **2,931 jump-head packets and 2,931 sequential V3 packets** from the
held-out observations: zero mismatches. Compact results, the model freeze and
audit are retained next to the model; full traces and movies remain under
`output/obstacle_v4/`.

The complete project suite passed **199/199 tests** in 103.734 seconds using
the existing `.venv-win` environment, a project-local temporary directory and
loopback access for the HTTP tests. Its exact runner was:

```powershell
.venv-win/Scripts/python.exe -c "import pathlib,tempfile,unittest; p=pathlib.Path('tmp/v4-test-temp').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

The retained full test log is `docs/agents/obstacle_v4/unit_test_log.txt`.

Commands from the repository root:

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.train --output output/obstacle_v4/training-v1
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.run --phase evaluate --split validation --count 6 --output output/obstacle_v4/validation-v1
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.run --phase evaluate --split test --count 20 --output output/obstacle_v4/test-v1
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v4.py -v
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.run --viewer --video --viewer-speed .5
```

Existing artifact destinations are exclusive: choose new paths for reruns.
To reproduce training without replacing the retained model, pass a fresh
`--model` path as documented in the experiment README.

## Limits

This first version starts already in the assessed launch stance: static
aligned low bars 23–24 cm ahead, 2–4 cm high, 1.2–2 cm deep, width at least
1.2 m, flat ground and friction .8–.9. It does not approach a distant bar,
estimate geometry from cameras, chain multiple jumps, handle general headings
or solve arbitrary terrain. The low-bar action exists only in this opt-in
V4 executor; the historical global SNN skill schemas are unchanged. No claim
of SNN superiority follows from this assessment.
