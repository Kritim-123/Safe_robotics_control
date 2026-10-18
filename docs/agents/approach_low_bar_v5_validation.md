# V5 persistent walk-up and observation-selected jump

Measured October 17, 2026. The new opt-in V5 episode now performs
**walk toward the bar → brake → supported stand → SNN-selected jump →
verified landing → walk to goal → supported stop**, with no intermediate
robot reset and no explicit jump request. The nominal course completes in
25.114 simulated seconds. Frozen held-out results are **15/20 complete
courses**, four learned refusals and one jump contact; no falls or approach
overshoots. These remaining failures are not counted as successes.

## What changed

All new controller/model code is under `experiments/obstacle_v5/`. The V3 goal
selector, V4 sources/model and original low-bar motor profile remain unchanged.
The approach uses a separate flat-ground gait setting: .06 m foot lift,
.18 m/s far command and .12 m/s inside .55 m. A classical 100 Hz range
controller initiates braking at .255 m. This is a measured-distance decision,
not a learned SNN stopping policy or a scheduled jump time.

Simply continuing the gait while decelerating allowed another front-foot
swing to hit the bar. The retained braking controller instead interpolates
the current joint targets to the standing pose with a .65 s quintic segment,
while slewing the command/activity to zero. It then requires >=.4 s of four-foot
supported stability and two fresh eligible neural jump proposals. The
observed jump entry must meet the existing speed/posture checks, all feet must
be behind the bar, and the landing corridor must be clear. The new current-pose
development gate uses .23–.30 m forward range and .015 m lateral alignment;
these are candidate bounds, not a guarantee throughout that continuous region.
Actual held-out jump-entry ranges were .2456–.2614 m.

The same frozen 168-cell LIF encoder processes the existing 13 V4 simulator
geometry/state observations. A new nearest spike-count association bank is
trained on actual post-walk jump outcomes. There is no task jump bit, elapsed
time, scene ID or outcome in the input features. Membranes/counts reset every
100 ms packet; each packet uses 16 numerical LIF updates. A committed jump
finishes its trajectory even if fresh airborne packets decode wait.

Once all foot spheres and collision geometry have crossed, with clearance,
shared flight, lateral containment and .8 s stable recovery, the controller
resumes walking and restores V3's goal/gait settings. The passed bar stays in
physical collision detection. Goal completion requires <=.12 m position error
and .8 s of stable supported stopping. One persistent plant and one PID own
all phases. Bounds remain 24 Nm walking/braking and the existing 72 Nm
simulation low-bar jump setting, applied to PID and actuator controls/forces.
Contacts, finite-state/height/tilt guards and torque cleanup remain active.

## Training and frozen assessment

Training, validation and held-out geometry seeds are 17501, 18501 and 19501.
The nominal demonstration is included in training and is not independent test
evidence. Training used teacher-forced *proposals*, still subject to entry and
physical guards, to collect actual full walking/jump outcomes. Runtime and
assessment use the learned head without teacher proposals or jump requests.

The retained 32 training worlds produced 28 full crossings/goal stops, three
jump contacts and one incomplete crossing. All 32 launch observations receive
their actual crossing outcome label. Another 80 eligibility-negative examples
are explicitly labelled as such, not asserted physics outcomes. The learned
bank fits all 112 training rows; that is training fit, not a reliability score.
Encoder weights/biases remain byte-identical to V4. The retained head semantic
checksum is
`e24988a3d342e993ab869923f7d7c2cff4c9d6a9d2e1bbc04422468443cd100c`.

| Cohort | Full crossing and goal stop | Other outcomes |
|---|---:|---|
| Retained teacher-collected training | 28/32 | 3 contacts, 1 incomplete crossing |
| Final validation | 7/8 | 1 learned refusal; no contact/fall |
| Frozen held-out low-bar worlds | 15/20 | 4 learned refusals, 1 jump contact; no fall |
| Unsuitable-target controls | 0/6 jumps | All refused, no contact/fall |
| Clear-path control | 1/1 goal | No jump |

The six unsuitable controls are tall, moving, blocked landing, low friction,
deep and misaligned targets. Tests also force a jump proposal in unsuitable
scenes and verify that eligibility vetoes it. Forcing wait on the eligible
nominal course lets the robot approach and stop but prevents takeoff, proving
the range controller does not secretly initiate the jump.

Among the 16 held-out jumps that were initiated, the robot first walked
.541–.898 m. Entry speed was at most .002602 m/s, all entries had four supported
feet and at least .508 s stable hold. Jump preparation began between 13.3 and
17.5 simulated seconds as obstacle distance changed; it was not a fixed timer.
All 27 held-out/control episodes had zero explicit jump requests and zero
teacher-forced proposals. All released torques; maximum actuator-limit ratio
was 1.0.

## Failures and development history

Held-out cases `v5-test-02`, `04`, `06` and `14` were refused by the learned
head. Their unexecuted jumps are not known to be either safe or successful.
`v5-test-08` caught the rear-left thigh on the bar during the jump, after a
successful approach/stand. Replaying its saved terminal pose identifies
`RL_thigh` against `obstacle_geom_0`. It had friction .80676 and a 3.968 cm bar,
with .25346 m launch range and .000585 m/s entry speed. This is a remaining
motor/selection failure, not an approach overshoot, and not fixed by this work.
More landing margin and better outcome discrimination remain necessary.

Earlier development candidates and their failures remain under
`output/obstacle_v5/validation-v1` through `validation-v7` and `training-v1`.
Fixed stopping distances exposed front-foot braking contacts, early stopping
exposed incomplete rear clearance, and a rejected speed-compensated candidate
could restart its gait repeatedly. On the same 32 training worlds, the earlier
gait-cycling collection completed 20 courses with ten contacts and one fall;
the revised approach/braking collection completed 28 with three contacts and
no falls. Multiple approach settings changed, so this is development evidence
for the combined revision, not an isolated causal ablation or held-out result.
`docs/agents/obstacle_v5/primary.npz` is that earlier, unpromoted bank. Runtime
explicitly loads `walk_up_v2.npz`. No held-out failures were added to training.

## Evidence and reproduction

All **204 project tests passed** in 114.350 s, including five V5 tests for the
full continuous walk/brake/jump/goal episode, one initial plant reset and
continuous positions, actual supported launch state, unchanged encoder/new
associations, learned-head necessity, forced-proposal vetoes and viewer closure
during walking with torque release. The test command used the existing
environment, project-local temporary files and loopback access for HTTP tests:

```powershell
.venv-win/Scripts/python.exe -c "import pathlib,tempfile,unittest; p=pathlib.Path('tmp/v5-test-temp').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

The frozen source/model manifest has **169 checks**, with zero mismatches.
All **162 inherited V4 freeze checks** also remain unchanged. Replaying the
held-out observations recomputes **5,500 V5 head packets and 5,500 sequential
V3 packets**, exactly matching every saved count vector/proposal with zero
errors. Compact summaries, result rows, test log, training records, source
freeze and audit are under `docs/agents/obstacle_v5/`; full traces and movies
remain in `output/obstacle_v5/`.

Nominal replay: 64.6 cm of walking, .2538 m launch range, four supported feet,
.508 s stable hold, .484 s flight, 7.44 cm full-body horizontal crossing margin
at recovered landing and 9.40 cm goal error. It starts walking at 2 s, brakes
at 13.34 s, selects jump preparation at 14.5 s, verifies crossing at 17.302 s
and completes supported goal stopping at 25.114 s. At half-speed viewer pacing
the jump preparation begins around 29 wall-clock seconds.

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.run --viewer --video --viewer-speed .5
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.run --phase evaluate --split test --count 20 --output output/obstacle_v5/replay-test
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5.py -v
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.train --output output/obstacle_v5/retrain-trials --model output/obstacle_v5/retrained/head.npz
```

Use fresh artifact paths for reruns. Training writes its own bank and measured
records; it never updates the retained model in place.

## Scope

This is a flat-ground, static aligned +X low-bar experiment, with assessed
starting distances .75–1.2 m, heights 2–4 cm, depths 1.2–2 cm, bar width 1.2 m,
friction .8–.9 and the tested goal 1.26 m beyond the bar. Its measurements come
from MuJoCo geometry/state, not robot cameras. It does not establish arbitrary
heading/terrain/moving-obstacle operation, repeated jumps, hardware safety or
SNN superiority. The action remains local to this opt-in version rather than
changing historical global skill schemas.
