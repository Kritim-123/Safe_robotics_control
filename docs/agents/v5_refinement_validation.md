# V5 reliability refinement

Measured October 17, 2026 in MuJoCo. The four reported refusals and the
rear-thigh contact now all complete autonomous walk–brake–jump–goal episodes.
On **40 untouched paired courses**, refined V5 completed **40/40**, compared
with **31/40** for the unchanged original V5. It gained nine successes, lost
none, and recorded no contact, fall, refusal or approach overshoot in that
fresh batch. All **211 project tests passed**.

## Changes and contract

The opt-in revision lives in `experiments/obstacle_v5_refinement/`. Original
V3/V4/V5 sources/models and the original jump profile are preserved. The
classical walk-up/braking controller, supported-stance requirement, two fresh
eligible confirmations, full foot/body crossing criteria, stable recovery,
physical contact/height/tilt guards and torque cleanup remain unchanged.
The mission stays `progress`; no explicit jump request is supplied.

The contact occurred during landing after a valid, stable launch. Motor
development selected the following combined change, retaining the existing
72 Nm jump and 24 Nm walk/brake bounds and the same landing PID gains:

| Profile field | Original V5 | Refined V5 |
|---|---:|---:|
| Thrust duration | .106979 s | .096281 s |
| Forward lean | 1.116103 rad | 1.156103 rad |
| Absorption thigh | 1.03 rad | .95 rad |
| Absorption calf | −2.06 rad | −1.90 rad |

Other trajectory settings are unchanged. Shorter thrust, adjusted lean and a
shallower landing stance improve horizontal travel and rear-thigh clearance
together; this assessment does not isolate each parameter's contribution.
The selected profile passed a six-world development screen and eight hard
high-bar/low-friction confirmations, with minimum recovered rear-body margin
4.80 cm. Two confirmation worlds overlap the screen. These were motor
development runs with teacher proposals, not fresh autonomous test successes.

The jump head preserves all original **168 LIF cells** and adds **96 fine
level-sensitive cells**, for 264 total. The new cells resolve range, height,
depth, friction and current posture/motion more finely. Current base pitch
(rad), base height (m) and angular speed (rad/s) extend the old numerical
observation contract. There is no episode clock, scene ID, outcome label or
task jump bit in the features. Each 100 ms packet starts with zero membranes
and counts and uses 16 numerical LIF updates. Encoder weights are fixed;
the learned component is a nearest spike-count outcome association bank.
Conflicting nearest labels and unfamiliar patterns still abstain.

New outcome coverage and finer encoding are assessed as one head revision,
not as an isolated proof that the extra neurons alone caused improvement.
The unchanged V3 selector handles walking/goal stopping after crossing; the
low-level joint controller is still classical PID.

## Data and evaluation separation

The original 20 V5 test worlds are now **development/training regressions**,
including cases 02, 04, 06, 08 and 14. They are no longer fresh evidence.
They were combined with 60 additional worlds from seed 17601 for collection
using the selected profile. All 80 actual physics episodes crossed and reached
their goals without contact/fall. Teacher proposals are explicitly marked in
collection logs and are absent from autonomous evaluation and the live demo.

The actuation observation and up to two preceding eligible stable packets
share their episode's actual crossing outcome, with observation/actuation
times recorded separately. Those 240 packets plus 96 separately identified
eligibility-negative examples form 336 training rows. The negative examples
are not misrepresented as executed robot outcomes. All 336 training rows fit;
that is training fit, not the held-out reliability score.

| Cohort | Original V5 | Refined V5 | Refined contact / fall |
|---|---:|---:|---:|
| Original 20, now development regression | Historical 15/20 | 20/20 | 0 / 0 |
| Separate validation, seed 18601 | Not rerun | 16/16 | 0 / 0 |
| Untouched paired assessment, seed 19601 | 31/40 | 40/40 | 0 / 0 |

Each assessed arm also ran six unsuitable-target controls and a clear path.
Tall/moving/deep/misaligned targets, low friction and blocked landings caused
no jumps or contact; the clear path reached its goal without jumping.
The paired baseline had four refusals, four contacts and one incomplete
crossing. The refined arm gained nine successes and lost none. Identical
scenario dictionaries were independently checked for every pair.

The five requested repairs are retained in
`docs/agents/obstacle_v5_refinement/five_case_repairs.json`. Their success is
regression evidence after using those worlds for development. The separate
40-world assessment is the generalization check; neither model nor profile
was changed after the source/model freeze or after seeing that batch.

## Direct contact-case comparison

Original and refined case `v5-test-08` initiate jump preparation at exactly
16.1 simulated seconds, with **zero difference in entry qpos or qvel**.
The original run caught `RL_thigh` on the bar. The refined run fully crosses,
recovers with **4.89 cm rear-body margin**, resumes walking and stops **9.00 cm
from the goal** at 26.846 s. This rules out a moved obstacle or a different
launch state as the explanation for that repair. The combined motor-profile
change fixes this matched-state contact; the changed head addresses refusals.

The default finite live demo replays this formerly contacting world. At
half-speed pacing it walks first and begins jump preparation around 32 wall
seconds. Fresh airborne packets may decode wait while the committed jump
finishes safely; the phase and latest proposal are logged separately.

## Validation and retained failures

Three rounds retained 134 motor-development episodes, including failed
candidates: first round 44 runs (22 contacts, two falls), second round 23
adaptively screened runs (six contacts, one fall), third round 67 runs (one
contact, no falls). A further 32 confirmation runs compare landing profiles
on eight repeated hard worlds. Earlier profiles that fixed the original
contact still failed some harder cases and were not promoted. The selected
profile passed the final confirmation. All failed-candidate records remain
available; they are not mixed into the selected profile's outcome labels.

Seven new tests cover preserved encoder cells, finer count resolution and
reset packets, normalization/model tamper detection, conservative conflict
abstention, all five repairs, continuous plant motion with one initial reset,
head necessity, forced-proposal vetoes, preparation interruption and viewer
closure with torque release. The full suite passed **211/211** in 145.868 s.
A separate post-thrust interruption finished its committed landing, recovered
stably and released torques without contact/fall, returned `interrupted`, and
received no completion credit.

The fresh audit verified **177 frozen source/model files** and **169 original
V5 freeze files**, with zero mismatches. It exactly replayed 10,208 original
jump-head packets, 10,918 refined packets and 21,126 sequential V3 packets:
**42,252 exact neural predictions**, zero mismatches. It also verified all
94 fresh/control episodes had no teacher proposals or explicit jump requests
and released torques. Maximum actuator-limit ratio was 1.0. The regression
audit adds another 11,718 exact predictions with zero errors.

The retained head checksum is
`9bfa0220c45605caf9103a4dc63896b1bd5c81b522d487b1a86ef57ba3fab79c`.
The head, selected profile, training records, compact outcomes, source freeze,
audits and full test log are in `docs/agents/obstacle_v5_refinement/`. Raw
decisions, full physics traces and videos remain under
`output/obstacle_v5_refinement/`.

## Reproduce

From the repository root, with fresh output paths:

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --viewer --video --scene former-contact --viewer-speed .5
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --phase regression --output output/obstacle_v5_refinement/replay-regression
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --phase fresh --count 40 --output output/obstacle_v5_refinement/replay-fresh
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.audit --batch output/obstacle_v5_refinement/replay-fresh --output output/obstacle_v5_refinement/replay-audit.json
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.train --output output/obstacle_v5_refinement/retrain-trials --model output/obstacle_v5_refinement/retrained/head.npz
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_refinement.py -v
```

The full-suite command used project-local temporary files and loopback access
for the HTTP tests:

```powershell
.venv-win/Scripts/python.exe -c "import pathlib,tempfile,unittest; p=pathlib.Path('tmp/v5r-test-temp').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

## Scope

This remains a restricted simulation result: flat ground, static aligned +X
bars, starting distances .75–1.2 m, heights 2–4 cm, depths 1.2–2 cm, width
1.2 m, friction .8–.9 and the tested goal 1.26 m beyond the bar. Geometry,
friction and posture come from MuJoCo, not camera perception. Forty successes
do not establish universal reliability, hardware safety or SNN superiority
over classical selection. Multiple jumps, arbitrary headings/terrain and
uncertain real-world perception remain outside this assessment.
