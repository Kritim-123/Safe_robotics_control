# V5 repeated jumps: validation

October 17, 2026, Windows project Python environment, MuJoCo simulation.

## Result

The separate V5 multi-obstacle extension completes **9/9 held-out courses**
containing two, three or four bars: **27/27 verified crossings**, stable goal
arrival, no obstacle contact or fall. An additional nominal three-bar demo
also completes, for 30 verified jumps across ten successful courses.
Two additional tighter-spacing courses are refused before any jump; these
remain failures to complete the goal, rather than being counted as successes.

| Assessment | Courses completed | Verified jumps | Contacts / falls |
| --- | --- | --- | --- |
| Seed 22701, varied 2/3/4 bars | 9/9 | 27/27 | 0 / 0 |
| Nominal 3 bars, 2.2 m spacing | 1/1 | 3/3 | 0 / 0 |
| 3 bars, 2.0 m spacing | 0/1, refused | 0/3 | 0 / 0 |
| 3 bars, 1.8 m spacing | 0/1, refused | 0/3 | 0 / 0 |

Randomized assessment uses spacing 2.15–2.55 m, heights 2–4 cm, friction
.8–.9, first-bar distance .8–1.15 m. Bars are 2 cm deep and 1.2 m wide.
The two closer courses use 2.5 cm bars. Their next bar enters the existing
1.2 m landing-corridor guard's conservative circular footprint; the runner
does not override this guard to obtain another success.

## What changed

The previous V5 runner was designed for one jump. New code in
`experiments/obstacle_v5_multi/` rearms target selection only after a verified
full crossing and stable landing. Each jump has separate entry, foot-clearance,
airtime and full-body crossing records. Fresh 100 ms neural packets and two
eligible learned jump proposals are required again for the next bar.

The repeated sequence is **walk/precision approach → brake → supported wait →
learned jump → verified stable landing → fresh approach**. After the last bar,
the original V3 readout selects walking and stopping at the fixed final goal.
Approach includes bounded lateral/yaw corrections to limit accumulated drift.
This correction is classical control; the jump association is neural.

Passed bars are removed from decision observations only after verified
crossing. All physical collision geometry stays active. The robot is placed
once, with no mid-episode reset, teleport or velocity injection. The original
launch eligibility, foot/body crossing checks and 24 Nm walking / 72 Nm
jumping bounds remain in force.

The viewer overlays current phase, cleared bars, neural proposal, simulated
time and actual goal distance. Each raw run saves events, observation/count
packets, state/torque traces and the terminal summary.

## Model and learning provenance

The refined V5 **264-cell encoder, normalization and existing association
rows are unchanged**. The V3 model and refined jump profile are unchanged.
A separate bank extends the retained associations with 108 measured packets
from 36 physical jumps in 12 seed-21701 collection courses. Collection
completed 12/12 goals without contacts/falls and explicitly marks teacher
proposals. Each jump's own measured outcome labels its actuation packet and
up to two preceding eligible packets. These are training results.

The separate frozen model checksum is
`a0780f10704a79c2b8ed6e871902e2caa9875224d1f09bab2e1c62e1240977d8`.
It retains all 336 inherited rows and adds 108 measured rows. No assessment
outcome is used to fit it. Final assessment and viewer use learned proposals,
no forced teacher and no explicit jump request. The input retains the actual
final-goal distance throughout, including early bars in a long course.

The unchanged refined V5 head with the new runner refuses the nominal first
bar: its prior bank does not cover the longer goal and visible next bar.
The initial uncorrected repeated motor diagnostic clears two bars but refuses
the third due to lateral drift. A corrected teacher diagnostic completes three
bars; it is retained as development evidence, separate from autonomous tests.
All four initial/development/control results are retained in the artifact bank.

## Checks and retained evidence

Five focused tests passed. They cover preserving the encoder/old bank,
three autonomous jumps with only one robot reset and continuous motion,
fresh eligible confirmations per jump, refusal of a later tall bar, refusal
when neural proposals are forced to wait, blocked landing, interruption and
closing the viewer. All these exit paths release torques.

The complete suite ran **216 tests** in 172.956 seconds: 214 passed and two
existing localhost HTTP checks were blocked by sandbox socket permissions
(WinError 10013). Rerunning those two checks with loopback access passed both.
All 216 tests are therefore validated across the suite and targeted rerun;
both original and rerun logs are retained rather than claiming a clean
single-run suite result.

Independent audit: **353 source/model hash checks**, **8,529 jump-head packet
replays**, **8,529 V3 packet replays**, **zero errors**. V3 replay reproduces
the reset of its goal commitment after each verified landing. The maximum
recorded base displacement per 2 ms step is **5.63 mm**, with finite state
and strictly increasing time. No episode used a teacher or jump request.

Retained model, collection records/outcomes, development failures, assessment
summaries/results, source freeze, audit and per-course skill transition times
are under `docs/agents/obstacle_v5_multi/`. Raw traces and logs remain under
`output/obstacle_v5_multi/`.

## Reproduce

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.run --viewer --viewer-speed .65
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.run --count 4 --spacing 2.4 --viewer
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.replay --output output/obstacle_v5_multi/replay-assessment
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.audit --batch output/obstacle_v5_multi/replay-assessment --artifacts output/obstacle_v5_multi/replay-assessment-artifacts --output output/obstacle_v5_multi/replay-audit.json
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.learn --output output/obstacle_v5_multi/retrain --model output/obstacle_v5_multi/retrained/primary.npz
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_multi.py -v
```

Use exclusive new output paths. `--video` additionally records an MP4 with
phase and count overlays. Original V4, V5 and refinement packages/artifacts
remain available for replay.

This assessment covers static aligned low bars on flat ground with privileged
MuJoCo sensing. Arbitrary obstacles, camera perception, tighter landing
corridors, rough terrain and physical robot deployment remain unvalidated.
