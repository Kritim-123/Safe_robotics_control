# V5 close-spacing repair

October 17, 2026, project Windows Python environment, MuJoCo simulation.

## Repair and mechanism

Both formerly refused three-bar courses now complete autonomously: **1.8 m
and 2.0 m center spacing**, with three verified jumps and stable goal arrival
in each. No contact, fall, teacher proposal, explicit jump request or
intermediate robot reset is used.

Frozen assessment completed **23/23 traversable courses and 69/69 jumps**:
the two repaired worlds, twelve new close worlds and nine wider regressions.
All three unsuitable controls refused the affected jump. Across all 26
assessment episodes there were **zero contacts and zero falls**.

| Group | Goals | Verified jumps | Interpretation |
| --- | --- | --- | --- |
| Former 1.8/2.0 m refusals | 2/2 | 6/6 | Development regressions |
| Fresh seed 23701 close courses | 12/12 | 36/36 | Frozen assessment |
| Previous nine wider courses | 9/9 | 27/27 | Preserved successes |
| Real landing blocker | 0/1, refused | 0 | Required veto |
| 1.5 m bar spacing | 0/1, refused | 0 | Required veto |
| Tall second bar | 0/1, refused | 1 | First bar cleared, second refused |

The failure was the landing-corridor approximation. A thin bar has half
extents .01 m along travel and .60 m across travel. Its circumscribed circle
has radius .60008 m, so the old guard treated its transverse width as a
forward obstruction. The SNN already proposed a jump, but the guard vetoed it.

The separate `experiments/obstacle_v5_close/` package checks the actual
oriented box using the existing point-to-box distance, retaining the **.42 m
robot disc, 1.2 m corridor, 3 s horizon and .15 m threshold**. It subtracts
half the 3.75 cm spatial sample interval (1.875 cm) and maximum obstacle travel
over half the temporal sample interval. These distance-function bounds cover
the gaps between samples. Static scenes need one temporal pass; moving
obstacles retain seventeen samples and a speed bound. Cylinders, rotated
other boxes and arena boundaries remain checked.

| Former refusal | Old corridor gap at first launch | New lower bound | Course result |
| --- | --- | --- | --- |
| 1.8 m spacing | −.166 m | .406 m | 3 jumps, goal at 67.114 s |
| 2.0 m spacing | .034 m | .606 m | 3 jumps, goal at 71.646 s |

The robot state and torque traces before the first jump match the old runs
**exactly, maximum difference zero**. At the new launch time both old and new
SNN outputs propose `low_bar_jump`; the old eligibility reason is `landing
corridor blocked`. The corrected observation changes twenty-one spike-count
cells in those packets, while preserving the learned jump proposal.

## Preserved control contract

The retained V5 multi-obstacle model checksum is unchanged:
`a0780f10704a79c2b8ed6e871902e2caa9875224d1f09bab2e1c62e1240977d8`.
All 444 association rows, weights, biases and normalization remain unchanged.
The refined jump profile and V3 goal selector remain unchanged. No training
or teacher data collection was needed for this repair.

The `landing_clearance` input now uses the same shape-aware calculation as
eligibility. Other feature definitions and units are unchanged, including the
actual final-goal distance. Existing multi-obstacle source/model bytes are
retained and independently hashed before development.

Each jump still needs two fresh learned proposals and the original supported
launch conditions. It must clear all four feet and all collision geometry,
then achieve a stable landing before rearming. Physical collisions remain
enabled throughout, including passed bars. Contact, falls, overshoot, refusal
or timeout abort the episode; interruption completes its defined settling
behavior and releases torques. Walking remains bounded at 24 Nm and jumping
at 72 Nm. The more conservative circular runtime emergency-clearance check
also remains unchanged; this revision corrects the pre-jump corridor and its
matching observation.

The 1.5 m course remains outside the permitted landing corridor. This repair
therefore does not imply that any spacing is traversable. The tall-bar control
demonstrates that a successful earlier jump does not force a later jump.

## Validation design

The two repaired worlds are development regressions, not new held-out cases.
Sources and the unchanged model are frozen before twelve new seed-23701
courses: two/three/four bars, spacing 1.7–2.1 m, heights 2–4 cm, friction
.8–.9, first distance .8–1.15 m. All nine previous wider seed-22701 worlds
are separately rerun as regressions. Additional controls contain a real
landing blocker, 1.5 m bar spacing and a tall second bar.

Five focused tests passed. They check the analytic thin-bar clearance,
sampling margin, cylinder/rotated-box/moving-blocker/boundary guards, both
repaired courses with exact neural confirmation, no jump under forced neural
wait, a real blocker even under a forced jump proposal, one robot reset,
continuous state, interruption and viewer closure. Unsafe controls are
allowed to refuse; refusal is not counted as goal success.

The full project suite passed **221 tests in 204.172 seconds** using the
existing Windows environment, with localhost access for the HTTP tests:

```powershell
.venv-win/Scripts/python.exe -c "import pathlib,tempfile,unittest; p=pathlib.Path('tmp/v5close-test-temp').resolve(); p.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); raise SystemExit(not r.wasSuccessful())"
```

Independent audit passed **362 source/model checks**, **17,624 jump-head
packet replays** and **17,624 V3 packet replays**, with zero errors. The eight
parent multi-obstacle source/model hashes also remain unchanged. All jump
entries have two eligible learned confirmations. Traces are finite and
continuous, with maximum base displacement **5.633 mm per 2 ms step** and
strictly increasing time. The unchanged source freeze covers the new runtime,
geometry, assessment script, test file, parent package and retained models.

Raw development and assessment traces, events and decisions are under
`output/obstacle_v5_close/`. Compact summaries, frozen hashes and audit data
are retained under `docs/agents/obstacle_v5_close/`.

The complete 26-episode results, neural replay totals, event timelines and
source audit are retained in its `assessment-v1/` subdirectory. The development
outcomes and exact state-matched diagnosis are retained alongside them. The
live demo was started with the command below and output directory
`output/obstacle_v5_close/live-18-v1` at half speed.

## Reproduce

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.run --viewer --viewer-speed .5
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.run --spacing 2 --viewer
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.evaluate --output output/obstacle_v5_close/replay-assessment
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.audit --batch output/obstacle_v5_close/replay-assessment --artifacts output/obstacle_v5_close/replay-assessment/artifacts --output output/obstacle_v5_close/replay-audit.json
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_close.py -v
```

Use new output folders. Add `--video` to retain an MP4 with skill and neural
count overlays. The normal viewer overlays time, phase, cleared bars and
distance to goal. The default is the repaired 1.8 m course.

This remains flat-ground, static aligned low-bar simulation using privileged
MuJoCo geometry and state, without camera perception. The result does not
establish reliable jumps at arbitrary spacing, rough terrain or hardware.
