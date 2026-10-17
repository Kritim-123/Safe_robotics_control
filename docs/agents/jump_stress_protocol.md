# Frozen jump stress protocol

One agent owns `experiments/jump_stress.py`, its evidence and this report. The
existing jump profile, PID, selection eligibility, guards and SNN remain fixed.
This is direct execution assessment of the motor primitive, not SNN training.

Declare 120 deterministic cases before execution (seed 20261026): ten each of
flat headings/starts, friction .10–1.4, uphill/downhill/cross-slopes 1–15 degrees,
uneven front-foot support 5–50 mm, 5–50 mm steps up/down with edges .30/.40 m
ahead, static blockers .8–1.7 m ahead, approaching blockers .05–.50 m/s, and
interruptions during preparation or after thrust starts. Freeze the saved
`smooth_forward_jump/config.json` and record source hashes and exact scenes.
No physical pushes are applied; the dynamic runner does not implement them.

Start from home once, settle two seconds, check current eligibility, execute
with 500 Hz physics and 100 Hz joint targets, enforce normal guards and timeout.
Meters, seconds, radians, newton meters and newtons are used. Slopes/supports
affect the starting posture through actual contacts, without manually tilting
the body. High supports may cause setup contact/instability and must be reported
as such, not treated as launched jumps. A continuous plane remains underneath
all patches; there are no gaps or holes in this assessment.

Record refused initiation separately from executed failures. Retain existing
forward qualification: completion, stable landing, no collision/fall, at least
.06 s continuous verified flight, at least .10 m heading-relative total forward
travel, and positive flight travel. Completion alone requires .04 s verified
flight. Qualification does not establish step clearance: total travel can
include ground sliding, and the robot may stop before the step. Record final
position and actual collisions, never describe this as validated obstacle jumping.

Interruption cases pass their recovery task only if they report interrupted,
settle stably and have no collision/fall; they are excluded from ordinary hop
success denominators. A refusal says the guard worked on its current-state
criterion, not that the robot learned terrain affordances. Moving obstacles
test monitoring as well as takeoff; clearance stops remain failed execution,
not successful recovery. Save every result and raw 500 Hz trace. Contact force
is summed foot-contact normal force, not a hardware motor load specification.

Run once without tuning, aggregate by family, retain failures, then optionally
replay representative successes and failures with finite videos. No comparisons
to historical results are presented as fresh paired tests. Reproduction:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.jump_stress --output output/smooth_jump/stress-v1
```
