# Count-pattern associations and live execution

October 16, 2026. One agent implemented and validated this milestone under
the [predeclared protocol](pattern_association_protocol.md).

## What now runs

Actual MuJoCo observations feed the archived seed-42 random LIF encoder. Its
16 spike counts feed a separate, learned association bank with 40 examples.
The bank proposes `dynamic/wait`, `dynamic/run`, or `dynamic/jump`; existing
eligibility checks choose that skill, an eligible wait fallback, or abstention.
The existing bounded joint PID executes the selected primitive. SNN weights
and biases remain frozen. This is association learning, not SNN weight training.

Exact count matches must agree on a label. For novel vectors, up to three
neighbors within Euclidean count distance 4 must agree, including ties at the
cutoff. Silent, unknown and conflicting patterns abstain. Scenario identifiers,
skill labels and outcomes never enter the decoder input.

The live demo alternates a learned forward run and requested in-place jump.
Each episode settles for two seconds, selects once, executes for four seconds,
and waits for stable recovery. Physics stays continuous within the episode.
Only the next episode resets the scene. Closing the viewer stops the loop.
Current diagnostic patterns do not trigger mid-skill switching or online updates.

## Actual evidence

Collection ran matched wait/run/jump trials in 40 training contexts and 10
validation contexts: **150 attempts**, including **36 rejected or failed attempts**.
There were no recorded obstacle contacts or falls. One candidate run in
`approaching-01` stopped at the clearance guard; rejected starts remain in the
attempt log. Training labels: 6 run, 20 jump, 14 local wait fallbacks. A wait
label does not mean the robot has solved forward navigation in that context.

Validation used the frozen bank, with no parameter updates: 8/10 agreement
with outcome-derived labels after supervision. Only 5/10 had a neural-pattern
proposal; fallback wait contributes to agreement and must not be mistaken for
proposal coverage.

The separate 50-context execution evaluation produced:

| Measure | Result |
| --- | ---: |
| Requested task fulfilled | 26/50 |
| Skill completed with stable recovery | 49/50 |
| Requested in-place jumps fulfilled | 24/30 |
| Forward-progress tasks fulfilled | 2/20 |
| Executed jump / run / wait | 25 / 2 / 23 |
| Unknown / conflicting patterns | 10 / 3 |
| Fallback selections | 13 |
| Obstacle contacts / falls | 0 / 0 |
| Maximum torque / limit ratio | 24 N m / 1.0 |

The one incomplete episode was `approaching-05`: a jump stopped when clearance
reached 0.034599 m, below the existing 0.035 m guard. It is a task failure.
The remaining task failures were completed waits. The 37 proposals all used
neighbor agreement; none of the held-out count vectors exactly matched a
training vector. Compact per-context decisions and outcomes are retained in
[held_out_results.json](pattern_association_v1/held_out_results.json).

The known-association recorded demo completed both tasks without fallback:
`clear-00` moved 0.810453 m along its selection heading; `clear-01` had 0.122 s
of verified flight and stable landing. Its videos contain rendered simulation
frames with actual skill, episode time, phase and current spike counts.

Encoder checksum:
`b72b9750bffd70ee37bd0aaedb5064b58157d8f5a13f58b5cd356ca281fa70ec`.
Association checksum:
`af09caff72026bfca7e8ff7b5a20e7011954c9e12cd5e56033b7ca07e3aeed00`.
The portable bank, training labels, original manifest, validation decisions,
and held-out summary are in `docs/agents/pattern_association_v1/`.

## Recording and reproducibility

Run from the project root with the existing environment:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.learn_associations
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --evaluate --output output/spike_pattern_skills/association-evaluation-v2
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --video --output output/spike_pattern_skills/association-recorded-demo-v1
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --viewer --repeat --output output/spike_pattern_skills/my-live-demo
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Choose new output names on reruns; nonempty experiment/episode directories
are refused. Collection output is `output/spike_pattern_skills/associations-v1/`.
It includes all attempted summaries, trajectories, JSONL labels/attempts,
encoder archive and SHA-256 manifest. The reusable bank is copied byte-for-byte
from that collection. No validation or held-out labels update it.

Every executing episode flushes `events.jsonl` and `telemetry.jsonl` live.
Events include start/reset policy, full decision before actuation, and final
outcome or error. Telemetry samples at approximately 10 Hz and includes named
physical inputs, spike IDs/counts/raster/membranes, pose, velocity and actuator
commands. `decision.json` preserves the selection pattern and nearest training
references. `trajectory.csv` retains the controller's approximately 50 Hz trace
and terminal state; `result.json` records completion, task fulfillment, guards,
eligibility, fallback, torque, clearance, flight and stable recovery.

The first held-out runner invocation exposed a return-value unpacking error
after one simulated episode. It retained partial telemetry and an error event
under `association-evaluation-v1/`. The error was corrected before the complete
50-context `v2` evaluation; the aborted run is excluded from those metrics.
Initial sandbox test attempts also encountered temporary-directory permission
errors; tests were rerun with normal filesystem access.

Eight new tests cover counts distinguishing identical active-ID sets, exact
conflicts, neighbor ties, silence/unknown patterns, eligibility and abstention,
artifact integrity, heading-relative progress, actual learned run/jump execution,
logging, overwrite refusal and interruption. The complete suite passed all
105 tests after the final viewer pacing adjustment, in 53.738 seconds. The
result is saved in `output/spike_pattern_skills/association-full-suite.log`.

The corrected live viewer was verified as an actual `pythonw` MuJoCo window,
with an empty stderr log. At the first archived live snapshot it had completed
ten episodes (five runs and five jumps), with ten task fulfillments, no
fallbacks, contacts or falls. This repeating known-context demonstration is
separate from held-out evaluation. Its ongoing logs are in
`output/spike_pattern_skills/live-associated-v2/`; an archived aggregate is
`pattern_association_v1/live_demo_snapshot.json`.

The only shared controller changes expose an optional viewer callback and pace
viewer synchronization at approximately 60 Hz instead of sleeping every 2 ms
physics step on Windows. Physics/control timesteps, gains, torques and guards
are unchanged. The training manifest fingerprints the collection-time source;
the subsequent viewer pacing change does not run during headless collection.

## Limits and next development

These are 100 deterministic synthetic contexts with 40/10/50 whole-context
splits. Their unlabeled neural responses had already been inspected in the
previous study. The final 50 are held out from association fitting and candidate
outcome collection, not an entirely unseen environment distribution.

Inputs currently come from causal simulator state through the established
14-feature adapter, not camera images. Jump request is an explicit task input.
The jump primitive is an in-place hop, not an obstacle-clearing policy. The
memory cannot infer the need to jump solely from seeing an object. Count
vectors omit within-window spike timing, which is nevertheless recorded.

The central weakness is forward progress around obstacles: only 2/20 held-out
forward tasks succeeded, and this three-skill bank has no bypass or retreat
option. Waits and the approaching-obstacle guard stop give concrete cases for
the next milestone. Extend the association contract to suitable tested navigation
skills, collect their outcomes, then evaluate a fresh set before adding repeated
decisions and interruptions. Simulator results do not establish hardware safety
or an advantage over other selectors.
