# Spike-pattern experiment: recording milestone

Measured October 16, 2026, in one chat. The isolated experiment now implements
seeded random LIF initialization, checked model save/load, named observations
and complete spike recording. It does not yet learn or execute skill associations.

## Implemented contract

The encoder is **14 physical/task inputs -> 16 pattern LIF neurons**, with no
hidden layer or recurrent connections. Initialization uses PCG64 seed 42:
weights Normal(0, 1/sqrt(14)), then biases Normal(0, .1). Save the actual float64
arrays, source feature order/units, scaling, neuron settings and runtime versions.
Model loading rejects checksum mismatches and inference refuses mutated encoder
contents. Network and simulation seeds are recorded separately.

Named features reuse the existing run/jump observation builder. They describe
the nearest obstacle, robot posture/motion, goal and an explicit jump request.
Fixed physical divisors normalize them, followed by clipping to [-3, 3]. They
come from MuJoCo state, not cameras. No outcome or skill label is an input.
See the full [observation contract](../../experiments/spike_pattern_skills/DESIGN.md).

A decision resets membranes to zero, then applies the observation as constant
current for 16 internal steps. Each neuron uses beta .9, threshold 1 and a
subtractive reset, with at most one binary spike per step. These encoding steps
are not successive sensor frames and have no assigned physical duration.

Every record retains the observation, model checksum, binary output raster,
per-neuron counts, sorted active neuron IDs and membrane values before/after
reset. Pattern neurons have no predefined skill meanings. Labels remain null.

## Actual capture and replay

Captured **six observations at simulation t=2 s**, after the existing bounded
controller stood the robot in six scene families, seed 100. The callback records
the observation and returns `None` before executing any dynamic skill. The
underlying wrapper's `no eligible skill` termination is deliberate capture-only
abstention; it is not reported as task success or a network-selected action.
There were no obstacle contacts or falls in these captures. Existing PID limits
and guards were retained.

| Scene | Active output neuron IDs |
|---|---|
| Clear | [1, 2, 3, 4, 11, 12] |
| Distant obstacle | [2, 3, 4, 11] |
| Blocked | [2, 3, 4, 8, 9, 15] |
| Crossing obstacle | [2, 3, 4, 8, 9, 15] |
| In-place jump requested, open space | [1, 2, 3, 4, 9, 11, 12] |
| In-place jump requested, restricted space | [1, 2, 4, 8, 9] |

There were **five distinct active-ID patterns and no silent outputs**. Blocked
and crossing produced the same active-ID pattern. Their counts differ—for
example neuron 2 fired seven times for blocked and three times for crossing.
This demonstrates a collision in the compressed representation; it does not
establish which decoder will generalize or which skill those scenes require.
The seed was not changed to eliminate the collision.

Repeated inference, inference after intervening observations, and model reload
produced exactly matching records. A separate CLI replay loaded the saved model
and raw observations without stepping physics; its `records.jsonl` was
byte-identical to capture output. This verifies this environment's deterministic
replay, not robustness to changed observations or cross-platform guarantees.

Encoder content checksum:
`b72b9750bffd70ee37bd0aaedb5064b58157d8f5a13f58b5cd356ca281fa70ec`.

Raw artifacts: `output/spike_pattern_skills/final-v1/` and
`output/spike_pattern_skills/replay-v1/`. They contain model arrays, runnable
config, observations, complete JSONL records and summaries with file/source
checksums. Compact evidence is retained in
[`spike_pattern_recording_metrics.json`](spike_pattern_recording_metrics.json).

## Commands and checks

Executed from the project root:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.record --output output/spike_pattern_skills/final-v1
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.record --model output/spike_pattern_skills/final-v1/model.npz --observations output/spike_pattern_skills/final-v1/observations.json --output output/spike_pattern_skills/replay-v1
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p test_spike_pattern_skills.py -v
```

**Eight focused tests passed in .437 s.** They cover independent hand-calculated
LIF transitions, binary/count/ID consistency, zero-state reset, feature ordering,
same/different seeds, frozen contents, save/load, checksum rejection, invalid
inputs, exclusion of outcome metadata from inference, real MuJoCo capture,
file replay and preservation of existing output directories. Log:
`output/spike_pattern_skills/tests-v1.log`.

An initial smoke attempt used an incorrect wrapper summary key (`peak_torque`
instead of `max_torque`); the capture adapter was corrected before the final
run. The existing simulation/controller source was not changed. The broader
regression suite and navigation evaluation batches were not rerun for this
isolated recording milestone.

## Next milestone

Collect actual skill attempts alongside these patterns and task objectives,
retain failures and rejected starts, then learn pattern-to-skill associations.
Keep conflicting labels, unseen patterns and decoder abstention explicit.
Compare active-ID matching with spike-count/temporal representations on separate
scenario groups. Only after that evaluation should decoded choices enter a
supervised robot execution loop. The current hop remains an in-place skill;
no observation here establishes obstacle-crossing jump feasibility.
