# Spike-pattern skill selection

An isolated workspace for the idea discussed on October 16, 2026:

> Initialize a seeded random SNN, record which output neurons fire for a robot
> observation, and learn an association between the firing pattern and a skill
> that successfully handles that situation.

**Status: count associations and live execution implemented.** A frozen random
LIF encoder feeds an outcome-labeled wait/run/jump association bank. Live
MuJoCo execution saves neural activity, decisions, motion and measured outcomes.

## Intended pipeline

```text
Robot/environment observations
    -> seeded random LIF network
    -> recorded spike pattern, e.g. [1, 3, 5, 7]
    -> learned pattern-to-skill association
    -> skill eligibility checks
    -> skill execution and measured outcome
```

The first version will keep the SNN's random weights fixed and learn the
association separately. Updating the encoder weights can be a later experiment;
that would also require updating the associations tied to its old patterns.

## Files

- `observations.py`: ordered 14-feature contract, units and fixed input scales.
- `encoder.py`: frozen random weights/biases, LIF inference and checked save/load.
- `record.py`: CLI capture/replay and JSONL spike recording.
- `DESIGN.md`: implemented observation/spike contract and proposed labeling work.
- `config.example.json`: runnable v1 settings, network seed 42 and scene seed 100.

## Run and inspect

From the repository root:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.live
```

This opens MuJoCo with a crossing obstacle, live active-neuron IDs, all 16
spike counts and a count chart. The frozen network observes at 10 Hz while the
existing PID holds the robot standing. Each readout uses the same reset 16-step
encoding window as the recorded experiment. No pattern-to-skill decoder runs.
Press **R** to restart the scene, **J** to toggle the explicit jump-request input
(this does not command a jump), or close the window to stop. Use
`--case static_box-00`, for example, to view another of the 100 study cases.
Guards stop physics on contact, low height, excessive tilt, nonfinite state or
insufficient clearance; R starts a fresh scene after a stop.

To record the original six-scene capture batch without opening a window:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.record --output output/spike_pattern_skills/my-capture
```

This captures six scenes after two seconds of standing. The existing bounded
standing controller supplies support; the capture callback deliberately abstains
before any skill runs. Random network outputs never command robot motion.
The underlying wrapper reports `no eligible skill` for this deliberate stop;
that is recorded as capture-only termination, not task success or a skill label.

Each output folder contains `model.npz`, `config.json`, `observations.json`,
`records.jsonl` and `summary.json`. A record includes raw named observations,
active neuron IDs, per-neuron counts, the full time-by-neuron binary raster,
membrane traces and encoder checksum. Existing nonempty output folders are
preserved; choose a new name for each run.

Replay those exact observations and weights without stepping physics:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.record --model output/spike_pattern_skills/my-capture/model.npz --observations output/spike_pattern_skills/my-capture/observations.json --output output/spike_pattern_skills/my-replay
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p test_spike_pattern_skills.py -v
```

The [first milestone report](../../docs/agents/spike_pattern_recording.md)
records the initial checks and observed pattern collision. Outcome-labeled
count associations were added in the later milestone described below.

## 100-situation study

The [100-situation report](../../docs/agents/spike_pattern_100.md) includes
committed data and the count heatmap: 40 active-ID patterns, 99 count vectors,
and 100 timing rasters from the same frozen encoder. Every neuron's count is
recorded in `docs/agents/spike_pattern_100/situations.csv`; full records and the
actual checkpoint are archived beside it.

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.study --model docs/agents/spike_pattern_100/model.npz --output output/spike_pattern_skills/my-study-100
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.audit
```

`situations.py` defines the fixed scene protocol, `study.py` runs capture and
exports counts, `analyze.py` compares representations, and `audit.py` independently
checks the published data and LIF arithmetic. Plot generation uses the existing
optional Matplotlib dependency in `requirements-experiments.txt`.

Run future commands from the project root. Save generated networks, recordings
and evaluation artifacts under `output/spike_pattern_skills/`, following the
project's existing output convention. That directory is git-ignored; retain
compact evidence and milestone reports under `docs/agents/`.

## Count associations and live execution

For the updated 94-example, terrain-aware model, run:

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.run_learned --viewer
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.run_learned --video
```

This finite demo executes a learned flat-course run, gentle ramp crossing and
requested short forward hop, with flushed neural/physical telemetry. Its new
default motor profile is `smooth_forward`; use `--jump-profile legacy` to replay
the earlier in-place hop. See the [jump update](../../docs/agents/smooth_forward_jump_validation.md).
It loads the
portable bundle from `docs/agents/expanded_association_v2/`. Close the viewer
to stop. See the [expanded outcome-learning report](../../docs/agents/expanded_association_validation.md)
for fresh comparisons, remaining failures and training commands. The original
v1 commands below retain their original model and observation contract.

The frozen random SNN now feeds a separate outcome-labeled association memory.
It uses all 16 spike counts to select `dynamic/wait`, `dynamic/run`, or
`dynamic/jump`, subject to the existing skill eligibility checks and actuator
limits. Unknown/conflicting patterns fall back to eligible wait or abstain.
The network weights stay frozen; this milestone learns the association memory.

```powershell
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --viewer --repeat
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --video --output output/spike_pattern_skills/my-recorded-demo
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.execute_associations --evaluate --output output/spike_pattern_skills/my-held-out-evaluation
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.learn_associations --output output/spike_pattern_skills/my-association-collection
```

The default live demo alternates a learned forward run and an in-place jump.
Close the viewer to stop. Each episode settles for two seconds, selects once,
executes, and recovers. A fresh scene starts between episodes. This is not yet
continuous navigation or online learning. New count windows during movement
are diagnostic and do not interrupt the selected skill.

Every episode flushes `events.jsonl` and `telemetry.jsonl` while running, then
writes `decision.json`, `trajectory.csv`, and `result.json`. The telemetry
contains named inputs, spike IDs, counts, raster, membranes, robot pose/velocity,
and actuator commands. `--video` also produces a labeled MP4. Errors retain
partial telemetry. Existing episode folders cannot be overwritten.

The reusable association bank and compact evidence are in
`docs/agents/pattern_association_v1/`. See the
[protocol](../../docs/agents/pattern_association_protocol.md) and
[validation report](../../docs/agents/pattern_association_validation.md).

`live.py` remains the earlier standing-only neural observer. Use
`execute_associations.py` for movement through the learned association bank.

## Gradual terrain curriculum

`curriculum.py` adds ten ordered levels with six variants each: flat control,
thresholds, up/down ramps, low steps, uneven blocks, cross-slopes, friction
transitions, static passages, moving crossings, and mixed courses. These are
actual collidable support surfaces. The original plane remains underneath;
there are no pits or gaps. Feet may contact terrain; other robot-body contacts
with terrain fail the episode. Scenario JSON includes full terrain geometry.

```powershell
# Run every case with frozen associations and a direct-run diagnostic baseline
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.run_curriculum --output output/spike_pattern_skills/my-terrain-assessment

# Enforce level gates: stop advancing each controller at its first failed level
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.run_curriculum --progressive --output output/spike_pattern_skills/my-progressive-run

# Watch selected levels in increasing order; this is a diagnostic preview
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.run_curriculum --stages 1 2 3 --variants 0 --modes direct_run --viewer --output output/spike_pattern_skills/my-terrain-viewer

# Render all ten levels without executing a skill
.\.venv-win\Scripts\python.exe -m experiments.spike_pattern_skills.preview_curriculum --output output/spike_pattern_skills/my-terrain-overview.png
```

Add `--video` to record MP4 episodes. All runs save scenarios, a source/geometry
manifest, flushed episode telemetry, decisions, terminal trajectories, results,
and `assessment.json`. Both primitive completion and actual course traversal
are measured: completion while waiting is not a successful crossing. Difficulty
gates require at least 80% traversal success in development and assessment
groups, without contact or falls. Later levels remain locked after a failure;
the default diagnostic sweep still measures them to identify weaknesses.

Variants 0-2 are development, 3 is validation, and 4-5 are held out; later
variants have harder family parameters. Flat variants repeat the same physical
reference condition. The current SNN has no terrain-height or local-friction
inputs and is not retrained by these commands. This milestone expands the
environment and assessment tools. See the
[terrain report](../../docs/agents/terrain_curriculum_validation.md) and
[protocol](../../docs/agents/terrain_curriculum_protocol.md).
