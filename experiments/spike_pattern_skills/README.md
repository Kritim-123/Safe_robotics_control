# Spike-pattern skill selection

An isolated workspace for the idea discussed on October 16, 2026:

> Initialize a seeded random SNN, record which output neurons fire for a robot
> observation, and learn an association between the firing pattern and a skill
> that successfully handles that situation.

**Status: recording v1 implemented.** A seeded random LIF encoder and spike
recorder now run on settled MuJoCo observations or saved observation JSON.
Pattern-to-skill learning and skill execution remain the next milestones.

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
records actual checks and the observed pattern collision. There are no trained
skill associations yet.

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

## Next steps

1. Collect patterns alongside attempted skills and actual simulation outcomes.
2. Implement pattern-to-skill learning, including unknown and ambiguous patterns.
3. Evaluate on separate scenarios before connecting the decoder to robot control.

Existing controllers, SNN checkpoints and skill-selection schemas remain the
reference systems for comparison. The new experiment will reuse their tested
skill interfaces and guards when integration is ready.
