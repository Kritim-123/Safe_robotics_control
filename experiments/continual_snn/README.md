# Continuous SNN skill selection

This opt-in experiment extends the frozen-pattern idea to **multiple decisions
within one persistent MuJoCo episode**. It does not change the older navigation
SNN or the one-shot `spike_pattern_skills.run_learned` demo.

```powershell
.venv-win/Scripts/python.exe -m experiments.continual_snn.demo --viewer --video
```

The finite default demo requests run, jump, run, then hold. It records actual
motion, per-neuron spike counts, requested/executed skills, handoffs, and motor
outcomes. Add `--family two_hops`, `--family wait_release`, or
`--family oscillating_gate`. The last is a stress diagnostic and can fail.
Choose `--mode one_shot` for the matched single-decision ablation. Every run
gets a new timestamped folder under `output/continual_snn/`.

The validated default is `docs/agents/continual_snn_v2/primary.npz`: an
80-neuron frozen encoder with a learned count-pattern association bank. The
first frozen study's ridge readout remains at `continual_snn_v1/primary.npz`.
For the compact original-16-neuron association alternative:

```powershell
.venv-win/Scripts/python.exe -m experiments.continual_snn.demo --viewer --model docs/agents/continual_snn_v1/compact_motion_bank16.npz
```

## Observation and action contract

Observations currently use privileged MuJoCo state and a static terrain map,
not robot camera images. The 22 sensory features are the existing nearest
obstacle geometry/velocity, footprint clearance, relative goal, base speed and
height, explicit jump-request bit, and eight terrain descriptors. The two new
inputs are predicted minimum run and jump clearance in metres, computed from
**currently observed** obstacle positions and velocities. They consider all
observed obstacles, arena bounds, a 1.6 s run horizon and 3 s hop reservation.
They never read future scripted stopping/reversal times. Constant-velocity
prediction is an approximation. The noise/delay tests affect obstacle sensing;
they do not simulate a complete camera pipeline.

The primary encoder has 80 frozen LIF neurons: 64 sensory neurons and a separate
16-neuron predictive population. The first 64 retain the validated sensory
encoder exactly; the first 16 also retain the original historical checkpoint.
Added sensory and predictive seeds are 20261031 and 20261101. Actual full weight
and bias arrays, feature schema and checksums are stored in the artifacts.
Every 100 ms inference starts with zero membrane state and fresh counts, runs
16 constant-input internal updates, and records IDs plus the **number of spikes
per neuron**. Those internal updates are not asserted to be biological time.

The encoder is not trained. The default bank learns explicit count vectors
and labels from measured historical contexts and separately tagged teacher
trajectories. New hard-scene teacher trajectories require successful whole episodes. The
earlier group/class-weighted ridge readout remains a comparison; its softmax
score is not calibrated confidence. The count-pattern bank preserves labels;
an exact conflicting vector, unknown distant vector, or silent pattern abstains.
The ridge readout also abstains on silence. No model learns during evaluation.
Training source labels distinguish historical measured outcomes from causal
geometry/task-teacher requests; the latter are not counterfactual motor outcomes.

Both models propose `dynamic/wait`, `dynamic/run`, or `dynamic/jump`. A separate
supervisor checks eligibility, needs two confirming packets, and avoids
restarting an unchanged skill. Run-to-other transitions brake through the
existing gait. Entry requires four-foot supported stability. Requests are
rechecked after braking. A hop may cancel before thrust; airborne hops retain
their landing/recovery commitment. A sustained jump request permits one hop
until released. Missing eligible fallback stops the episode. A stale packet
requests checked wait/braking. Packet reset does not reset robot state or PID.

One existing PID/controller writes motor commands: 500 Hz physics/PID,
100 Hz joint targets, 10 Hz selection. Bounds remain 24 N m for run/wait and
28 N m for the smooth short hop. The separate 72 N m obstacle-jump experiment
is not selectable here. Contact, posture, height and clearance stops remain.

## Reproducible evidence

See [research](../../docs/agents/continual_snn_research.md),
[protocol](../../docs/agents/snn_three_hour_protocol.md), and
[validation report](../../docs/agents/continual_snn_validation.md).
`evaluate.py` declares matched scenes, arms, noise settings, model hashes and a
wall-clock cutoff before final runs. All final runtime/dependency sources and
model files are checked against that freeze; every batch also snapshots sources
and models. The original session's fixed cutoff is intentionally recorded.
For later individual reproductions use `demo.py`, or `study.py` with a new output
directory rather than overwriting archived batches.

Useful modules: `fit.py`, `fit_memory.py`, `fit_prototypes.py`, and
`fit_predictive.py` compare offline configurations; `study.py` collects/evaluates
finite batches; `audit.py` replays decisions and checks traces/provenance;
`report.py` builds CSVs and scientific plots. Failure logs remain available.
An episode marked `completed` can still fail its task because progress, verified
hops, hold stability, or hop counts were wrong. Simulation evidence establishes
neither physical robot safety nor an advantage over non-spiking controllers.
