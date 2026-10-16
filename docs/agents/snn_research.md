# SNN memory for changing obstacles

Research brief, 15 October 2026. Scope: high-level skill selection in MuJoCo;
joint torque PID, limits, and supervisor guards continue to control execution.
This brief proposes experiments; it does not report new robot experiments.
Fresh commands/results belong in the integration report.

## Inspected baseline and agreed scope

Read `AGENTS.md`, `README.md`, `docs/project_status.md`,
`docs/hybrid_system.md`, `docs/hybrid_validation.md`, the researcher role,
`hybrid/snn.py`, `hybrid/perception.py`, `hybrid/train_selector.py`,
`skills/avoidance.py`, and the skill README; inspected selector calls in the
supervisor. Historical validation results are not fresh evidence.

The inspected baseline has four selector actions, in checkpoint order:
`bypass_left`, `bypass_right`, `wait`, `back_off`. `walk` and `navigate` are
separate standalone entry points. One saved learned artifact selects skills;
there are no imported learned locomotion policies. Its 12-64-32-4 LIF network
resets both membranes on every prediction, repeats one analog observation for
16 internal steps, uses subtractive threshold reset without an explicit
refractory interval, and decodes mean second-layer spikes to four real scores.
It has memory inside a decision window, but none across observations.

The skill developer proposed two optional classical composites, `yield_hold`
and `retreat_hold`: a 1.5 s clear-route dwell, at most 60 s execution, and a
0.55 m retreat before holding for the latter. The agreed approach preserves
the four learned logits and exposes composites through an optional,
deterministic mapping of wait/back-off. These are additional executable skills,
not two newly learned output classes. Consult the skill catalog for the final
tested contract. Use the same mapping in every compared selector arm.

## Neuron choices and evidence

The following suitability judgments are proposals for this project. Published
classification or synthetic memory results do not demonstrate Go2 navigation
improvements.

| Variant | Memory and reset | Stability/training | Incremental inference cost and fit |
|---|---|---|---|
| Existing reset-window LIF | 16-step integration, then zero; subtractive reset, no explicit refractory state | Existing surrogate-gradient checkpoint; bounded leak below one | Baseline; no cross-frame memory |
| Retained LIF | Carry both membranes between sensor frames; reset counts per window | Same weights permit a clean mechanism ablation, but initial-state distribution differs from training; extra carry can create saturation or stale decisions | 96 retained floats and linear state-decay work; first experiment |
| Adaptive LIF (ALIF/LSNN) | Slow spike-triggered threshold adaptation plus membrane state; adaptation survives individual spikes | Requires sequence training; constrain decays and adaptation gain, specify refractory/reset equations; long unrolls complicate gradients | One extra adaptation state per neuron; recurrent synapses add matrix work if used; strongest next candidate for longer history |
| Parametric LIF (PLIF) | Learn decay/time constants rather than fixing them; still requires state retention across frames | Bound decay through a parameterization; train on sequences; learned long time constants can slow reversal response | Approximately LIF inference cost with learned constants; useful alternative to ALIF |
| Synaptic-current LIF | Separate decaying synaptic current and membrane state | Two stable decay factors; surrogate training and an explicit current-reset policy | Additional state/update per neuron; plausible noise integration but can lag abrupt changes |

Bellec et al. demonstrate adaptive-threshold recurrent SNN memory on computational
tasks; their official implementation includes sequential MNIST and store-recall
tutorials. It is TensorFlow-era code, licensed BSD-3-Clause-Clear, not a Go2
controller. [Paper](https://arxiv.org/abs/1803.09574),
[official code and license](https://github.com/IGITUGraz/LSNN-official).

Fang et al. demonstrate learnable membrane time constants on image/event
classification datasets, not physical robot navigation. Their official research
repository provides code/logs but did not expose a root license in the inspected
listing; do not infer reuse permission from availability. The current
SpikingJelly framework advertises Apache-2.0; verify the actual pinned version
and its notices if importing code.
[Paper](https://arxiv.org/abs/2007.05785),
[official research code](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron),
[framework](https://github.com/fangwei123456/spikingjelly).

The official snnTorch documentation supplies LIF and second-order synaptic
models, including recurrent forms. Source is MIT; documentation has separate
CC BY-SA terms. These are implementation resources, not evidence of robot
performance. [Neuron documentation](https://snntorch.readthedocs.io/en/latest/snntorch.html),
[code/license](https://github.com/jeshraghian/snntorch).

Surrogate gradients keep binary spikes in the forward pass while supplying a
tractable backward derivative. They fit the project's existing supervised
training path; no online learning requirement currently justifies adding
STDP, e-prop, or reinforcement learning to this first ablation.
[Neftci, Mostafa and Zenke](https://arxiv.org/abs/1901.09948).

## Input/output and memory contract

Preserve checkpoint feature order: obstacle forward/left position (m), obstacle
forward/left velocity (m/s), radius (m), current clearance (m), left/right/back
path clearances (m), predicted wait clearance (m), goal distance (m), heading
error (rad). Axes follow the goal ray, not robot heading. Velocities are observed
obstacle velocities projected on those axes, not robot-relative velocities.
Require a finite float vector, four eligibility flags, timestamp in seconds,
episode identity, and persistent obstacle identity. Timestamp/identity are
lifecycle metadata, not extra inputs to the unchanged network.

Keep the saved training-only mean/std and clipping at five standard deviations.
Report clipped feature frequency during evaluation. Signed constant-current
encoding already supports the features; Poisson encoding would add sampling
variance without a demonstrated benefit. Temporal differences or ON/OFF spikes
are later experiments requiring retraining and unchanged input information
across baselines. More microsteps increase compute and can erase prior-state
influence; they do not create more physical observation history.

Update once per new 10 Hz encounter observation, including during active skills,
for both reset and retained arms. Keep readout spike counts local to each
window. Reset retained state on episode start, obstacle change, cleared threat,
invalid/backward time, or stale observations; do not advance twice for one
timestamp. Log reset reason, state norm, update count, score margin, raw and
masked choices. A reset after threat clearance deliberately loses previous
encounter history: this is encounter-local memory, not global scene memory.

The integrator proposed inter-frame decay `exp(-dt/0.5)` and a 0.5 s stale-state
cutoff. These are provisional engineering settings, to freeze before evaluation.
At 10 Hz the homogeneous voltage carry through the full window is
`0.9**16 * exp(-0.1/0.5) ~= 0.152`, corresponding to about 0.053 s e-fold decay
if spike sequences remain fixed. Threshold crossings can change this local
calculation. Thus a 0.5 s outer decay parameter does not establish 0.5 s effective
memory. Do not change beta on a frozen checkpoint to manufacture longer memory.

Return four finite scores plus diagnostics; apply current eligibility masks
outside the network. A score margin is not a calibrated probability. Keep the
existing explicit no-eligible-action failure and guards. A confidence-based
wait fallback is a separate intervention requiring calibration on validation
episodes and a wait eligibility check; it must not silently enter only one arm.

## First experiment and acceptance criteria

1. Freeze checkpoint hash, code revision, memory settings, skill mapping,
   sensing/noise/filter settings and replanning. Compare rules, original reset
   LIF, and retained LIF on identical scenario seeds. Independent timing during
   each run is needed because continuous inference changes call frequency.
2. Before physics, replay two causal observation histories with the same final
   observation. Reset LIF must give identical final scores; retained LIF should
   demonstrate history-sensitive state/scores. Check episode isolation, identity
   changes, stale gaps, nonfinite input, masks and reset equivalence. Score
   differences alone do not demonstrate better decisions.
3. Use static controls, late crossings, velocity reversals, stop/start obstacles
   and noisy/delayed observations. Predeclare development and held-out seeds;
   never tune decay or dwell from held-out outcomes. Give all arms the same
   accessible observations and event schedules, never future scripted motion.
4. Report paired goal success, physical contact/falls, clearance-stop failures,
   minimum true clearance, completion time, interventions and unnecessary
   switches, reversal reaction delay, score/mask overrides and inference latency.
   Count failures in outcome summaries; report completion time on common-success
   pairs separately to avoid survivor bias. Separate sensor smoothing from
   neural retention effects.
5. Engineering acceptance: finite scores, unchanged schema/checkpoint and PID,
   reproducible resets, successful guard/handoff tests, and zero new contact or
   fall failures in the matched smoke batch. Scientific acceptance for a claim
   of improvement: held-out paired success gain with uncertainty excluding zero,
   no observed safety regression, and no material reaction-delay regression.
   A small tied batch supports only “no demonstrated benefit.” Use paired
   intervals/discordant-pair analysis; do not count repeated deterministic clear
   scenes or frames as independent trials. A larger claim needs more episodes.

This frozen-weight ablation answers whether retaining the existing membranes
helps this implementation. A negative result does not reject trained SNN memory.
Do not attribute improvements from new skills or replanning to retention.

## Next training experiment and integrator questions

If the ablation is inconclusive, train retained LIF and ALIF on the same causal
encounter sequences, with matched widths, seeds and budget; add reset-LIF and
rule comparators. Split by entire scenario/trajectory and motion family before
windowing; fit normalization only on training trajectories. Reset state at
episode boundaries and use explicit burn-in/truncated-BPTT rules. Include
ambiguous present observations whose histories differ, otherwise memory has
little reason to help. Teacher agreement is useful for debugging, but a
memoryless teacher cannot supervise an advantage requiring hidden history.
Use causal history-aware labels or outcome-based costs, with oracle future
information confined to training labels/evaluation and never runtime features.
Start with weighted cross entropy, gradient clipping and a separately tuned
small spike-rate penalty; export training/deployment parity tests and provenance.

Open questions: do threat-clear resets discard useful reversal history? Does
goal-frame rotation demand state reset or a different encoding? Are noisy
obstacle velocities already sufficient without memory? Is the effective horizon
too short? Does composite mapping improve behavior independently of retention?
Are raw score changes actually changing eligible actions? Resolve these with
logged ablations before escalating to larger networks or policy learning.
