# Project progress summary

Updated October 16, 2026 (America/Denver). This summary covers the current
Go2 MuJoCo implementation and its recorded experiments. Earlier reports retain
their original dates and results; experiment totals below are historical evidence,
not newly rerun evaluations for this summary.

## What has been built

1. **Simulation and classical motion.** Imported the licensed MuJoCo Menagerie
   Go2 model, built flat and furnished indoor environments, and implemented
   standing, walking and world-frame waypoint navigation. Motion uses foot
   trajectories, inverse kinematics and bounded joint torque PID with
   anti-windup. Navigation reports arrival, contact, falls and timeouts.
2. **Hybrid obstacle avoidance.** Implemented persistent physics through goal
   tracking, skill execution, stable recovery and return to goal tracking. The
   original learned selector chooses `bypass_left`, `bypass_right`, `wait` or
   `back_off`; geometric eligibility masks and failure guards supervise execution.
   Joint PID remains active during skill selection and execution.
3. **Trained spiking selector.** Delivered a 12-64-32-4 LIF model with 16-step
   current encoding, binary hidden spikes and four real-valued output scores.
   Training imitates a geometric teacher; exported NumPy weights, feature/action
   ordering and provenance are tracked in `skills/policies/selector_snn/`.
4. **Disturbances and replanning.** Added moving and reversing obstacles, boxes,
   pushes, friction variation, noisy/delayed observations and optional smoothing.
   Optional active-skill replanning can interrupt an invalidated bypass and
   reselect without resetting the simulation.
5. **Tools and diagnostics.** Added runnable notebooks, CLI commands, editable
   scenarios, evaluation batches, trajectory/event logs, videos, spike plots,
   an interactive SNN inspector and a live simulation dashboard.
6. **Memory experiment and composite skills.** Added optional retained/reset
   membrane inference and `yield_hold`/`retreat_hold` composites. These reuse
   the four-action checkpoint; they do not add learned output classes. The
   documented retention study found no decision benefit, and composite holds
   can time out behind persistent blockers. They remain experimental.
7. **Run/jump and actual output spikes.** Added a faster classical trot and an
   in-place hop with measured flight and stable landing. A separate 14-32-3 LIF
   pipeline collects real simulated outcomes and trains one binary-spiking
   output neuron per `wait`, `run` and `jump`. It does not replace the original
   avoidance selector or provide obstacle-clearing jumps.
8. **Pretrained motor-policy transfer.** Vendored MIT-licensed, pinned
   SafeRoboticsLab inference code and walker/jump weights with checksums. The
   experimental adapter supplies terrain scans and policy history while keeping
   torque limits. Flat walking worked; gap-crossing transfer did not. The
   imported jump policy is excluded from SNN selection.

## Recorded results and what they mean

| Experiment | Recorded outcome | Interpretation |
|---|---|---|
| Main navigation holdout | Baseline 18/40; rules 40/40; SNN 40/40 | Working hybrid loop; no demonstrated SNN advantage over the teacher |
| Filtered noisy navigation holdout | Baseline 2/30; rules and SNN 30/30 each | Success under a modest simulated noise model, not real perception |
| Optional replanning, new late-crossing seeds | SNN 5/5 | Targeted improvement under a separate optional protocol |
| Memory/composite study | 87 episodes; 21 matched reset/retain pairs chose identical skill sequences | No measured benefit from retaining the frozen model's membranes; composite timeouts remain |
| Dynamic primitives | Run progressed about 2.94 times farther than walk over the same four active seconds; hop had 0.124 s continuous flight | Faster trot and in-place jumping demonstrated on flat terrain |
| Outcome dataset | 252 attempts across 84 conditions, including 70 unsuccessful/rejected attempts | Failures retained; sibling attempts grouped across train/validation/test splits |
| New-seed wait/run/jump execution | 18/18 primitive completions; 9/18 task fulfillments; no contacts/falls | Nine waits were local fallbacks, not fulfilled forward/jump tasks |
| Frozen selector on new obstacle contexts | 17/18 primitive completions; 3/18 task fulfillments | Limited generalization; separate open-space jump control succeeded 3/3 |
| Pretrained gap policy | Flat walker control succeeded; gap crossings 0/7 | Transfer remains unsuccessful under the current model and limits |

The wait/run/jump diagnostic test split was inspected before checkpoint-selection
criteria were revised. Its reported 18/18 label agreement is not an untouched
final estimate. The subsequent execution and new-obstacle reports distinguish
physical primitive completion from fulfillment of the requested task.

Sources: [navigation validation](hybrid_validation.md),
[memory/composite validation](agents/integration_validation.md),
[dynamic primitives](agents/dynamic_skills.md),
[dataset and output-spiking selector](agents/skill_lif_validation.md),
[new-obstacle showcase](agents/new_obstacle_showcase.md), and
[pretrained transfer](agents/pretrained_jump.md).

## What is retained in Git

The repository contains simulation/controller code, notebooks, tests, guides,
compact experiment metrics, the original avoidance checkpoint, and the imported
motor-policy assets with license/provenance. Project workflow instructions and
the three reusable role definitions are also included; the standing workflow
uses one agent in one chat at a time.

`output/`, `tmp/` and local virtual environments remain ignored. In particular,
the collected CSV dataset, per-attempt trajectories, videos and experimental
wait/run/jump checkpoint under `output/skill_lif/` are local artifacts, not
included in this Git snapshot. Collection, training and execution commands are
documented in the linked reports so those outputs can be regenerated.

## Current verification

The full regression suite was rerun for this snapshot:

```powershell
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

**79 tests passed in 50.237 seconds.** The initial sandbox run encountered
temporary-file and localhost permission errors; the rerun with the required
access passed. Its log is retained locally at `output/project-summary-tests.log`.
All 10 vendored provenance hashes and this summary's relative links were also
verified. `git diff --check` passed.

Historical milestone reports record their own test counts. No training run or
full evaluation batch was repeated as part of this documentation update.

## Remaining work

- Improve composite termination so a remaining static blocker permits fresh
  planning instead of indefinite holding.
- Train and evaluate temporal models on causal sequences with genuinely
  history-dependent targets.
- Expand held-out scene diversity and continuous skill transitions; improve
  task fulfillment beyond conservative waiting.
- Add sensor-derived perception and broader route planning.
- Adapt pretrained locomotion/jump policies to the actual robot model and torque
  limits, then validate clean crossings before adding selector eligibility.

The system remains a simulation prototype using privileged robot/obstacle
observations. It has not established hardware safety, broad generalization,
neuromorphic energy savings or SNN superiority. For setup and commands, start
with the [README](../README.md) and [hybrid guide](hybrid_system.md); the earlier
development narrative is in [PROJECT_HISTORY.md](../PROJECT_HISTORY.md).
