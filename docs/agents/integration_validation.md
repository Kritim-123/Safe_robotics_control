# Memory and expanded skills: measured validation

Measured October 15, 2026 in the Windows project environment. **Retaining the
existing LIF membranes did not improve decisions in this study.** All 21 matched
reset/retain pairs had identical chosen skill sequences, episode elapsed times,
success outcomes and minimum footprint clearance. This is a negative result for
frozen-checkpoint retention, not a test of a sequence-trained memory network.

## Implementation

`TemporalSelector` updates at 10 Hz, including active skills and recovery. It
carries two membrane arrays, decays them by `exp(-dt/0.5)`, and resets spike
counts each 16-step window. Both arms use identical weights, inputs, four outputs,
masks, expanded skill mapping and replanning. Reset mode zeros membranes each
window; retain mode preserves them. Episode, obstacle identity, no encounter,
gaps over 0.5 s, invalid observations and backwards time reset state. Duplicate
same-obstacle timestamps reuse the result. A tracked encounter can persist
through skill execution/recovery even when the goal ray is clear.

The default legacy mode remains unchanged. `--expanded-skills` maps wait/back-off
scores to `yield_hold`/`retreat_hold`: classical composites, not new trained
logits. See [catalog](skill_catalog.md). Physics, actuator limits, gait/IK/PID and
clearance/contact/fall guards persist. Selector failure and no eligible action
terminate explicitly. At beta 0.9, homogeneous carry through one complete frame
is about 0.152; this is short encounter-local memory. The outer decay constant
does not establish a 0.5 s effective memory horizon. See [research](snn_research.md).

## Fresh results

Parameters were fixed before runs; development tests used seed 0, clean
evaluation used 810–812, and noisy evaluation used 820–822. No retraining or
tuning on these results. Clean batch: five families, three seeds, five variants,
**75 episodes**. All hybrid arms enable replanning. Rules use expanded skills;
legacy SNN uses legacy skills. Only reset versus retain isolates memory.

| Clean scenario | Baseline | Rules, expanded | Legacy SNN | Reset, expanded | Retain, expanded |
|---|---:|---:|---:|---:|---:|
| Static | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Crossing | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Head-on then stop | 0/3 | 0/3 | 3/3 | 0/3 | 0/3 |
| Oscillating gate | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Late crossing | 0/3 | 0/3 | 3/3 | 0/3 | 0/3 |
| Total | 3/15 | 9/15 | 15/15 | 9/15 | 9/15 |

Expanded failures were skill timeouts. Requiring all obstacles to clear can hold
indefinitely behind a static blocker: the head-on obstacle stops, and the
late-crossing original blocker remains after the new hazard passes. Legacy
execution finishes the local action and reselects a bypass. This regression is
from composite semantics, not memory; the catalog remains experimental.

The **12 noisy episodes** compare reset/retain on oscillating gate and late
crossing, three seeds each. Position noise 0.04 m, velocity noise 0.01 m/s,
delay 0.3 s, smoothing 0.2 s. Both variants: **3/6 success**, all gates successful
and all late crossings ending in skill timeout.

No hybrid arm had physical contacts or falls. Baseline had 12 contacts. Clean
reset/retain minimum measured footprint clearance: 0.14095 m; maximum torque:
21.0154 Nm. Noisy values: 0.37593 m and 20.8387 Nm. Each reset/retain arm had
three clean and three noisy preemptions. Common-success completion-time delta
was zero. No discordant success pairs: no evidence of superiority in this sample.

Memory changed scores in 15/18 clean selection windows and 9/9 noisy windows;
maximum absolute differences were 0.5263 and 0.5554. Raw argmax and masked
decisions never changed. Mean per-episode inference means were approximately
0.131/0.127 ms clean reset/retain and 0.131/0.127 ms noisy. Concurrent CPU timings
are descriptive, not a speed advantage. Timing covers sensory updates; spike
totals cover selected windows only.

## Reproduction and checks

Executed from the repository root:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.evaluate_memory --workers 3
.\.venv-win\Scripts\python.exe -m hybrid.evaluate_memory --scenarios oscillating_gate late_crossing --seeds 820 821 822 --variants reset retain --noisy --workers 2 --output output/hybrid/memory-noisy
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p 'test_expanded_skills.py' -v
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p 'test_memory_selector.py' -v
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Independent skills: 8 tests passed, including six physics executions. Initial
memory tests: 6 passed; three supervisor tests were then added. Final full suite:
**61 tests passed in 38.212 s**, including history dependence, reset equivalence,
identity/gap resets, eligibility, selector failure, expanded hold recovery and
persistent physics. Initial sandbox tests hit temporary-file/localhost permission
errors; the final suite ran with required local access. Log:
`output/hybrid/memory-tests.log`. No notebook/live dashboard integration was done.

Optional reproduction example (not an additional measured run):

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.run --scenario crossing --memory retain --expanded-skills --replan --headless --output output/hybrid/memory-demo
```

Legacy brain replay cannot reconstruct retained state and the CLI rejects it
for temporal modes. Use saved event scores. Raw scenarios, summaries, events,
trajectories and reports are in `output/hybrid/memory-study/` and
`output/hybrid/memory-noisy/` (git ignored). [Compact metrics](memory_metrics.json)
retain paired deltas, aggregates, report/source hashes and unchanged checkpoint hash.

## Remaining work

Revise composite termination on development scenes to allow fresh planning
around a remaining static blocker, then evaluate new held-out seeds. Train
retained LIF/ALIF on causal sequences with history-dependent targets, keeping
skills matched. This study does not separately measure reversal reaction delay,
feature clipping frequency, all-tick state traces, energy or broad generalization.
Sensing is privileged simulator data; no physical robot was tested.
