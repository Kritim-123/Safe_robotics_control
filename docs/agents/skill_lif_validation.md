# Run/jump dataset and spiking skill selection

Measured October 16, 2026. This experimental pipeline adds a fast trot and
in-place jump, collects actual MuJoCo outcomes, trains an output-spiking LIF
selector and executes its selected skills. It does not replace the legacy
navigation selector or claim obstacle-jumping capability.

## Behaviors and data

The `run` primitive is a fast classical trot: 0.6082 m during four commanded
seconds versus 0.2068 m for the original walk (about 2.94 times the progress).
Its 0.8274 m total includes deceleration and settling. The jump achieved
0.124 s contiguous all-feet flight, about 0.078 m base rise, and a stable landing.
Torque stays within 24 Nm/model limits. Jump trials include an initial tilt
failure and 18 development parameter trials; the final profile was frozen before
dataset collection. See [dynamic skill evidence](dynamic_skills.md).

`output/skill_lif/dataset/decisions.csv` contains **252 attempted skill executions
under 84 independent conditions**, including **70 unsuccessful or rejected
attempts**. Every condition tries wait, run and jump in fresh matched simulations.
Six families: clear, distant obstacle, blocked, crossing, jump requested in open
space, and jump requested with restricted space. Friction varies from 0.65 to 1.0.
Each row records 14 causal pre-action features, eligibility, attempted/applied
action, physical outcome, task fulfillment, utility-derived target and trace hash.
Per-attempt CSV trajectories and JSON summaries are in `dataset/trials/`.

There are 48 training contexts (seeds 1100–1107), 18 validation (2100–2102) and
18 diagnostic test contexts (3100–3102). Candidate siblings stay together;
normalization is fitted on training contexts only. Training uses one observation
per context rather than treating three sibling attempts as independent examples.
Labels rank successful eligible physical outcomes under an explicit task:
forward progress prefers a successful run; requested in-place jumping prefers a
verified jump; wait is a local fallback. A successful wait does not fulfill a
forward-progress or jump request. Future outcomes are never network inputs.

## LIF protocol and training

The new schema has 14 inputs, 32 hidden LIF neurons and **three output LIF
neurons**, ordered `wait`, `run`, `jump`. Each of 16 steps applies
`v = 0.9*v + input`, emits a binary spike at threshold 1 and subtracts the
threshold on firing. Output spike counts select the highest-firing eligible
skill. Ties follow saved action order; silence uses eligible wait or abstains.
Masks remain external to the learned network. Joint torque PID executes the
chosen primitive. State resets per decision window; this is not a trained
across-observation memory network.

The model was trained for 200 epochs using surrogate-gradient cross entropy on
output spike counts, seed 37. NumPy export counts exactly match PyTorch counts.
The first checkpoint-selection rule retained the earliest maximum validation
accuracy, giving 16/18 diagnostic test agreement and silent fallback cases.
After inspecting these results, selection was revised to break validation
accuracy ties with lower validation cross entropy. The final model has 48/48
training, 18/18 validation and 18/18 diagnostic test agreement, with no silent
eligible outputs or tied eligible maxima. Because that test split was inspected
before the revision, it is not an untouched final performance estimate.

The frozen final model was then executed on **18 new conditions**, seeds
4100–4102 across all six families. All 18 selected primitives completed without
contacts or falls: six runs, three jumps and nine waits. **9/18 fulfilled the
requested task**; the other nine were local wait fallbacks in blocked, crossing
or restricted-jump conditions. This demonstrates firing-to-skill execution,
not general obstacle navigation or a 100% task-success result.

Every execution saves `summary.json`, `trajectory.csv` and `output_spikes.csv`
with one column per final neuron. Aggregate results are in
`output/skill_lif/execution/evaluation.json`. Model and training provenance are
in `output/skill_lif/model/model.npz` and `training.json`.

## Reproduction and independent checks

Executed from the repository root with the existing environment:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.behavior_dataset --workers 3
.\.venv-win\Scripts\python.exe -m hybrid.train_skill_lif --epochs 200 --seed 37
.\.venv-win\Scripts\python.exe -m hybrid.run_skill_lif --seeds 4100 4101 4102
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Final full suite: **75 tests passed in 42.521 s**. Log:
`output/skill_lif/tests-final.log`. It covers physical flight/landing, run/walk
comparison, initiation refusal, interruption/timeout, torque cleanup, actual
binary output spikes, masking, tie/silence behavior and existing regressions.
Independent artifact audit verified all 252 trace hashes, disjoint seed groups,
candidate completeness, finite features, successful eligible utility winners,
torque limits and output-raster/count equality. See [audit](skill_dataset_audit.json)
and [compact metrics](skill_lif_metrics.json). `git diff --check` passed.

Deliberate selector abstention handling was added after collection; recorded
source hashes preserve the collection-time version. Unexpected callback errors
still propagate after torque cleanup. Raw outputs are git-ignored; retain the
output folders when moving or archiving the dataset. The collection manifest
records units, versions, label rules, source hashes and file hashes.

This is a small, privileged-sensing, single-decision dataset with the same scene
families across splits. It supplies neither perception training nor obstacle
clearance jumps, global planning, sequence-memory learning or hardware evidence.
The jump request is an explicit input command, not something inferred from an
obstacle. Larger diverse scenes and continuous skill transitions remain future work.
