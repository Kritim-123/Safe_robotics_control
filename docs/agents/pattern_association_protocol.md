# Pattern-to-skill association protocol v1

October 16, 2026. The user authorized associating frozen SNN outputs with skills,
watching execution and recording evidence for further development. One agent
owns the experiment modules, tests and reports, plus an optional viewer callback
in `skills/dynamic.py`. Existing actuation, guards, targets and gains persist.

## Contract fixed before outcome collection

Use the saved seed-42 random encoder from the 100-situation study, checksum
`b72b9750bffd70ee37bd0aaedb5064b58157d8f5a13f58b5cd356ca281fa70ec`.
Retain its observation units, ordering, normalization and zero-state resets.
The decoder receives only the 16 output spike counts. Scenario names, future
outcomes and skill labels are not extra decoder features.

Skills are `dynamic/wait`, `dynamic/run` and `dynamic/jump`, using the existing
tested implementations. Run is a four-second body-forward fast trot. Jump is
an explicitly requested in-place hop, not obstacle crossing. Preserve existing
eligibility, flight/landing, stable recovery, interruption, timeout and failure
conditions. One existing PID writes bounded torque, with persistent physics
through selection/execution/recovery. Reset only between separate episodes.

## Data and associations

Use the 100 recorded situations, split by whole context: variants 0-3 in each
family for training (40), variant 4 for validation (10), variants 5-9 for selected
execution evaluation (50). Their unlabeled representations were already viewed;
outcomes were not used before this milestone. Collect matched wait/run/jump
attempts for training and validation contexts: 150 attempts, including failures.

For forward tasks, prefer a successful eligible run with >.10 m displacement
along the heading at selection. For explicit jump tasks, prefer a successful
hop with >=.04 s flight and stable landing. Successful eligible wait has zero
task utility and is a local hold fallback, never task fulfillment. Unrequested
jumping and running instead of a requested jump do not compete. Keep unlabeled
contexts when no candidate succeeds. Retain all trials and rejected starts.

Store one association per valid training context and preserve conflicting labels.
Decoder v1 checks exact count matches first; conflicting matches abstain. Novel
vectors use up to three nearest training vectors within Euclidean count distance
4.0. Include all ties at the neighbor cutoff and propose a skill only when all
included neighbors agree. Silence, no neighbors and conflicting labels abstain.
These parameters are fixed before collection, without validation/evaluation tuning.

Keep preference separate from supervision: unknown, ambiguous or ineligible
proposals use currently eligible wait, otherwise no skill. Freeze the association
artifact before evaluation. No online model updates occur during the live demo.

## Evidence and acceptance

Save causal observations, spike rasters/counts/IDs, eligibility, candidate actions,
primitive outcomes, task fulfillment, body-frame progress, flight, torque,
trajectories and hashes. Flush live telemetry as JSONL and save the decision and
terminal summary, including unexpected errors or viewer closure.

Show actual MuJoCo frames with live diagnostic spikes, the frozen decision
pattern, learned proposal, actual skill and progress. Each episode has one
settled-state decision, skill and recovery. Live diagnostic spikes do not cause
unvalidated mid-skill switching. Log episode resets explicitly.

Verify known matches, conflict/unknown abstention, eligibility/fallback, actual
run and hop completion, refusal, interruption, timeout, recovery, logging and
callback cleanup. Run the full suite after the shared wrapper extension. Report
coverage, primitive completion and task fulfillment separately, including every
failure; no retrospective accuracy target. No hardware deployment is included.
