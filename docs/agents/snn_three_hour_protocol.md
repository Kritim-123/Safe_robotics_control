# Three-hour SNN improvement session

Authorized October 17, 2026. One agent owns this milestone sequentially; no
delegation. Start: 2026-10-17 07:09:45 UTC (01:09:45 America/Denver).
Requested work-session end: 2026-10-17 10:09:45 UTC (04:09:45 America/Denver).
Use actual elapsed time, not simulated episode duration, for this request.

## Scope and ownership

Improve the frozen-pattern SNN experiment, especially repeated decisions and
within-episode skill changes discussed with the user. Own new research,
experiments, supervisor/executor, tests, models and report in
`experiments/continual_snn/`, `tests/test_continual_snn*.py`,
`docs/agents/continual_snn*`, and `output/continual_snn/`. Add narrow links to
README/status when results exist. Existing unrelated workspace changes and
all frozen historical checkpoints/banks remain intact. New motor behavior is
out of scope: preserve the existing wait/run/smooth-forward-hop primitives and
24/28 N m bounds. The 72 N m low-bar experiment remains outside SNN selection.

## Initial findings and interface agreement

The current demo encodes observations repeatedly but chooses a skill once.
Its dense 22-input, 16-output-neuron, 16-internal-step random LIF checkpoint
is frozen. The 94-entry bank mostly records settled pre-action observations.
Historic fresh fulfillment was 7/28, tied with direct-run in total but on
different tasks. Many failures are motor capability or conservative hold
labels, so model-only label accuracy is not a task-success claim.

Use the user's fresh-pattern idea: each real 100 ms decision packet has fresh
spike counts. Resetting a packet must not teleport the robot, erase telemetry,
zero motor torque during motion, or restart an unchanged skill. Retaining
membranes/history is a separately labelled ablation, not an implicit change.
Observe meters, seconds, radians, m/s, and binary explicit task requests.
Current observations are privileged MuJoCo state/map, not camera images.
Do not supply future scripted obstacle stopping/reversal times to policies.

Primary skill IDs remain `dynamic/wait`, `dynamic/run`, `dynamic/jump`.
Wait/run/jump requests are distinct from accepted entries and physical skill
completion. A pending request must be revalidated at handoff. Run->wait/jump
decelerates through the existing gait and requires supported stable entry.
An airborne jump finishes landing/recovery before ordinary handoff. Unknown,
silent, conflicting, stale or invalid inputs request a checked fallback.
Repeated same-skill decisions continue without restarting. Explicit jump
requests latch for one hop until released, preventing repeated hopping from
a sustained request. All contact, posture, clearance and finite-state guards
remain independent. Log each proposal, mask, buffer reset, deferred request,
handoff, interruption, task outcome, and motor trajectory.

## Research and experiments

Read primary sources for options/termination, fixed spiking reservoirs with
learned readouts, adaptive LIF, causal spike-train similarity and surrogate or
eligibility-trace learning. Inspect existing local negative memory evidence.
Do not equate a more elaborate neuron with demonstrated task improvement.

1. Freeze and reproduce existing bank/checkpoint behavior as a baseline.
2. Implement packet/reset/handoff engineering and causal scenario streams.
   Test same-action continuity, debounce, queued jump, stale requests, invalid
   input/fallback, episode isolation, cleanup and no state discontinuities.
3. Compare count-only, temporal-raster/filtered-spike readouts, and suitable
   neuron/representation variants on grouped train/validation/test contexts.
   Preserve seeded frozen weights as baseline. Train readouts/banks separately.
   Include a non-spiking feature comparator as a diagnostic, not a replacement
   for the selected project direction. Fit scales/hyperparameters on train/val.
4. Collect causal observations across actual motion/recovery where justified.
   Keep rule-teacher labels separate from measured outcome labels. Never label
   an unsafe/failed hop successful or use final outcomes as runtime features.
5. Run matched physical comparisons of one-shot versus repeated inference and
   baseline versus selected model on fresh changing scenes. Separate selector,
   supervisor and low-level motor effects through ablations. Include static
   controls, crossings/clearing blockers, stop/start task requests, held-out
   random starts/headings where supported, noise/delay and OOD diagnostics.
6. Freeze selected configuration before a final untouched confirmation batch.
   Retain all development and failed evaluation evidence. Discard unsupported
   ideas from recommendations, not from the audit trail.

## Acceptance and reporting

Engineering: finite reproducible outputs; preserved historical model hashes;
fresh independent count packets; actual multiple skill changes in one persistent
physics episode; no restart for repeated identical choices; jump commitment;
stable handoffs; guards and motor bounds; tests for eligibility/fallback/cleanup.
Scientific: matched held-out task success, physical contact/posture/clearance
stops, progress, switching count, reaction delay, common-success completion time,
latency and uncertainty. Count episodes/groups, not frames, as independent
units. Improvements from supervision/observations are not SNN-neuron advantages.
Negative comparisons remain negative. No hardware or energy-efficiency claims.

### Development amendment, 07:55 UTC

Unseen oscillating/multiple-obstacle development scenes exposed a representation
limit: nearest-obstacle features can be identical while a farther approaching
obstacle changes the feasible run request. A causal unit counterexample checks
this explicitly. Compare a versioned predictive-input extension using current
observed positions/velocities and fixed horizons, with no scripted future
motion. Preserve the original sensory population and train only on the existing
training groups, not stress outcome labels. Keep final test/confirmation seeds
unopened until the observation/model configuration is selected. Attribute any
gain to the extra information as well as the learned spiking representation;
it is not evidence that an elaborate neuron model is superior.

Run relevant checks first and the full suite after integration. The existing
HTTP tests require loopback access; use the existing project Python and project
`tmp/` for Windows temporary files. Save exact commands, configurations, splits,
seeds, runtime/source hashes, traces, compact artifacts and a concise research
and validation report. Use the three hours for useful work and repeat tests
only for new evidence, changes or unresolved concerns.
