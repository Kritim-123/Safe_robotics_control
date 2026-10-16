# Simulation data and spiking skill outputs

Protocol proposed 15 October 2026. This document specifies the new training
milestone; implementation and measured results must be reported separately.
It does not claim that run/jump or a newly trained model have passed tests.

## Starting point and boundary

The required project documents and current selector, perception, trainer,
memory wrapper, skill catalog and integration report were inspected. There
are four legacy learned outputs: bypass_left, bypass_right, wait, back_off.
Walk/navigate are standalone skills; yield_hold/retreat_hold are optional
classical mappings, not additional learned outputs. The earlier retention
experiment reported no changed decisions in 21 paired trials; its checkpoint
was not sequence-trained. Preserve that result as historical evidence.

The present final layer produces analog scores from hidden spike averages.
The requested model must instead have one actual output LIF neuron for each
skill in a versioned, explicitly saved action list. Adding run/jump requires
a new checkpoint schema and retraining. Do not append untrained rows to the
old checkpoint or count aliases as separately learned classes. The precise
action list follows the independently tested skill catalog. The integrator and
skill developer agreed a separate experimental menu: `wait`, `run`, `jump`
(three output neurons), with that order used for ties. This is not an expanded
six-class replacement for the legacy avoidance model. Run initially means a
faster classical trot, without claiming aerial running. Jump initially means
an in-place crouch/thrust/landing maneuver; airtime and stable landing require
measurement before calling its execution successful.

The selector issues skill identifiers, never joint torques. Each skill owns
its preconditions, target generation, duration, completion, interruption and
failure conditions; the supervisor retains physics, PID and actuator limits.
Jump eligibility needs obstacle height/width and landing-space evidence plus
robot posture/speed, or an explicitly restricted test scene. Circular 2D
obstacle radii alone cannot establish jump feasibility. A failed jump remains
a failed outcome even if a command was successfully sent.

For an in-place jump, a causal task request such as `jump_requested` can define
why that action is desirable. Store task context separately from obstacle
measurements and include it explicitly in the feature schema. Label preference
must combine task fulfillment with physical success: standing safely does not
fulfill a requested jump or forward-progress task. Do not label an in-place
jump as clearing an obstacle or report request-conditioned skill classification
as learned obstacle-crossing navigation. The skill developer requires supported,
stable flat posture and an open footprint for these initial jump trials.

## Causal CSV contract

Prefer one decision table and one outcome table joined by rollout identifiers;
a combined CSV is acceptable if the feature allowlist is explicit. Never
automatically treat every numeric column as a network input.

| Group | Suggested columns | Use |
|---|---|---|
| Provenance | schema_version, episode_id, encounter_group_id, rollout_id, scenario_family, scenario_seed, sensing_seed, split, source_revision | Grouping and reproducibility; excluded from features |
| Time | decision_index, observation_time_s, observation_age_s, obstacle_id, reset_reason | Causal ordering/state lifecycle; only age is an optional trained feature |
| Observation | Explicit named physical measurements and units; existing 12 goal-relative features, plus justified run/jump capability measurements | Network inputs available at decision time |
| Applicability | allowed_<skill> for every action, eligibility_reason | External mask from current observation |
| Behavior | requested_skill, executed_skill, selection_source, policy_hash | What actually ran; not automatically a desirable target |
| Label | label_skill, label_source, label_valid, label_cost | Supervision with recorded derivation |
| Outcome | success, termination_reason, contact, fall, timeout, duration_s, progress_m, minimum_clearance_m, max_torque_nm | Future measurements; never runtime input |

Keep raw failures, rejected starts, aborted episodes and no-eligible-action
cases. Exclude or specially handle invalid labels in training, while reporting
their count. A recorded action establishes behavior cloning data; it does not
establish that another action would have been worse. Successful demonstrator
actions may supply imitation targets with that limitation stated.

For outcome-based labels, run eligible alternatives from matched initial
conditions, with identical obstacle/sensor schedules and controller state.
Use independent fresh episodes or complete state clones, not approximate
position resets. Group these counterfactual siblings under one encounter ID.
Predeclare lexicographic preference: successful completion without contact/fall,
then task progress/completion time, then a deterministic tie rule. Keep cases
where all alternatives fail as failures, rather than choosing the least-bad
option and calling it successful. Full-episode success and individual-skill
success are separate fields if both are collected.

If collection uses only one action per initial condition, call resulting
labels demonstrator/heuristic labels; avoid claiming counterfactual optimality.
Synthetic geometric teacher samples can supplement training but must have a
different source tag from actual articulated MuJoCo rollouts. Dataset reports
must give physical episode counts, not just CSV row counts.

Split before window generation by entire encounter groups and motion schedules.
All seeds, counterfactual alternatives and overlapping history windows from
one encounter remain in one split. Do not random-split adjacent rows. Reserve
scenario families or obstacle-motion ranges when claiming generalization.
Fit normalization, clipping parameters, class weights and any label thresholds
using training/development data only; choose epochs on validation data and
evaluate test data once. Report class and outcome coverage in each split;
missing successful jump examples cannot support learned jump selection.

Save CSV hashes, exact feature/action order and units, split IDs, RNG seeds,
collection configuration, scenario files, code revision, dependency versions,
checkpoint hash and label-generation settings in a manifest. Nonfinite values
must be rejected or represented by a documented validity channel, not silently
converted into successful labels.

## Actual output LIF and decoding

For each internal step, after the last hidden layer emits binary spikes `s2`:

```text
u3_pre = beta3 * u3 + W3 @ s2 + b3
s3 = (u3_pre >= threshold3)             # one binary neuron per skill
u3 = u3_pre - threshold3 * s3           # subtractive reset
counts += s3
```

Save beta/threshold, immediate-versus-delayed reset semantics, internal step
count, initial state and action order. Use exactly the same equations in
training and NumPy deployment. No explicit refractory period is required for
the first implementation; this means at most one spike per internal step,
not a biological firing-frequency guarantee. Bound leak below one and monitor
saturation, dead outputs, and finite membrane values.

Decode the maximum output spike count among currently eligible actions. Resolve
positive-count ties in a declared fixed action order (or an explicitly defined
spike-time tie rule), never by an undisclosed analog readout. If all eligible
counts are zero, emit an explicit no-spike event and select an eligible hold
fallback if the catalog supplies one; otherwise report no supported decision.
If no actions are eligible, fail explicitly. A fallback is supervisory, not a
neural spike choice; report its frequency separately. Eligibility masks must
not erase the recorded raw spikes, so masked overrides remain auditable.

Train binary output spikes with surrogate-gradient backpropagation and weighted
cross entropy on accumulated counts. Keep gradients through the surrogate,
not through an integer cast. Detach the spike reset consistently with the
current trainer, clip gradients, and tune any count scaling or activity penalty
on validation episodes. Raw counts preserve a useful logit range; dividing by
the window length changes loss scaling and must be deliberate. Avoid a strong
spike penalty that rewards silent outputs. Log per-output firing, no-spike/tie
rates and class confusion, not just loss.

The official snnTorch count-loss implementation explicitly sums output spikes
before cross entropy. Its tutorial describes output count and latency decoding.
These are software/computational learning examples, not evidence for physical
Go2 safety or performance. Source code is MIT (documentation has separate
CC BY-SA terms); using the concepts does not require adding the library.
[Count loss](https://snntorch.readthedocs.io/en/latest/snntorch.functional.html),
[training and decoding tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html),
[source/license](https://github.com/jeshraghian/snntorch).

## Memory and evaluation

If data contains genuine consecutive observations, preserve chronological
hidden state during training and inference, reset at episode/identity changes,
and use documented burn-in and truncated-BPTT boundaries. Keep per-decision
counts separate from retained membranes; explicitly decide whether output
membrane also persists. A sequence is multiple sensor frames, not 16 copies
of one observation. If only independent decision rows are collected, deliver
a reset-window trained SNN and state that sequence learning remains untested.

The retained/reset memory comparison must use the same skills, data and output
decoder. Adding output neurons, new skills and a new training set simultaneously
cannot isolate the value of memory. Retain rules and the legacy SNN as matched
comparators only where their action menus are compatible; otherwise label menu
differences and include an expanded-menu rule comparator.

Acceptance checks: disjoint encounter splits; no future-outcome columns in the
input allowlist; every labeled action eligible; binary output raster shape
`[steps, number_of_skills]`; exact count-to-choice decoding; tested ties,
silence and no-eligible-action behavior; training/NumPy spike/count/choice parity;
actual independent run/jump physics outcomes; and closed-loop held-out contact,
fall, timeout, success, progress and latency metrics. Inspect failed classes
rather than hiding them in aggregate accuracy. Training completion, teacher
agreement and end-to-end navigation success are distinct results.

Open integration decisions: final skill IDs and capability limits, collection
episode count per class, whether labels are demonstrator or counterfactual,
available history length, and safe eligible no-spike fallback. Freeze these
before data collection and report any deviations in the measured report.
