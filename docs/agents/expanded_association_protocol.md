# Expanded outcome learning protocol

October 16, 2026. User authorized more tests, collecting skill outcomes, and
learning associations. One agent owns the new outcome-learning dataset/runner,
terrain encoder, tests and reports, and small generic encoder/recorder changes.
Preserve other workspace changes, old artifacts, motor targets and guards.

Fix before collection: 96 new contexts, comprising the 60-level curriculum
with fresh seed 20261019 and 36 flat/obstacle contexts with seed 20261020.
Terrain variants 0-2 train, 3 validate, 4-5 held out. Four flat families have
nine variants each: 0-5 train, 6 validate, 7-8 held out. This yields 54 training,
14 validation and 28 held-out contexts. Collect matched wait/run/jump candidates
only for training and validation: 204 new attempts. Preserve every failed start,
guard stop and partial result. Terrain skill duration 8 s, timeout 18 s; other
skills 4 s, timeout 12 s. Initial settling and stable recovery remain unchanged.

Reuse the original 40 training associations with their saved causal observations
and hashed trial evidence; no old validation or held-out labels are added.
Three frozen selectors will be compared: original 40-example bank, expanded
legacy-input bank, expanded terrain-input bank. Neighborhood remains three
unanimous neighbors, distance <=4 with all cutoff ties included. No cutoff,
weights, bias, scale, scene or label-policy tuning using validation or held-out
outcomes. Conflicting exact vectors abstain and remain recorded.

The terrain encoder preserves all original 14 input columns and biases, and
appends eight random columns initialized once with seed 4242, standard deviation
1/sqrt(8). Keep 16 LIF outputs, 16 window steps, beta .9, threshold 1 and zero
state per observation. New inputs: terrain present; maximum height rise; largest
up/down sampled height changes; average forward and cross slope; second-difference
roughness; and friction loss relative to floor. Sample current static support at
forward distances 0,.2,.4,.6,.8,1,1.2,1.4 m and lateral offsets -.2,0,.2 m,
in the robot's current heading. Units and scales are explicit in terrain_encoding.py.
This is privileged simulator-map sensing, not camera perception. Future outcomes,
scene names and task labels do not become observation inputs. Flat terrain extras
are zero, preserving the original neural response for the same original inputs.

Labels come from actual outcomes: require no collision/fall, torque within limits
and primitive completion. Terrain tasks additionally require final X>=1.55 m,
absolute Y<=.35 m. Flat forward task requires >.1 m heading-relative progress.
Requested in-place hop requires >=.04 s verified flight and stable landing.
Only the matching task's skill competes; an eligible successful wait is a zero
utility fallback, never task fulfillment. Unrequested hopping does not count as
terrain traversal. Keep unlabeled cases if no candidate succeeds.

Freeze both banks before validation scoring and physical held-out evaluation.
Evaluate all three selectors on the same 28 held-out cases, plus a direct-run
diagnostic control. Direct-run controls are not oracle labels for requested hops.
Also compare original and updated models on twelve old, previously exposed
contexts (variants 6,7 of the first six original families); call this regression,
not independent generalization. No fitting occurs during either evaluation.

Acceptance: actual learned artifacts with provenance and training-only examples;
saved comparisons of coverage, eligibility/fallback, primitive/task success,
contact/fall/torque and count aliasing; candidate failures retained; relevant
and full regression tests; reproducible commands. Report unsuccessful or
regressed behavior, without retroactively tuning on test data. Promote a new
default only if held-out task fulfillment improves without a new collision/fall
or a loss of fulfilled regression tasks. Otherwise keep it explicitly selectable.
Do not claim a trained jump-over-obstacle skill or trained SNN weights.
