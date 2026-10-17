# Targeted request-decoder refinement

Declared 2026-10-17 20:00 UTC. The V2 clean final test exposed ten additional
passage/crossing failures versus rule control, with false wait requests even
when run was eligible. These results are development evidence for this new
candidate, not its final confirmation. Keep all prior releases and runtime
sources unchanged. One agent owns the new refinement script and records.

Collect 20 new seed-15400 static-passage/moving-crossing training scenes under
V2 and rule control. Append causal raw-task/mask labels to the published base
request data. Do not use goal-latched teacher wait as a base-request label.
Preserve encoder arrays, alpha 0.001, goal count threshold 92, two-packet entry,
release threshold 25, two-packet release, classical gait and physical guards.
This is request-readout training, not motor training or counterfactual outcomes.

Validation: 20 new obstacle scenes from seed 15401 plus ten flat seed-15402 and
ten mixed seed-15403 scenes, paired candidate/V2/rule. Require at least two
additional boundary-scene successes over V2, no contact/fall/limit violations and no aggregate
regression on the 20 flat/mixed cases. Recheck the existing 160-scene request
regression: at least 150/160 with no contact/fall/torque/hold violations.
Also require at least 44/60 on the original course, as regression evidence.
Preserve every previously successful regression/original-course case, rather
than offsetting lost successes with gains elsewhere.

Only if qualified, freeze before fresh seed-15500 100-scene paired confirmation
and seed-15501 40-scene mild-noise confirmation. Use a 20:42 UTC batch cutoff
between complete matched groups. Incomplete checks mean the candidate remains
unqualified. Do not select or retune it on these confirmation results. If it
fails qualification, retain its evidence and recommend the existing frozen V2.

For retention after confirmation, require all 100 clean and 40 noisy matched
scenes complete, at least two additional clean successes over V2, no aggregate
loss on noisy goals, and no candidate contact/fall/limit/hold violations. Report
paired regressions. A failed retention gate leaves V2 as the recommended model;
it does not trigger another fit or parameter search on confirmation scenes.
