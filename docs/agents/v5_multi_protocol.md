# V5 repeated-jump experiment

Use the refined V5 motor trajectory, 264-cell encoder and V3 goal model in
one persistent physics episode. The new runner owns repeated target selection,
per-jump crossing records and rearming after a stable verified landing.
Original V4/V5/refinement sources and artifacts remain frozen.

Observations retain the sixteen refined V5 features and their units, including
the actual final-goal distance, current geometry, posture and measured speeds.
Only verified passed obstacles leave the decision observation; all collision
geometry remains active. No explicit jump request is supplied. The final goal
is fixed for the entire episode. Fresh spike packets and two confirmed learned
proposals are required for each jump, with the existing launch eligibility and
landing-corridor checks. Approach correction controls lateral/yaw drift while
preserving the strict launch guard, 24 Nm walking and 72 Nm jumping limits.

Baseline: run the frozen refined head in a three-bar course and retain failures.
Development: separately flagged teacher proposals collect physical jump outcomes
for longer goals and visible subsequent bars. Keep the seeded encoder and motor
profile unchanged; extend a separate association bank with measured packets.
Do not use assessment outcomes to fit that bank. No teacher is allowed in final
assessment or viewer runs.

Assess progressively: two, three and four aligned low bars, varied spacing,
height and friction. Record all attempted courses, contacts, falls, refusals,
actual jump entries, full-body crossing margins, skill transition times and
goal stopping. Exercise an unsuitable later bar, blocked landing corridor,
forced-wait neural output, viewer closure and interruption. Verify only one
robot placement/reset, continuous recorded positions and released torques.
Acceptance for the demonstration: at least three autonomous verified crossings,
stable goal arrival, no collision/fall and no intermediate robot reset. Run
the project suite after integration. Report failures as well as the demo.
