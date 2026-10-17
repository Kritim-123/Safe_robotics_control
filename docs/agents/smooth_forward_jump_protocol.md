# Smooth forward jump protocol

User explicitly requested smoother jumping, a little forward distance, and
removal of the old torque restriction. One agent owns `skills/jump_profile.py`,
the opt-in extension in `skills/dynamic.py`, tuning/evaluation runner, tests and
reports, and the updated learned-model demo integration. Existing unrelated
workspace changes and historical models/data remain intact.

Replace discontinuous joint-target jumps with piecewise quintic segments whose
position, velocity and acceleration agree at boundaries. Add a backward foot
push through a thigh offset during extension to impart forward body momentum.
Use a retract/absorb/recover sequence and test landing gains. No external force,
teleport, qpos correction, velocity injection, or hidden walking approach.

The old 24 N m cap is not a tuning restriction for this simulation milestone.
Test explicit finite motor bounds, updating the actual MuJoCo force/control
ranges and PID consistently. Report applied torque, saturation and physical
responses. Preserve finite-state, posture, collision and clearance guards.
Old wait/run and legacy-hop behavior keep their original configuration.

Keep the legacy hop available as the paired baseline. The new forward-hop
profile checks the projected takeoff/landing corridor before entry. Preserve
initial settling, continuous physics, timeout, stable recovery, and torque
cleanup on errors or viewer closure. An interruption before thrust finishes
preparation then returns smoothly without launching; after thrust begins it
finishes the landing sequence and reports interrupted.

Tune on a declared flat development scene, recording every candidate including
failures. Select a candidate with verified flight >=.06 s, stable recovery,
no collision/fall, heading-relative total progress >=.10 m and positive flight
progress. Prefer target distance .15-.40 m and lower measured landing impact
and command discontinuity. Measure target-step/velocity-step, torque slew,
base acceleration, contact force, airborne travel, peak height and recovery.
Do not describe greater distance alone as smoothness.

Freeze the chosen configuration before fresh yaw/friction/start variations.
Evaluate paired old/new profiles and held-out contact/terrain stress separately.
Retain all unsuccessful cases and report the validation range. Integrate only
after physical evidence; keep historical outcome collection pinned to its legacy
primitive and make the updated demo's profile explicit. Recheck learned model
eligibility, refusal, interruption and handoffs, plus the full regression suite.
No SNN weight or association fitting is part of this motor-skill milestone.
