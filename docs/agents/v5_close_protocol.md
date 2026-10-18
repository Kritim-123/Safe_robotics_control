# Close-spacing V5 validation contract

The milestone is autonomous completion of the two formerly refused three-bar
courses at 1.8 m and 2.0 m spacing, followed by separate held-out close courses
and regression of the previous wider courses. Each jump requires confirmed
learned proposals, existing launch eligibility, full crossing and stable landing.
The episode ends only after all bars are passed and the robot stops at its goal.
No contact, fall, overshoot, intermediate placement or explicit jump request is
allowed in a successful episode.

The single agent owns the new `experiments/obstacle_v5_close/` package, its
tests and close-spacing documentation/artifacts. Existing multi-obstacle and
older runtime/model sources remain frozen. This revision first tests the same
retained 444-row association model with no retraining or teacher proposals.

Replace the circumscribed-circle landing check with the existing point-to-box
distance and a 0.42 m robot disc. Preserve the 1.2 m corridor, 3 s horizon and
0.15 m clearance threshold. Subtract half the spatial sampling interval and
the maximum obstacle travel over half the temporal interval so a sampled path
cannot falsely imply a continuously clear path. Cylinders and arena boundaries
remain checked; moving and rotated other obstacles are included. Physical
contact/fall checks, interruption behavior and 24/72 Nm actuator bounds remain.
The `landing_clearance` observation must use the same corrected geometry; keep
all other feature definitions and the real final goal unchanged.

Assess the two development regressions; freeze sources/model before testing
12 new seed-23701 courses with 2/3/4 bars, spacing 1.7–2.1 m, heights 2–4 cm,
friction .8–.9 and first distance .8–1.15 m. Replay all nine previous wider
assessment worlds. Include a truly blocked landing, an overly close next bar,
a later tall bar, forced-wait readout and interruption/viewer-close checks.
Audit exact predictions, source hashes and continuous physics; run the broader
project suite. Retain all failures and launch a passing 1.8 m viewer course.
