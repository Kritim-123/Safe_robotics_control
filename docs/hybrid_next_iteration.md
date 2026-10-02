# Active-skill replanning and remaining work

The first prototype is implemented and measured. The most concrete remaining
control failure is reproducible with:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.run --scenario late_crossing --seed 600
```

The SNN chooses an initially clear bypass. A second cylinder enters that path;
the clearance guard ends the episode at about 27.7 s. The goal is not reached.
This is the next feature to build, rather than interpreting the main batch's
40/40 successes as general obstacle avoidance. An experimental implementation
is now available with `--replan`; it monitors moving hazards on the current
skill segment, invokes a fresh masked selection, and keeps physics continuous.
The original default remains available for the controlled comparison.

## Implemented first step and remaining extensions

The optional mode adds preemption, 0.2 s conflict confirmation, a one-second
minimum skill dwell, obstacle-specific wait completion and a 12-intervention
limit. It also extends the wait timeout to 60 s. These are explicit protocol
changes. It does not yet time-parameterize an entire remaining route or measure
stopping distances. The following list distinguishes that first implementation
from the broader work still needed:

1. In `hybrid/perception.py`, evaluate the **remaining active skill route**, not
   only the original goal ray. Use both obstacle velocity and an estimate of
   robot travel time. The current `path_clearance` checks geometry at one
   instant, so it cannot establish that a moving obstacle leaves the route clear.
2. `hybrid/system.py` now has a bounded preemption/replanning transition when
   that route becomes unsuitable. Preserve physics, gait phase and the single
   torque writer. Do not invoke standalone `walk()` or `navigate()` inside an
   episode; those APIs intentionally create new simulations.
3. The SNN receives a fresh encounter and mask when a skill is preempted. A stopped
   pose is not automatically safe: an approaching obstacle can hit a waiting
   robot. Measure the gait's stopping delay before choosing a predictive margin.
   If no option is applicable, report failure rather than silently resuming
   motion or inventing a successful recovery.
4. Events record the interrupted skill, obstacle identity, conflict reason, new scores,
   masks and replacement. The intervention count is bounded; further tests should probe
   hysteresis against rapidly changing observations without suppressing an
   urgent stop. Return to ordinary goal tracking only after an actual successful
   recovery.

Acceptance should include the known seeds 600–602, a new untouched seed group,
and the existing static/crossing/heading/noise batches. Record all contacts,
clearance stops, timeouts and goal successes. A longer episode timeout should
be declared as a changed protocol, not used to hide new failures.

## Learning and sensing follow-up

The current SNN imitates explicit geometric scores. Its successful deployment
does not show an advantage over those scores. To test learned selection more
meaningfully, collect **outcomes of candidate skills** from simulator rollouts:
progress, contact, fall, duration and completion. Use separate training,
validation and test scene generators. Compare the learned selector with the
same classical planner/teacher under the same observations and masks.

Include broad heading variation in training. The current synthetic generator
uses heading errors within ±0.3 rad; executed back-off maneuvers can produce
larger errors. Keep observation normalization and feature/action ordering
versioned with every checkpoint. Revalidate NumPy export after retraining.

Replace privileged obstacle state with a detector/tracker interface only after
the control loop handles preemption. The current optional sensor adapter has
known obstacle identities, exact radii, noisy velocities and buffered delay;
it does not model missed detections or association errors. Keep a clean-sensing
run as a diagnostic control when evaluating that future perception pipeline.

For the neuromorphic study, report SNN architecture, binary raster, temporal
encoding, accuracy and measured CPU latency separately. An energy-efficiency
claim would need a defined hardware measurement and a matched nonspiking
comparison. Spike counts alone do not establish energy savings.
