# Improving Go2 jumping: research and proposed experiments

October 16, 2026. Research and local code/data review only; no controller changes
or new jumping trials were performed for this report. Work was done in one thread.

**Recommendation:** develop a feedback-controlled, torque-bounded jump from our
working hop; audit the imported policy separately. Then use curriculum learning
on our exact model if the classical controller cannot meet the required range.
Keep the SNN responsible for skill selection and the motor controller responsible
for takeoff, flight and landing.

## What the local evidence establishes

The existing hop achieved 0.124 s continuous flight and 0.078 m base-height rise
from its settled pose, with about 0.068 m displacement predominantly backward.
It has not cleared an obstacle. Base-height rise includes leg extension and is
not the same as ballistic center-of-mass rise after liftoff. See
[existing measured results](dynamic_skills.md).

The imported forward-gap actor completed 0/7 gap conditions; its flat walking
control traversed and settled. These solve different tasks, so the counts do not
rank the two controllers. All trials reached 24 Nm, but peak torque alone cannot
identify the cause of failure. Source training allows 45 Nm at calf joints;
model/contact dynamics and handoff semantics also differ. See
[transfer results](pretrained_jump.md).

Re-reading saved terminal states shows four gap trials ended above 35 degrees
tilt (.12 m at commands 1 and 2; .20 m at 1; .30 m at 1.5). The .30 m / 2 trial
ended below 0.15 m base height. The remaining two have neither threshold exceeded
in their terminal state, consistent with the code's non-foot contact condition.
The combined failure reason does not retain contact identities, so those contact
cases need replay/instrumentation before detailed diagnosis. This is analysis of
existing `output/pretrained_jump/evaluation/*/observations.npz`, not new trials.

## Specific changes worth testing

| Priority | Local finding | Proposed experiment | Evidence needed |
|---|---|---|---|
| 1 | Hop crouch switches abruptly to extension at 0.5 s | Smooth bounded takeoff trajectory with explicit desired velocity; compare to current hop | Takeoff speed, saturation duration, pitch, landing stability |
| 1 | Hop phases are entirely elapsed-time based | Contact-aware crouch/push/flight/landing/recovery states, each with a timeout | Actual contact transitions, no premature standing command |
| 1 | Imported handoff occurs at fixed x; return occurs when base passes gap + .45 m | Retain jump controller through detected touchdown and recovery; separately compare upstream value-filter engagement | Handoff state distribution, crossing and landing rates |
| 1 | Failure trace merges contact/height/tilt and saves only cumulative peak torque | Separate failure causes; log per-joint requested/applied torque, contacts and forces at physics rate | Which joint saturates, for how long, and which body first contacts terrain |
| 2 | Imported actor was trained on another dynamics/control contract | Verify observations/actions against source; adapt or retrain under our limits | Matched-state inference tests and controlled transfer ablations |
| 3 | Current hop has little forward range | Optimize front/rear push timing and landing placement for short targets, then train progressively | Clean flight and stable landing in a target region |

The target discontinuity is concrete: the thigh changes from 1.1 to .45 rad,
and the calf from -2.2 to -.9 rad. Dividing by the existing .01 s target period
creates approximately -65 and +130 rad/s desired velocities. With Kd=3, their
velocity-feedforward contributions alone are approximately -195 and +390 Nm
before measured-velocity feedback, position feedback and clipping. Actual applied
torque remains bounded at 24 Nm. Smoothing is a hypothesis to test: excessive
smoothing could reduce the useful launch impulse, so it is not automatically an
improvement.

For landing, let the legs absorb energy with a tuned compliant response and
restore standing gradually after contact. Do not demand rigid tracking of the
flight pose after a foot lands. A trajectory-optimization plus RL study explicitly
relaxes reference tracking during landing to improve terrain adaptation.
[Wang et al., 2025](https://arxiv.org/html/2509.12776v1).

## What the external research supports

**Curriculum learning is the strongest training template.** Atanassov et al.
progress from vertical jumping to directional jumping and then obstacles. Their
repository trains Go1 with an older Isaac Gym stack; it is useful as a method,
not a compatible Go2 checkpoint. Randomization includes dynamics, motor strength,
latency and terrain properties. Start with narrow ranges appropriate to our model,
then expand after reliable nominal behavior.
[Paper](https://arxiv.org/html/2401.16337v2),
[author code](https://github.com/vassil-atn/Curriculum-Quadruped-Jumping-DRL).

**Smooth trajectories plus force control offer a smaller first step than full RL.**
Yue et al. combine minimum-jerk trajectories, virtual-model control and a QP with
whole-body landing control on a modified Mini Cheetah. This motivates testing
smooth references here; it does not establish that smoothing our PID targets alone
will reproduce their results. A later force-based controller must satisfy torque,
joint, friction and unilateral-contact constraints, not just a desired height.
[Paper](https://arxiv.org/abs/2407.00658).

**The existing pretrained candidate needs a faithful transfer audit.** Its public
package defines 50 Hz control, action scaling, observation history and higher calf
torque capacity. The vendored `Go2ValueFilter` also contains a landing latch; our
adapter currently bypasses this logic. Restore or reproduce its contact-aware
handoff behavior as an ablation, while keeping local guards. Its learned value
function is not a safety certificate for our changed dynamics.
[Source package](https://github.com/SafeRoboticsLab/go2_atomic_skills),
[pinned implementation](https://github.com/SafeRoboticsLab/go2_atomic_skills/blob/ce0169d179525c72c712df8e62475e2a0536d124/go2_atomic_skills/skills.py).

The landing issue cannot explain all current failures: none of the seven trials
reached the adapter's far-side handoff position. Fixing landing is necessary for
a complete controller but takeoff/approach diagnosis comes first.

## Recommended development sequence

1. **Instrument before tuning.** Record phase, actual base velocity, angular
   velocity, joint state, desired/applied torque, saturation fraction, per-foot
   contact/force, terrain-relative foot clearance, takeoff state, touchdown state,
   exact stop reason and checkpoint/configuration hash. Verify that the 50 Hz
   command/5-frame history and joint order match source expectations.
2. **Improve the existing hop.** Use six phases: crouch, push, flight, landing,
   recovery and complete. Contact measurements govern transitions with bounded
   timeouts. Search a small number of trajectory durations, crouch depth,
   front/rear timing and landing gains under the existing 24 Nm ceiling.
3. **Measure a feasible range.** First target a short forward displacement on
   continuous ground; only then introduce low bars or small gaps. Condition on
   landing target, obstacle geometry and actual approach velocity. Require clearance
   of the body and rear feet, not just the base crossing an x coordinate.
4. **Train if needed.** Use PPO with a progression from vertical hop to short
   forward jump to obstacle clearance. Train on the local Go2 model and actuator
   limits. A bounded residual over a verified trajectory is an alternative to a
   completely new actor. Reward landing accuracy, clearance and settled recovery;
   penalize collisions, saturation duration, excessive impact and action changes.
   Do not optimize airtime or height alone. A CSV of outcomes is useful for a
   feasibility model/selector; it is insufficient by itself to train the entire
   low-level motor controller without state/action trajectories and interaction.
5. **Expose validated capability to the SNN.** Preserve `jump` as one skill output
   with a parameterized target, or introduce separately versioned hop/gap-jump
   identifiers. Add gap width, near-edge distance, obstacle height, landing-area
   clearance, actual speed and skill readiness to observations. Changing features
   requires retraining the selector. Once airborne, latch execution until landing
   recovery; output spikes must not cause arbitrary midflight switching.

## Proposed acceptance criteria (not achieved results)

Freeze the baseline and tuned candidates before evaluation. Use at least 100
held-out episodes with independently varied initial pose, approach state and
terrain, plus separate stress conditions. Reserve entire parameter combinations
for testing and report each condition as well as overall rate and uncertainty.
Repeating deterministic scenes with different seed labels is not independent evidence.

Initially aim for at least 95/100 successful flat hops and 90/100 short forward
target landings, with zero body/obstacle contacts and no torque-limit violations.
For hops, require at least 40 ms verified all-feet flight, continuous terrain-relative
clearance, and 0.8 s stable four-foot recovery; report median and worst-case drift,
pitch, saturation fraction and impact. For crossings add whole-body/foot clearance
and landing-target accuracy. Match approach conditions when comparing gap actors.
Keep the 35-degree guard for initial tests and treat any future guard revision as
a separate reviewed experiment, never a way to relabel failed trials successful.

No external paper or current trace establishes that 24 Nm makes useful jumping
impossible. The existing hop already disproves that broad claim. The open question
is what forward range and robustness this particular model can achieve within it.
