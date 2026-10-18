# V5 walk-up low-bar contract

User-authorized milestone: walk toward a low obstacle, decide when to stop,
select the jump without a jump request, recover and reach the goal in one
persistent MuJoCo episode, test it, and open the viewer.

One writer owns `experiments/obstacle_v5/`, `tests/test_obstacle_v5.py`, V5
model/evidence files and reports. V3/V4 and the existing motor profile remain
unchanged. No delegation. Initial scope remains flat ground, +X travel,
static aligned thin bars 2–4 cm high and friction .8–.9.

Walk-up uses current obstacle range and measured motion to brake into a
standing launch region. No scheduled jump time, explicit jump task, hidden
teleport, pose reset or injected velocity is allowed. Any broadened launch
region is a new development envelope and must be assessed, not represented
as covered by the old 23–24 cm initial-pose tests. One persistent robot and
PID own all skills. Torque/force limits remain 24 Nm walking and the existing
72 Nm low-bar profile. Full foot/body crossing, physical contact, finite-state,
posture, stable-recovery, interruption and torque-release checks remain.

Observation contract: the V4 numerical simulator geometry/state features in
metres, seconds and radians, measured speed and support. The mission stays
progress. The SNN jump decision is evaluated on fresh packets after braking;
approach distance control is classical, and must be reported as such. A new
outcome association bank may be trained from walking-to-launch trials using
the same reset LIF encoder. Any teacher-forced jump collection is marked as
training data, not autonomous validation. Training, validation and held-out
world seeds remain disjoint; freeze before final assessment.

Acceptance: nominal episode makes >=.4 m measured walking progress before
jump initiation, brakes to four supported/stopped feet, independently selects
a jump from counts, verifies full crossing, resumes walking and stops <=.12 m
from its goal. Record jump entry speed, stance, distance, physical continuity,
torque and all failures. Assess fresh bar distances/dimensions/friction and
controls for tall/moving targets, blocked landing and clear path. Verify the
learned head is necessary, guard vetoes, no intermediate plant reset, and
run the full relevant project suite before the finite live demo.
