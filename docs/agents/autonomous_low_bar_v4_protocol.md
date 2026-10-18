# V4 observation-selected low-bar jump

Agreed scope: user authorized a separate revision that selects a low-obstacle
jump without an explicit request, tests it, and shows it in MuJoCo.

One writer owns `experiments/obstacle_v4/`, `tests/test_obstacle_v4.py`, this
protocol, its validation report and model artifacts. Existing V3 and low-bar
source/model files remain unchanged. No delegation or hardware deployment.

Observation contract: current simulated obstacle presence, forward/lateral
position, height, depth, lateral span, measured speed/yaw alignment, landing
corridor clearance, supported/stopped robot state, goal distance and friction.
Units are metres, seconds, radians and dimensionless friction/support flags.
No task jump bit, scene ID, elapsed episode time or future obstacle schedule
enters the learned jump head. These are privileged simulator observations,
not a camera pipeline. LIF membranes/counts reset for each 100 ms packet.

Skills: `wait`, V3 `run`/goal hold, and a separately bounded `low_bar_jump`.
The initial milestone starts in a stationary +X launch stance, with an aligned
static bar 23–24 cm ahead, 2–4 cm high, 1.2–2 cm deep, >=1.2 m wide,
flat ground and friction .8–.9. Walking up to a distant launch stance is
outside this first contract. A learned jump proposal requires two fresh
confirmations and supported/stopped posture. A separate eligibility veto
checks dimensions, heading, initial stance, all feet behind the bar and
landing corridor. It cannot cause a jump on its own.

During committed flight, finish the existing trajectory and monitor physical
contacts, finite state, clearance, height and tilt. A guard failure stops and
is reported as failure; it is not credited as recovery. Closing the viewer
releases torques. Crossing requires the complete existing foot/body test and
stable landing for .8 s. Only then may V3 walk to the goal, with the crossed
bar behind it. All collision geometry remains active. One initial placement
only; one PID writer; existing 72 Nm low-bar / 24 Nm walking bounds.

Learning: fixed seeded LIF observation encoder and nearest count-pattern
associations. Positive/failed jump labels come from actual standalone physics
trials using the frozen profile. Out-of-contract negative examples are
explicitly labelled eligibility examples, not successful physics outcomes.
Freeze the data/model before held-out scenarios. Report all failures, separate
neural proposals from guard vetoes, and retain raw logs under `output/`.

Acceptance: nominal autonomous full crossing followed by goal tracking and
supported stop in one episode; no explicit jump request; no jump in tall-bar,
moving-target, blocked-landing and no-obstacle controls; fresh low-bar tests
with honest crossing/goal/contact counts. Replay same observations/counts,
check guards and torque cleanup, then run the integrated project test suite.
Simulation results do not establish hardware safety or SNN superiority.
