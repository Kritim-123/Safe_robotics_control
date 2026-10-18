# V5 reliability refinement

User authorized fixing four learned refusals and the rear-thigh jump contact,
retesting and showing MuJoCo. One writer owns `experiments/obstacle_v5_refinement/`,
its new tests/model/profile/report and root status additions. V3, V4, original
V5 code/models and original 72 Nm jump profile remain unchanged. No delegation.

Use the existing observation/skill contract: flat ground, aligned static +X
thin bars, measured range and robot stability, no explicit jump request,
one physics plant/PID with an initial reset only. Approach/braking remains
classical; jump selection is learned from fresh reset LIF counts. Preserve
24 Nm walk/brake and 72 Nm jump bounds and all physical contact, foot/body
crossing, stable recovery, interruption and torque cleanup guards.

Known cases 02/04/06/08/14 from the original V5 test cohort are now development
regressions, not fresh evidence. First diagnose nearest count associations
and the landing contact. Develop a separate trajectory/profile with more
clearance, using actual full-episode physics outcomes. Then collect outcome
associations for that profile. Candidate changes are allowed on development
and validation only; freeze before the new final test cohort.

Proposed observation extension: preserve the V4 encoder's original 168 cells
and add finer level-sensitive LIF cells for range, height, depth, friction and
current posture/motion. If used, pitch is radians, body height metres and
angular speed radians/s, derived only from the current robot state. No scene
IDs, outcomes, teacher request or episode clock enters the head. Distinguish
teacher-forced outcome collection from autonomous assessment.

Acceptance: the five requested cases complete autonomous full crossings and
goal stops; preserve prior successes and negative-control refusals; compare
original and refined controllers on >=40 untouched paired worlds with the
same distribution. Report all failures, contacts, falls, refusals and
regressions. Recompute saved SNN packets and check source/model freezes;
run the full project suite and launch a finite passing live replay. Restricted
simulation evidence does not establish hardware safety or SNN superiority.
