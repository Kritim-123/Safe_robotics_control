# Obstacle jump development contract

One agent owns the new obstacle-jump skill, runner, tests and evidence. Existing
smooth-hop defaults, old datasets and SNN/association models stay unchanged.
This milestone develops a callable motor skill in MuJoCo, not a new learned
selector output. Existing velocity policies and classical trajectories are
inspected as possible motor sources before integration.

Use a static narrow low box perpendicular to travel, full width across the
robot's path. Accept only an actual crossing: all four foot collision spheres
pass the bar above its top, there is a shared verified airborne interval,
no robot geom contacts the bar, all four feet finish beyond its far edge,
and the robot settles for .8 s with linear speed <.04 m/s, angular speed <.2
rad/s, tilt <12 degrees and four supported feet. A hop beside a bar or with
rear feet left behind fails. Record contact events at every 2 ms physics step,
and foot positions, joint torques, base posture and trajectory.

Observations are privileged MuJoCo geometry in meters/radians/seconds. The
target must be an explicitly named static low box within the skill's validated
height/depth/distance/heading envelope. Other obstacles and walls retain normal
eligibility and clearance checks. The target's physical contacts always fail;
its 2D footprint may be exempted only for this explicit 3D crossing contract.
Reject incompatible targets rather than silently disabling collision guards.
Preserve finite-state, height, 35-degree tilt, timeout and torque cleanup guards.
Preparation interruption returns without launching; after thrust, finish the
landing trajectory and report interruption. Any run-up must be controller-driven
in persistent physics and explicitly logged. No external launch forces, velocity
injections, intermediate qpos assignments, or invisible lifting of obstacles.

First measure longer-jump candidates on flat development scenes, then develop
against a low bar and freeze one profile before fresh obstacle/heading/friction
variations. Record failed candidates. Update actual PID and MuJoCo force/control
ranges consistently for any explicit torque-bound experiment, under the user's
previous simulation torque authorization. State the actual bound; it is not a
hardware rating. Produce a finite video showing the complete crossing, run
focused tests and the full suite for shared controller changes, and retain
measured limits. If the target cannot be cleared, report failure honestly.
