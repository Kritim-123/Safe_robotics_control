# Classical skill catalog (October 15, 2026)

Inventories are distinct: two standalone callable entry points (`walk`,
`navigate`), four original persistent options, and two optional temporal options.
The saved selector has exactly four outputs (`bypass_left`, `bypass_right`,
`wait`, `back_off`). There is one learned selector checkpoint and no learned
locomotion policy. The optional catalog maps the wait/back-off action families
to new classical implementations; it does not train six output logits.

| Identifier | Execution and completion | Timeout |
|---|---|---|
| walk | Existing periodic foot trajectories and joint PID; fixed duration | Caller duration |
| navigate | Existing waypoint tracker, final distance <=.12 m, speed <.04 m/s, gait stopped, 1 s settling | 120 s default |
| bypass_left / bypass_right | Goal-relative lateral waypoint followed by forward waypoint; final error <.14 m | 100 s |
| wait | Zero velocity command; supervisor verifies clear route for .7 s (selected obstacle only in legacy replanning mode) | 25 s; 60 s with replanning |
| back_off | Goal-relative reverse .55 m waypoint; error <.14 m | 30 s |
| yield_hold | Zero velocity command; **all observed route threats** clear continuously for 1.5 s | 60 s |
| retreat_hold | Goal-relative reverse .55 m waypoint, then same all-threat hold; arrival alone is not completion | 60 s including travel |

All are classical controllers. Backward waypoints can rotate the robot; these
are not dedicated backward gaits. Standalone entry points create episodes;
persistent options never reset or step physics. Parameters and observations
use world XY metres, world obstacle velocity m/s, radius metres, simulator
seconds, and heading radians where needed. All assume supported flat terrain
and a stable upright robot. Existing legacy masks and guards remain unchanged.

New option contract lives in `skills/temporal.py`. `build_temporal_skills`
returns both options and eligibility booleans, using current position, nonzero
goal direction, obstacle tuples and ordered rectangular bounds. Current
inflated-footprint clearance must exceed .08 m. Yield also checks continuous
constant-velocity obstacle forecasts for three seconds; retreat checks its
entire current geometric segment. The singular factory rejects ineligible
initiation with `ValueError`. Invalid/nonfinite geometry, nonpositive durations,
or timeout <= hold duration also raise `ValueError`.

At each sensing tick call `update(position, now, route_clear)` with monotonic
time and a boolean reflecting **all** observed obstacles. A threat resets the
dwell. `holding` or `target(...) is None` means command zero through the
existing slew/gait/PID, never pass None to the waypoint tracker. Terminal
statuses are `complete`, `timeout`, and `interrupted`; timeout has priority at
the deadline. `interrupt()` is immediate and terminal for the option. It does
not imply the moving robot stops instantaneously. The supervisor must continue
dynamic clearance/contact/fall checks, then recover with zero commands until
gait activity is zero, planar speed <.04 m/s, angular speed <.2 rad/s and tilt
<12 degrees, with its existing .8 s minimum / 5 s deadline. A moving obstacle
may invalidate a safe initiation; these are local checks, not guarantees.

Fresh targeted evidence: `.venv-win/Scripts/python.exe -m unittest discover -s
tests -p test_expanded_skills.py -v` passed **8 tests in 2.150 s**. Six independent
MuJoCo executions cover both new options succeeding, timing out, and being
interrupted, followed by stable recovery. Assertions retain torque limits,
finite states, height/tilt/contact/clearance checks and position continuity.
Failed initiation is checked against unchanged MuJoCo qpos/qvel/time; boundary
tests cover rear obstacles/walls, approaching hazards, dwell reset, invalid
input and time reversal. These independent physical executions use clear flat
terrain; changing-obstacle integrated evidence belongs in the integration
report. No physical robot has been tested.

The original checkpoint provenance remains in
`skills/policies/selector_snn/manifest.json`: 12-64-32-4 LIF, 16-step encoding,
beta .9, threshold 1, supervised geometric teacher, train/validation/test data
seeds 17/10017/20017. No checkpoint, normalization, actuator interface or license
was changed by this skill expansion.
