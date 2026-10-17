# Gradual terrain curriculum protocol

October 16, 2026. User requested more terrain and environments with gradually
increasing complexity. One agent owns `hybrid/terrain.py`, the optional terrain
extension in `hybrid/arena.py`, experiment curriculum/runner, tests and this
milestone's reports. Existing user edits and policy artifacts are preserved.

Create ten ordered levels with six reproducible variants each: flat, thresholds,
ramps, low steps, uneven blocks, cross-slope, friction transitions, static passage,
moving crossing and mixed course. Variants increase the family's principal
difficulty and add seeded placement/height variation. Stage order is a proposed
curriculum, not a claim that measured failure probability increases monotonically.
Development variants 0-2, validation 3, held-out 4-5; seed 20261017. Freeze geometry
before rollout. No outcomes tune the geometry, skills or association bank.

Terrain is static collidable support, distinct from hazardous obstacles. Distances
are metres, time seconds, angles radians, friction dimensionless. Boxes form
thresholds/steps/blocks, convex wedges form ramps. The original plane remains
underneath: this milestone has no holes or gaps. Terrain-foot contact is allowed;
terrain contact by other robot geoms is a collision failure. Patch contact priority
must override the foot default, with the original six contact dimensions. Test
actual friction in foot contacts, not merely XML attributes.
Preserve the foot solver softness when changing contact priority. The six flat
variants intentionally repeat an identical physical control condition; they
are not six independent terrain geometries.

The existing 14-input random SNN remains frozen and terrain-blind. Terrain data
is saved in scenario metadata and telemetry, not silently added to its inputs.
Use the frozen 16-count association bank and existing `dynamic/wait`,
`dynamic/run`, `dynamic/jump` interfaces. All curriculum tasks request forward
traversal; none provides a jump-request shortcut. Compare selected execution
against direct `dynamic/run` to distinguish selector abstention from motor-skill
limitations. Direct execution is a diagnostic baseline, not learned selection.

Run eight-second primitives, two-second initial settling and existing stable
recovery, with an 18-second episode timeout. Preserve torque limits, posture,
collision and clearance guards, eligibility, interruption and cleanup. One
selection per episode; fresh reset between episodes only. Traversal success
requires primitive completion, no collision/fall, torque within limits, final
world X >=1.55 m and absolute final Y <=.35 m. Start is (0,0), yaw zero. Count
waiting as primitive completion only. Record final position from the terminal
trajectory because failed initiation can use a different displacement origin.

Save live inputs/spikes, eligibility, proposed/actual skills, full scenario,
trajectory, physical outcomes and separate traversal success. Advance a stage
only if both of its validation/held-out set (variants 3-5) and development set
each have at least 80% traversal success and no collisions/falls. With three
cases per set this means all three must pass. Report locked later stages even
when diagnostic sweeps evaluate them. This is an assessment gate, not training.
`--progressive` enforces the gates per controller and stops advancing at its
first failed level. It requires all six variants and contiguous levels from zero.

Acceptance: all 60 models compile; deterministic splits and increasing family
parameters; real support and body-contact distinctions; contact friction;
scenario JSON round trip; meaningful rollouts in both modes; failures retained;
full regression suite and a rendered curriculum preview. Do not claim stair,
rough-terrain or general navigation competence from merely adding environments.
