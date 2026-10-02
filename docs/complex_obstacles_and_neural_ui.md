# Complex obstacles, faster reactions and the neural control observatory

## Run the demonstration

In WSL from the project directory:

```bash
./.venv-win/Scripts/python.exe -m hybrid.live
```

Open http://127.0.0.1:8766. Choose a course and reaction profile, then Start.
PowerShell uses `.\.venv-win\Scripts\python.exe`. Camera can follow the robot
or show the whole course; the map always shows the arena. Camera changes take
effect on the next physics frame. Pause/Resume preserves the episode state.
Stop ends and saves the episode. The default command limit remains 0.24 m/s
in the dashboard, with nominal 2x viewer playback.

Run without graphics:

```bash
./.venv-win/Scripts/python.exe -m hybrid.run --scenario mixed_course --reaction fast --max-speed 0.24 --replan --headless --brain --output output/hybrid/mixed-course
```

## New scenarios and physical shapes

| Scenario | Added challenge |
| --- | --- |
| `crate` | Rotated rectangular crate |
| `moving_gate` | Sliding rectangular barrier |
| `oscillating_gate` | Barrier moving sinusoidally and reversing direction |
| `mixed_course` | 6.5 m goal, rotated crate, barrel and delayed moving barrier |
| `late_crossing` | Existing encounter that interrupts an active bypass |

Crates are physical MuJoCo boxes, not recolored cylinders. `Obstacle` now
supports `shape`, `half_extents` (horizontal half-widths), `height`, `yaw`,
`motion`, `amplitude`, `period` and `start_after`. Existing JSON configurations
continue to work. Oscillating motion uses the derivative of its position for
sensed velocity and freezes both position and velocity at the configured stop.

Planning encloses a box in its circumscribed circle, radius
`hypot(half_extents[0], half_extents[1])`. This contains its corners at any yaw,
but may unnecessarily reject narrow passages. The trained SNN still receives
the same 12-feature schema, with effective planning radius as the radius input.
It has not been trained to recognize object classes. The four skills remain
bypass left, bypass right, wait and back off; none can jump, climb or step over
terrain. The predictor extrapolates current velocity, not the future motion
script, so reversal remains a real prediction limitation.

## What became faster

The SNN was already sub-millisecond in these local runs. Two separate changes
address the controller around it:

1. Vectorized path-clearance, waiting-clearance and threat-prediction geometry.
   The same seven prediction samples and safety thresholds remain in use.
2. Optional `--reaction fast`: scene checks every 20 ms (50 Hz), versus 100 ms
   (10 Hz); persistent moving-route conflicts confirmed after 80 ms instead
   of 200 ms. The existing one-second minimum active-skill age before
   preemption and other recovery/settling rules are unchanged.

Standard remains the CLI default. Fast is the dashboard default. Physics
stays at 2 ms and joint target updates at 10 ms. The SNN keeps its 16 internal
steps and exact trained weights. Faster polling is not faster learning.
Sensor delay, filtering, gait dynamics and confirmation still affect actual
response; these settings do not promise a hard real-time deadline.

Local CPU benchmark (identical inputs and weights, best of five batches):

| Obstacles | Threat check before → after | Feature + SNN + mask before → after |
| --- | --- | --- |
| 3 | 0.137 → 0.024 ms (5.8x) | 0.275 → 0.247 ms (1.11x) |
| 12 | 0.551 → 0.031 ms (17.5x) | 0.641 → 0.260 ms (2.46x) |

Each batch ran 500 threat checks or 200 complete decisions. These are
microbenchmarks, not guaranteed latency under UI/rendering load. The dashboard
shows measured SNN-only and full selection latency separately from sampling
and confirmation intervals. With seed 0, late-crossing selected wait at 15.22 s
in fast mode versus 15.60 s in standard mode. This is an observed episode
comparison, not a universal 380 ms improvement: changing command sampling can
also change the robot's trajectory.

## What the improved visuals demonstrate

- **Physics and motor commands:** follow/overview camera, measured motion,
  torque, slewed body commands, and the active control stage.
- **Planning map:** real robot trail and obstacle positions, conservative
  circles, the goal, current waypoint and saved candidate skill paths.
- **Decision panel:** detected obstacle, raw scores, mask eligibility, selected
  skill, decision age and separate compute/reaction timings.
- **Neural circuit:** all 12 inputs, 64 first-layer and 32 second-layer neurons,
  and four outputs. Bright neurons actually fired at the selected internal
  step. Layer connectors are structural summaries, not individual synapses.
- **Spike and membrane traces:** replay the 16-step computation slowly or
  inspect one step and any neuron. No activity is invented between decisions.
- **Readout evidence:** for the selected skill and best other allowed skill,
  neuron `j` contributes `(w_selected[j] - w_alternative[j]) * spike_rate[j]`.
  Contributions plus the bias difference sum to the final score margin.
  The display groups the other 26 neurons after the six strongest bars. If no
  other skill is allowed, it shows the selected raw score decomposition.
- **Input sensitivity:** one-at-a-time perturbations ranked by score response,
  including exact recorded feature values and units.
- **Timeline:** actual state transitions, skill changes, recovery and completion.

Neuron contributions exactly explain the final linear readout, but are not
semantic interpretations of neurons or an additive attribution to inputs.
The SNN is a software decision component; these displays do not establish
biological cognition. It selects skills; the classical gait/PID drives motors.

Large neural traces are fetched only when the decision ID changes, instead
of on every video poll. Frame metadata includes that ID; the browser rejects
an asynchronously fetched trace belonging to a different decision. The
inspection sensitivity calculations are observational and never choose actions.

## Validation

- Five scenarios × two profiles at seed 0: **10/10 successful**.
- Fast mode, five scenarios × additional seeds 1 and 2: **10/10 successful**.
- Fast mode, mixed and reversing courses at seeds 10 and 11, with position
  noise 0.025 m, velocity noise 0.005 m/s, delay 0.1 s and smoothing tau 0.2 s:
  **4/4 successful**.
- No collision or fall in these 24 runs. The live mixed-course run also
  reached its 6.5 m goal with 6.65 cm error, traveling 9.71 m.
- 43 tests passed together, followed by the two live tests after adding one
  new API test: **44 distinct passing tests**. New checks cover physical box
  dimensions, conservative bounds, motion derivatives/stopping, randomized
  scalar-versus-vectorized geometry, preserved timesteps, exact evidence sums,
  control validation and separate trace delivery.

These are targeted simulation checks, not a general safety guarantee or
evidence that the SNN beats the existing rule selector. New obstacle families
reuse the trained selector; broader generalization requires more evaluation.
Per-run records and benchmark values are in `output/hybrid/challenge-check`.
Earlier evaluation snapshots remain historical and were not overwritten.
