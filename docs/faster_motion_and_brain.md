# Faster motion and visible SNN decisions

The saved `late_crossing` run reached its goal after 132.64 simulated seconds,
with 6.08 m of travel around obstacles and 6.4 cm final goal error. It made
three SNN decisions, including a deliberate wait from 20.8 to 46.6 seconds.
The robot is not learning during these episodes. The trained SNN selects
waypoint skills; the classical gait and joint PID execute them.

## Run farther and faster

From WSL/Bash in the project directory:

```bash
./.venv-win/Scripts/python.exe -m hybrid.run --config configs/hybrid_long_route.json --max-speed 0.24 --playback-speed 2 --replan --brain --output output/hybrid/long-route
```

For the moving-obstacle demonstration:

```bash
./.venv-win/Scripts/python.exe -m hybrid.run --scenario late_crossing --max-speed 0.24 --playback-speed 2 --replan --brain --output output/hybrid/faster-crossing
```

In PowerShell replace `./.venv-win/Scripts/python.exe` with
`.\.venv-win\Scripts\python.exe`. Add `--headless` to run without a viewer.

`--max-speed` changes the maximum body translation command. It is not measured
ground speed: foot slip, turning, acceleration and settling affect progress.
The default remains 0.18 m/s. `--playback-speed` changes viewer pacing only;
physics, sensing and control timesteps remain unchanged. Actual playback is
limited by computer performance. Video recording keeps its separate 4x setting.
The viewer frames the configured arena so the longer route remains visible.

## Measured comparison

Single-seed (seed 0) SNN runs with replanning enabled, seconds to completion:

| Command limit | Clear | Static obstacle | Late crossing |
| --- | ---: | ---: | ---: |
| 0.18 m/s | 64.04 | 94.34 | 132.64 |
| 0.24 m/s | 43.84 | 67.64 | 108.34 |
| 0.30 m/s | 36.04 | 56.14 | 98.44 |

All nine runs reached their goals without collision or falling. These are
targeted checks, not a replacement for the earlier multi-seed evaluation.
Start with 0.24; 0.30 was successful here but has not been broadly evaluated.
No gains, model weights or safety masks were changed.

The new six-metre goal with a static obstacle succeeded at 0.24 m/s in
101.24 simulated seconds, traveling 8.05 m and ending 6.5 cm from the goal.
Minimum footprint clearance was 0.145 m and peak torque was 22.14 Nm.
Records are in `output/hybrid/speed-check/` and `output/hybrid/long-route/`.

## Inspect the robot's decision signals

`--brain` generates `brain/index.html` inside the run directory after the
episode. Open that file in a browser. It is an offline report, not a live
neural monitor. To inspect an existing run:

```bash
./.venv-win/Scripts/python.exe -m hybrid.brain --run output/hybrid/latest
```

The report includes:

- Actual goal distance, measured speed, peak absolute joint torque and, for
  newly recorded runs, slewed body translation commands over simulation time.
- All 12 observed features and their normalized values. Raw distances are in
  metres, velocities in m/s, and heading error in radians.
- A binary spike raster for 64 first-layer and 32 second-layer neurons.
- One representative second-layer neuron's membrane before and after reset.
- The four readout scores across each 16-step decision and which options the
  geometric safety mask disallowed. Scores are not probabilities.

`brain/traces.json` contains every neuron's recorded replay trace for further
analysis. Model hashes and replayed final scores must match the saved episode;
the report refuses mismatched models. Membranes start from zero at each
decision. Each step leaks the previous membrane by beta=0.9, adds current,
emits one binary spike at a value of at least 1, then subtracts 1. This is a
soft reset: a membrane can remain above threshold after emitting a spike.
The values are dimensionless model states, not measured biological voltages.
Internal steps are not simulation seconds.

At 20.8 seconds in the saved run, both bypass paths had negative predicted
clearance and were masked. The SNN emitted 632 spikes and gave `wait` a score
of 6.04, above the available `back_off` score of -1.84. The supervisor then
held a standing gait until the hazard cleared. PID continued controlling the
joints during this wait; no neural activity is invented between decisions.

The signal chain is observations → SNN spikes → skill scores → safety mask →
skill waypoints → body commands → gait joint targets → PID motor torques.
Individual neuron plots illustrate the implemented dynamics; they do not
establish a semantic meaning or causal explanation for that neuron.

Validation: the existing 30 tests passed, plus a new test checking that traced
and ordinary predictions are identical and verifying firing/reset equations,
spike totals and final readout parity across eight input vectors. The original
validation snapshot remains historical; the faster settings have only the
targeted checks documented above.
