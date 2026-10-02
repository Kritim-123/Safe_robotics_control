# Watch robot motion and SNN decisions side by side

From WSL in the project directory:

```bash
./.venv-win/Scripts/python.exe -m hybrid.live
```

PowerShell:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.live
```

Open http://127.0.0.1:8766 and click **Start / restart episode**. Keep the
terminal running. Ctrl+C stops the server and requests simulation shutdown.
Use `--no-browser` to suppress automatic browser opening, `--port 8767` to
change ports, or `--speed 1` for nominal real-time viewing (default 2x).
The original offline inspector remains available separately on port 8765.

For the new obstacle courses, faster reaction profile and expanded graphics,
see [Complex obstacles and neural UI](complex_obstacles_and_neural_ui.md).

The left side renders live MuJoCo physics, current supervisor state, time,
goal distance, measured speed and motor effort. The right side shows the
latest actual decision's features, scores, safety mask, spike raster and
selectable neuron membrane trace. Expand the local sensitivity table to see
which individual input perturbations affect scores most in that encounter.

**Pause** freezes physics at the next telemetry sample; **Resume** continues
the same physics and controller state. **Stop** ends the episode and saves
the partial result. Restart creates a fresh episode after the previous one
finishes. Select among mixed-course, crate, sliding-barrier, reversing-barrier
and late-crossing scenarios. The dashboard uses seed 0, a 0.24 m/s body-command
limit, active-skill replanning and the selected reaction profile (fast by default).

Video and telemetry are published together at 10 simulated samples/second;
browser polling may skip samples. The neural display holds the last real
decision and explicitly shows its age. No neural spikes are invented during
walking or waiting. Analysis recomputes traces using the same model and
actual decision inputs, without changing the controller's chosen action.
Rendering, analysis and browser delays can reduce the wall-clock playback
rate but do not change the fixed simulation/control timesteps.

Unlike the offline inspector, inputs here cannot be edited: displayed
values must remain those used by the running robot. The model stays frozen;
this is inference and visualization, not online training. The demo is a
simulation and does not connect to physical hardware.

Completed or stopped episodes are written to `output/hybrid/live` (overwritten
by the next saved episode). Supply `--output output/hybrid/live-demo-2` to
keep a separate run. Records use the existing summary/events/trajectory
format and can be opened later with `hybrid.inspector` or `hybrid.brain`.
