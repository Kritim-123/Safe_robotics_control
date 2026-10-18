# V5: repeated obstacle-selected jumps

One robot walks to several low bars, stops, selects each jump from fresh
LIF counts, verifies crossing and stable landing, then resumes walking.
After the final bar the unchanged V3 selector handles goal stopping.
No intermediate robot placement or explicit jump request is used.

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.run --viewer --video
# Four bars; the model and motor profile remain frozen.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.run --count 4 --spacing 2.4 --viewer
# Replay autonomous assessment to a new output directory.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.replay --output output/obstacle_v5_multi/replay-assessment
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_multi.py -v
```

The 264-cell refined V5 encoder, normalization, existing associations, jump
profile and actuator bounds are preserved. A separate model extends the
association bank with measured outcomes from seed 21701 multi-bar courses.
Original V5 sources/models remain unchanged. Collection uses explicitly
flagged teacher proposals; assessment and the demo use learned proposals.
The real final-goal distance remains an input throughout the episode.

Approach and braking use classical range control, with bounded lateral/yaw
correction to prevent cumulative drift. Jump selection requires two fresh
learned proposals plus all existing launch/landing eligibility checks. Each
landing must physically clear the bar and remain stable before a new target
can be selected. Passed bars remain collision geometry; only their decision
observations are retired. A failed crossing, contact, fall, refusal or timeout
terminates the episode and releases torques.

Raw events, observation/count decisions, state/torque traces and summaries
are written into an exclusive directory under `output/obstacle_v5_multi/`.
With `--video`, `recording.mp4` overlays the actual phase, completed-bar count
and current neural counts. Compact results and model provenance are under
`docs/agents/obstacle_v5_multi/`.

This experiment covers aligned static 2–4 cm low bars on flat ground with
MuJoCo geometry/state sensing. It does not add camera perception, arbitrary
obstacle navigation or a new low-level learned locomotion policy.
