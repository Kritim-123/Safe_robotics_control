# V4: select a low-bar jump from observations

This opt-in version adds a 168-neuron reset LIF observation head above the
unchanged V3 goal/walking selector. A learned count-pattern association proposes
`low_bar_jump`; a separate eligibility check may veto it. There is no jump
request input. The sole mission throughout is `progress` toward the goal.

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.run --viewer --video --viewer-speed .5
```

The finite demo settles, selects a jump over a static 4 cm bar, verifies complete
foot/body crossing and stable landing, resumes walking and stops at the goal.
Playback speed changes viewer pacing only. Fresh counts, decisions, handoffs,
physics traces and optional video go to an exclusive directory under
`output/obstacle_v4/`. Closing the viewer releases torques.

The initial robot is already in the tested launch stance: aligned +X, bar
23–24 cm ahead, 2–4 cm high and 1.2–2 cm deep, flat ground, friction .8–.9.
Approaching a distant bar, arbitrary headings, multiple jumps, and camera
perception are not implemented. Geometry and friction are simulator data.
The unchanged low-bar profile uses its existing 72 Nm simulation bound;
walking returns to 24 Nm. These are distinct from V3's short hop.

Training contains 32 actual frozen-profile jump trials (30 successes, 2 contact
failures) and 80 separately identified eligibility-negative examples. The fixed
seeded encoder runs 16 LIF updates per fresh observation, then matches spike
counts against the saved outcome association bank. Encoder weights are not
trained. Conflicting nearest labels and unfamiliar count patterns abstain.
The old V3 model and source files are unchanged.

Held-out seed 19401: 18/20 complete crossings and goal stops, 2 count-conflict
refusals, no contact/fall. Six negative controls made no jump; clear-path
control reached its goal without jumping. This is a small, restricted test,
not a general obstacle policy. See the [report](../../docs/agents/autonomous_low_bar_v4_validation.md).

```powershell
# New exclusive output folder required for each invocation.
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.run --phase evaluate --split test --count 20 --output output/obstacle_v4/replay-test
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v4.py -v

# Reproduce training into a new artifact path; do not overwrite the retained model.
.venv-win/Scripts/python.exe -m experiments.obstacle_v4.train --output output/obstacle_v4/retrain-trials --model output/obstacle_v4/retrained/primary.npz
```
