# Frozen skill selector on new obstacles

Measured October 16, 2026, using the unchanged wait/run/jump checkpoint
`f86266fafeb454bb92ec7ce3db7244dddc8851f7f656a26047c8d439497707a5`.
No training or policy tuning was performed. All episodes run sequentially.

Six new conditions, three seeds each (5100–5102), produced **3/18 task
fulfillments**, **17/18 primitive completions**, zero contacts and zero falls.
Completion of a wait is not completion of the requested forward or jump task.

| Condition | Observed behavior | Task fulfilled |
|---|---|---:|
| Lane between two box crates | Fast trot; about 0.83 m including deceleration | 3/3 |
| Low, wide bar (8 cm high) | Wait twice; one clearance stop before selection | 0/3 |
| Sliding box gate | Wait | 0/3 |
| Approaching box crate | Wait | 0/3 |
| Requested jump near two posts | Wait despite jump eligibility | 0/3 |
| Requested jump on friction 0.25–0.35 with a distant crate | Wait despite jump eligibility | 0/3 |

An additional open-space jump control completed the requested jump in **3/3**
episodes. It used the same fixed profile and model, with an explicit jump request.
Thus the jump primitive remains functional, but the selector did not generalize
the request to the newly introduced obstacle contexts. The slippery trials did
not execute jumps and therefore do not validate slippery-surface jumping.

Example first-seed output spike counts `[wait, run, jump]`: crate lane `[2,4,0]`,
posts `[6,1,0]`, slippery scene `[4,2,0]`. Jump was eligible in the latter two:
the neural preference, rather than its eligibility mask, prevented selection.
The low bar is treated conservatively using its circumscribed planar radius;
one placement caused a clearance stop before inference. The robot has no
obstacle-clearing jump and the selector makes one decision per episode.

## Watch and inspect

- `output/skill_lif/showcase/showcase.mp4`: all six first-seed obstacle trials,
  actual MuJoCo frames, 30 fps, real-time playback, output spikes and outcomes.
- `output/skill_lif/showcase-control/showcase.mp4`: successful jump control.
- `output/skill_lif/showcase/results.csv`: compact results from all 18 trials.
- Each run folder contains scenario JSON, full summary/decision/spike raster,
  trajectory CSV, and a result image for recorded episodes.

Commands executed:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.showcase --video
.\.venv-win\Scripts\python.exe -m hybrid.showcase --video --scenarios jump_control --output output/skill_lif/showcase-control
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -p 'test_dynamic_skills.py' -v
```

Eight relevant regression tests passed in 5.108 s. Recording uses an optional
frame callback; control parameters and model weights remain unchanged. Videos
were rendered from live simulation state, not reconstructed from body-position
logs. The crate-lane final frame and an airborne control frame were inspected.
These are small simulator tests with privileged observations, not hardware evidence.
