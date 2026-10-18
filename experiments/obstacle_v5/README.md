# V5: walk up, brake, select a jump, continue to goal

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.run --viewer --video --viewer-speed .5
```

The finite nominal episode starts .9 m from a static 4 cm bar. It walks about
65 cm, brakes to a supported stand about 25 cm before the bar, selects the
jump from fresh SNN counts, verifies full crossing/landing, then walks to the
goal and stops. The mission is always `progress`; there is no jump request.
At half speed, jump preparation begins about 29 seconds after physics starts.

Approach and braking are classical distance control: 6 cm foot lift, .18 m/s
far command, .12 m/s within .55 m, and a .255 m braking trigger checked at
100 Hz. Braking returns the current joint targets to standing over .65 s
instead of starting another gait swing. The robot must then remain supported
and stopped for >=.4 s, followed by two fresh eligible neural proposals.

The V4 168-cell LIF encoder is unchanged. A new outcome association bank learns
from 32 walk-up training episodes (28 crossings, 3 contacts, 1 incomplete
crossing) plus 80 explicitly marked eligibility-negative examples. Collection
uses teacher proposals; validation, tests and the live demo do not. The
retained model is `docs/agents/obstacle_v5/walk_up_v2.npz`; `primary.npz` in
that directory is an earlier rejected development candidate.

Frozen held-out assessment: **15/20** complete courses, four learned refusals
and one rear-thigh contact during the jump; no falls or approach overshoots.
Six unsuitable-target controls made no jump/contact. The clear-path control
reached the goal without jumping. All **204 project tests passed**.
See the [report](../../docs/agents/approach_low_bar_v5_validation.md).

This remains an opt-in simulation experiment: flat ground, +X travel, aligned
static thin bars, starts .75–1.2 m from the bar, heights 2–4 cm, friction .8–.9,
and the tested goal 1.26 m beyond the bar. Geometry/friction come from MuJoCo,
not a camera. Existing 24 Nm walking and 72 Nm low-bar bounds remain. V3, V4
and the low-bar jump trajectory are unchanged. Physical safety and general
obstacle reliability are not established.

Each invocation needs fresh output paths:

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.run --phase evaluate --split test --count 20 --output output/obstacle_v5/replay-test
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5.py -v
.venv-win/Scripts/python.exe -m experiments.obstacle_v5.train --output output/obstacle_v5/retrain-trials --model output/obstacle_v5/retrained/head.npz
```

Decisions, phases, launch-state measurements, complete physics traces and
optional neuron-count movies are saved under `output/obstacle_v5/`.
