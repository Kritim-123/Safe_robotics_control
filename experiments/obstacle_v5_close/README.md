# V5 close-spacing courses

This separate runner repairs the false landing-corridor refusals at 1.8 m
and 2.0 m bar spacing. It preserves the multi-obstacle V5 model, encoder,
association bank, motor profile, full crossing checks and actuator bounds.
The original multi-obstacle package remains a reproducible baseline.

```powershell
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.run --viewer --viewer-speed .5
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.run --spacing 2 --viewer
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_close.evaluate --output output/obstacle_v5_close/replay-assessment
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_multi.audit --batch output/obstacle_v5_close/replay-assessment --artifacts output/obstacle_v5_close/replay-assessment/artifacts --output output/obstacle_v5_close/replay-audit.json
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_close.py -v
```

The old landing guard circumscribed a thin 1.2 m-wide bar with a circle
whose radius was about .6 m. Its width then incorrectly occupied .6 m along
the forward path. The new check uses the actual oriented box, the same .42 m
robot disc, the same 1.2 m corridor, 3 s horizon and .15 m minimum gap.
Distance sampling subtracts a 1.875 cm spatial bound and a maximum-speed
temporal bound. Cylinders and arena bounds remain included.

The `landing_clearance` feature uses the corrected geometry too. This changes
the resulting spike packets without changing the frozen network or learned
rows. Each jump still requires two eligible learned proposals, supported
launching, verified physical crossing and stable landing. No explicit jump
request, teacher proposal or model fitting occurs in these runs.

Default: three 2.5 cm-high bars at 1.8 m center spacing. The viewer shows the
current skill and cleared-bar count. `--video` records the phase and neural
counts. Use new output paths to preserve previous evidence.

The validation keeps the repaired cases, twelve new close courses, all nine
earlier wider worlds and actual blocked/too-close/tall controls separate.
See [report](../../docs/agents/v5_close_validation.md) and retained artifacts
under `docs/agents/obstacle_v5_close/`. This remains simulated aligned low-bar
locomotion using privileged geometry/state observations.
