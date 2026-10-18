# Refined V5: landing clearance and outcome associations

The four original refusals and the contact case now complete autonomously.
Fresh paired assessment: **40/40** complete courses versus original V5's
**31/40**, no refined contacts/falls and no lost successes. All **211 tests
passed**. See the [validation report](../../docs/agents/v5_refinement_validation.md).

This opt-in revision preserves the original V5 as a baseline. It adjusts the
jump's forward lean, thrust timing and landing absorption pose while keeping
the 72 Nm jump / 24 Nm walking bounds and all contact/crossing guards.
The approach, braking and V3 goal controller are unchanged.

The original 168 LIF cells remain intact. Another 96 fine level-sensitive
cells encode geometry, friction and current posture. Inputs additionally
include measured body pitch (rad), base height (m) and angular speed (rad/s).
Membranes/counts reset for each packet. A new outcome-association bank learns
the retained profile's measured crossings; no jump-request input is supplied.

```powershell
# Replay the previously contacting course; inference and recording only.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --viewer --video --scene former-contact --viewer-speed .5

# Original twenty courses are now development regressions.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --phase regression --output output/obstacle_v5_refinement/replay-regression

# Fresh paired comparison with original V5; use an exclusive output directory.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.run --phase fresh --count 40 --output output/obstacle_v5_refinement/replay-fresh
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_obstacle_v5_refinement.py -v
```

Teacher-proposal collection is explicitly marked in its logs and is distinct
from autonomous assessment. The original twenty worlds are used for targeted
training alongside sixty additional training worlds; validation/fresh seeds
are disjoint. Up to two preceding eligible packets share the actual jump's
outcome label, with actuation/observation times recorded separately.

```powershell
# Reproduce collection/fitting to new paths, preserving the retained model.
.venv-win/Scripts/python.exe -m experiments.obstacle_v5_refinement.train --output output/obstacle_v5_refinement/retrain-trials --model output/obstacle_v5_refinement/retrained/head.npz
```

This remains a restricted simulator experiment: flat ground, static aligned
thin low bars and MuJoCo geometry/state observations. It does not use camera
perception, certify physical robot safety or establish an SNN advantage.
