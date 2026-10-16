# Pretrained Go2 jump transfer experiment — October 16, 2026

Imported a genuine pretrained jump actor, but **it does not yet improve the
working in-place hop**. The candidate is available as an experimental module;
it is not eligible for SNN selection and does not replace `DynamicSkill('jump')`.

## Source and implementation

[SafeRoboticsLab/go2_atomic_skills](https://github.com/SafeRoboticsLab/go2_atomic_skills),
MIT license, pinned commit `ce0169d179525c72c712df8e62475e2a0536d124`.
Vendored inference code, walker actor/normalizer and `hando_w30` jump
actor/normalizer are in `third_party/go2_atomic_skills`. `PROVENANCE.json`
records local SHA256 checksums. The unsafe legacy pickle-loading fallback was
removed; Torch loads weights with `weights_only=True`.

The jump is a reach-avoid **forward gap-crossing** actor, not an on-demand vertical
hop. It consumes 235 features across five frames (1175 inputs), including a
187-ray terrain scan. `skills/pretrained_jump.py` supplies real downward rays,
excludes robot geometry, maps FL/FR/RL/RR joints, updates both policy clocks and
history at 50 Hz, shares the preceding applied action across handoffs, and
returns joint targets to bounded torque PD. The existing LIF model remains a
skill selector; this imported actor is an ordinary neural motor policy.

The scene uses the project's Go2 XML, two platforms separated by a real 12–30 cm
gap and a pit floor 1 m below them. The walker accelerates toward the gap;
handoff occurs 0.4 m before the near edge. After clearing the gap, the walker
receives zero velocity and must settle for 0.8 s. This geometric handoff is
experimental: it does not reproduce upstream's critic-based value filter.

Training PD/armature are reproduced, but the project's calf torque ceiling stays
24 Nm (upstream uses 45 Nm); hip/thigh limits are 23.5 Nm. Physics uses the
project's 2 ms timestep versus upstream's 5 ms. Model/contact differences and
handoff strategy are additional transfer mismatches. Saturation is observed;
these tests do **not** isolate its causal contribution to failures.

## Measured evidence

Eight deterministic conditions were run on the local model, with no teleporting
or external lift forces. All use friction 0.8.

| Gap (m) | Command (m/s) | Result | Jump-phase maximum airborne interval (s) |
|---|---|---|---|
| 0, walker only | 1.0 | Traversed and settled; x=1.866 m | 0 |
| .12 | 1.0 | Guard stop, no crossing | .052 |
| .20 | 1.0 | Guard stop, no crossing | .080 |
| .30 | 1.0 | Guard stop, no crossing | .058 |
| .30 | 1.5 | Guard stop, no crossing | .104 |
| .30 | 2.0 | Guard stop, no crossing | .026 |
| .30 | 2.5 | Guard stop during approach, before jump | 0 |
| .12 | 2.0 | Guard stop, no crossing | .016 |

**0/7 gap tasks succeeded**. All trials reached the 24 Nm ceiling. Guards stop
on non-foot terrain contact, base height below 0.15 m, tilt above 35 degrees,
nonfinite state or arena escape; torque is released on exit. Airtime alone is
not a successful jump. Success requires crossing, stable landing and at least
40 ms airborne during the jump phase. These are development trials, not held-out
generalization evidence. The existing hop solves a different task, so this is
not a paired ranking of the two controllers.

Per-condition `trace.csv`, `observations.npz` and `summary.json` are under
`output/pretrained_jump/evaluation/`; `results.json` aggregates outcomes.
NPZ control samples contain qpos(19), qvel(18), joint target(12), terrain scan(187),
plus terminal state. CSV includes a terminal row. These failed executions must
remain labeled failures if later used for training; they are not teacher jumps.

## Reproduce and view

```powershell
.venv-win/Scripts/python.exe -m skills.pretrained_jump --width .3 --speed 1 --viewer --output output/pretrained_jump/viewer
.venv-win/Scripts/python.exe -m skills.pretrained_jump --width 0 --speed 1 --walk-only --output output/pretrained_jump/flat-control
.venv-win/Scripts/python.exe -m unittest discover -s tests -p test_pretrained_jump.py -v
```

Four focused tests passed: actual gap sensing without robot self-hits,
policy history/reset/handoff, bounded rollout failure reporting, invalid input.
The full command `.venv-win/Scripts/python.exe -m unittest discover -s tests -v`
also passed: **79 tests in 50.505 s**.
Further improvement requires training/adapting the actor to this model and its
24 Nm limits, then demonstrating clean crossings before exposing it to the SNN.
The current environment includes Torch already; this adapter adds no new package
dependency.
