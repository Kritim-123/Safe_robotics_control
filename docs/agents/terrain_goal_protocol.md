# Terrain and goal transfer: three-hour protocol

Session starts 2026-10-17 18:01:40 UTC (12:01:40 America/Denver), ends no earlier
than 21:01:40 UTC. One agent owns all new files and any explicitly versioned
integration edits. Preserve V1/V2 models, original manifests, failed evidence,
motor limits, PID and clearance/fall guards.

## Acceptance and contracts

1. Reproduce the original 60-case terrain benchmark with frozen V2 and causal
rule controls. Preserve its 1.55 m progress, hold and termination criteria.
2. Diagnose goal/room distribution shift separately from physical terrain
limitations. New goal-arrival evaluation requires measured distance <=0.12 m,
supported stopped gait and >=0.8 s stable hold. It is a new contract, not a
retroactive change to the 0/60 benchmark.
3. Train only on declared new training cases; choose on separate validation
cases; freeze before fresh final scenes. Historical 60 cases are development
evidence for this milestone. Report old-case improvement and genuinely fresh
generalization separately.
4. Compare learned count associations, non-spiking controls and unchanged V2.
Require no motor-limit violations; retain collisions, falls, refusals and
timeouts as failures. Do not label unsuccessful counterfactual skills as good.
5. Test same-skill continuation, braking, eligibility, stale input and hold;
run relevant tests followed by the integrated suite for controller changes.

Observations retain metres/seconds/radians and privileged MuJoCo state/map.
The three current selector classes are wait/run/jump. The existing classical
navigation and imported velocity policies are callable implementations, not
automatically new selector classes. No camera or hardware claim is made.
The separate 72 Nm obstacle-jump option remains excluded. Wait/run limits stay
24 Nm and short-hop 28 Nm. Any new target adapter must have explicit initiation,
continuation, interruption, terminal success/failure and trace provenance.

Ownership: this single agent writes `experiments/terrain_goal/`, new tests,
`docs/agents/terrain_goal*`, `output/terrain_goal/`, and minimal documented
opt-in integration hooks if needed. Other existing user changes are preserved.
Failed ideas remain in the report but are excluded from recommendations.

## Development clarification before final holdouts

For new goal tests, terminal position alone is insufficient: the low-level
goal tracker could orbit the destination until a timed hold request. Require
the selector to enter a supported wait for >=0.8 s while the mission still
requests progress, within 6 s of first entering the 0.12 m goal region, then
never depart beyond 0.16 m during that mission. Record actual stopping delay.
Six seconds accommodates the existing slow classical gait's approach/braking
and support qualification; the initially probed five-second threshold rejected
a development flat controller. None of these refinements relabel the original
60-case traversal score. Final goal cohorts will use this frozen stronger
contract. Earlier terminal-only goal development results remain identified.
