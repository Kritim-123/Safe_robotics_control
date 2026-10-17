# Research notes — 2026-10-17

Primary sources accessed during this session:

* [Schaul et al., UVFA, ICML 2015](https://proceedings.mlr.press/v37/schaul15.html):
  condition predictions on goals and evaluate unseen goals. Here the immediate
  application is an explicit arrival/hold contract and goal-disjoint tests;
  implementing that contract is not an implementation of UVFA reinforcement learning.
* [Ross et al., DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html):
  policy errors change future observations. Collect causal teacher labels on
  states reached by the learner, using training scenes only. Keep failed
  physical outcomes separate from teacher preferences. This does not make a
  teacher request a verified counterfactual skill outcome.
* [Andrychowicz et al., HER](https://arxiv.org/abs/1707.01495):
  achieved-goal replay can reuse sparse-reward experience. Do not relabel our
  failed 1.55 m benchmark as a successful shorter traversal. An off-policy goal
  learner would need explicit transitions/rewards, which the current bank lacks.
* [Miki et al., perceptive locomotion, 2022](https://leggedrobotics.github.io/rl-perceptiveloco/):
  fuse terrain perception and proprioception, accounting for unreliable terrain
  observations. Our initial bounded test uses privileged map heights with the
  existing gait. It is not their recurrent learned controller, and perfect map
  performance would not establish robustness to camera errors.
* [Zhang et al., Motion Priors Reimagined, CoRL 2025](https://proceedings.mlr.press/v305/zhang25j.html):
  a goal-conditioned high-level policy and low-level terrain residuals address
  distinct problems. Separate selection improvements from motor adaptation in
  factorial controls. We do not import their code or assert equivalent results.

Initial hypotheses: (1) shorter goal/room contexts move frozen spike patterns
outside useful bank coverage; (2) encoding task identity only through zeroed
goal coordinates entangles movement intent with distance; (3) fixed foot
clearance/support targets and drift cause physical terrain failures; (4) timed
progress scoring is distinct from reaching and holding a specified goal.

Tests: unchanged V2/rule replay, same-encoder new-data associations, explicit
task population, factorized decision representation, matched motor-adapter
ablations, and new goal/terrain held-outs after validation selection. Keep
non-spiking rule control. Do not increase neurons or import another large
network until a measured failure warrants it.

Further sources used in development:

* [HIM, ICLR 2024](https://arxiv.org/abs/2312.11460) motivates testing the
  already imported proprioceptive-history motor policy. This milestone trains
  no motor weights. Its 50 Hz history is separate from the 10 Hz SNN packets;
  clearing spike packets must not clear motor-policy history during braking.
* [Population coding](https://neuronaldynamics.epfl.ch/online/Ch12.S1.html)
  motivates an explicit distance-sensitive input population. Our Gaussian
  distance features plus frozen LIF projection are an engineered representation,
  not a claim about biological tuning or a reproduction of a neuroscience study.
* [Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/abs/1912.06088)
  is a future route to goal learning from interaction. The implemented learner
  remains supervised by causal teacher requests; it is not that expert-free RL
  algorithm. Original failed goals are not relabelled as benchmark successes.

Provenance: the existing numeric HIM artifact comes from the pinned rl_sar
bundle recorded in `skills/policies/rl_sar/himloco/manifest.json`. Its original
paper/project are attributed above. The current original
[HIMLoco repository](https://github.com/InternRobotics/HIMLoco) lists
CC BY-NC-SA 4.0, whereas the retained rl_sar wrapper bundle includes Apache-2.0.
Do not infer that a wrapper licence establishes the original model's licence.
No new external model or source code was downloaded or redistributed here;
the existing artifact hashes and notices are retained.

The frozen V1 noise test motivated a separate confirmed goal-hold readout.
[Hespanha, Liberzon and Morse, 2003](https://www.sciencedirect.com/science/article/pii/S0005109802002418)
studies hysteresis in supervisory switching for uncertain linear systems.
Here the useful principle is separated entry/release evidence to reduce
noise-driven switching. Our nonlinear robot does not inherit that paper's
stability theorem. V2 empirically tests a learned entry count of 92, a release
count of 25 corresponding to the existing 0.16 m resume boundary, and two fresh
packets at each boundary. Membrane/count reset and goal-hold commitment remain
separate mechanisms. V2 uses new validation and confirmation scenes; V1 noisy
outcomes remain visible as development evidence for the second iteration.

The OOD block diagnostic narrows the next motor problem: the rear-left calf
contacts a block before the goal is reached, with identical trajectories under
the SNN and rule controls. Raising swing feet alone does not constrain the
whole leg. [TAMOLS, Jenelten et al.](https://arxiv.org/abs/2206.14049) jointly
plans body motion and footholds against a heightmap. Its relevant lesson is
to treat terrain traversal as a geometric and dynamic motion problem, rather
than expecting a different wait/run label to produce missing foothold control.
No TAMOLS optimizer has been implemented in this milestone.

[RMA, Kumar et al.](https://ashish-kmr.github.io/rma-legged-robots/) learns a
motor policy and an adaptation module that uses recent interaction history.
That suggests separating friction/motor adaptation memory from resettable
SNN decision packets. It would require motor training and its own validation;
it is not achieved by adding more spike-pattern associations. The present
controller has no learned friction estimator and uses the existing bounded PID.

## Proposed next milestone

### Neuron learning alternatives

[Bellec et al., 2018](https://arxiv.org/abs/1803.09574) shows that adaptive
neurons in trained recurrent spiking networks can support longer temporal
dependencies. [Neftci, Mostafa and Zenke, 2019](https://arxiv.org/abs/1901.09948)
describes surrogate gradients for optimizing spiking networks. These are
credible alternatives when learning a history-dependent observation model.
They have not been implemented or benchmarked here. Changing LIF to ALIF while
clearing all adaptive state every 100 ms would remove the cross-packet memory
we would be trying to test. A fair future comparison must specify which state
persists, compare equal observation histories and training data, and measure
closed-loop goal completion rather than packet accuracy alone.

For this milestone, the measured failures warranted representation, teacher
coverage and readout commitment changes first. A constant biased goal vector
can still be wrong even with a perfectly stable recurrent neuron: better
memory cannot identify an unobserved true position by itself. Similarly, no
neuron substitution creates a missing terrain foothold controller. Preserve
the resettable seeded baseline as a control if recurrent training is later added.

### Observation and motor contracts

First, add an explicit terrain observation contract: local height samples and
validity, surface normal, estimated friction uncertainty, actual support feet,
body tilt/velocity, and distance to untraversable cells. Keep metres, radians
and seconds, record map age, and keep unknown cells distinct from flat ground.
Privileged map training and degraded-map testing must be reported separately.

Develop a callable `terrain_walk` option before making it another selector
class. It needs checked whole-leg/body clearance over the predicted swing,
reachable touchdown targets, bounded speed, stable entry and interruption,
explicit success/failure signals and a recovery path. A refusal is a safe
failure, not traversal success. Evaluate the existing calf-contact case as a
regression, then freeze the option and use new terrain seeds and dimensions.
Compare the same SNN with old/new motor implementations and the same rule
selector with both motors. Require no torque-limit changes, count contact and
fall rates, and measure supported goal completion and lateral deviation.

For moving crossings, add a route/wait decision using predicted obstacle
occupancy and a finite blocked timeout. The existing run/wait/jump vocabulary
cannot by itself express a validated detour skill. Count original-deadline
success separately from success with more time; do not extend deadlines to
erase current failures. If a new option changes observations or labels, version
the contract and retrain only from declared training scenes.

Only then replace the privileged map with a depth/elevation pipeline, including
occlusion and map-age tests. A learned motor policy such as an RMA-style
alternative is a separate research branch with explicit model provenance and
bounded integration; it is not an immediate replacement justified by these
simulation results.
