# Observation and spike-pattern contract v1

The observation adapter, random encoder and recorder below are implemented.
Associations, skill labels and integrated control remain proposed future work.
One agent in the current chat owns this folder and its tests, following the
repository's standing workflow.

## Observations

The 14 ordered inputs reuse the existing behavior-dataset feature meanings:

| Features | Units/frame | Fixed divisor |
|---|---|---|
| obstacle_present | binary | 1 |
| obstacle_forward, obstacle_left | m, robot heading frame | 3, 3 |
| obstacle_v_forward, obstacle_v_left | m/s, robot heading frame | .5, .5 |
| obstacle_radius | m, conservative planning radius | .5 |
| obstacle_height, obstacle_width | m | 1, 1 |
| footprint_clearance | m, signed conservative footprint gap | 3 |
| goal_forward, goal_left | m, robot heading frame | 3, 3 |
| speed | m/s, planar magnitude | .5 |
| base_height | m, world vertical | .35 |
| jump_requested | binary, explicit task request | 1 |

Divide by these fixed scales and clip to [-3, 3]. No outcome-derived fitting is
used. The nearest obstacle supplies its features; absent obstacles have zeroed
obstacle fields. This is a compressed scene description, not raw camera pixels.
Record observation time in simulation seconds and retain the raw named inputs.
Only the feature dictionary enters the network; labels and metadata do not.

The current project obtains robot and obstacle states from MuJoCo. Camera/LiDAR
perception is a future observation source, not an existing capability.

## Seeded SNN encoder

V1 is a single dense layer from 14 inputs to 16 pattern LIF neurons. PCG64 seed
42 draws row-major weights from Normal(0, 1/sqrt(14)), then biases from
Normal(0, .1). Both stay frozen. The network uses float64 arithmetic, 16 internal
constant-current steps, beta .9 and threshold 1:

```text
current = weights @ normalized_features + biases
membrane = .9 * membrane + current
spike = membrane >= 1
membrane = membrane - spike
```

Each neuron emits at most one binary spike per internal step. These steps have
no assigned physical duration and are not successive camera/sensor frames.
There is no hidden layer or recurrent connection in this first encoder.

- Initialize random weights and biases using a recorded network seed.
- Save the actual arrays as well as the seed, initialization method, feature
  schema, dimensions, neuron parameters and dependency versions.
- Start with LIF neurons and a fixed decision window. Reset membrane state at
  each independent observation window for the first experiment.
- Keep encoder weights fixed while collecting patterns and learning associations.
- Assign a model checksum to recordings so associations cannot silently attach
  to a different encoder or feature order.

Repeatability requires the same saved model, inputs, encoding and initial state.
A seed alone does not guarantee the same pattern under changed observations or
changed software. Simulator variation should have its own separately recorded
seed/configuration.

## Spike recording

Retain the full binary raster indexed by time step and output neuron, plus:

- the sorted, unique IDs of neurons that fired at least once;
- a spike count for each output neuron;
- the observation ID, encoder checksum and window/reset metadata.

In the first representation, `[1, 3, 5, 7]` means those neurons were active
somewhere in the window. It does not encode their firing order. Keeping the
raster allows later comparison with temporal patterns without recollecting data.

## Associations and labels

An output neuron is a component of a pattern, rather than a named skill class.
The decoder associates complete patterns or their recorded features with skill
identifiers. Start by inspecting exact-pattern associations; evaluate similarity
or a learned decoder when exact matches fail to generalize.

For every attempt, record the scenario/context ID, pattern, requested skill,
eligibility, executed skill, task objective, physical completion, task fulfillment
and failure reason. Preserve rejected and unsuccessful attempts. Use namespaced
skill identifiers from the actual skill inventory when integration begins.

A skill label must reflect a verified successful response to the task. An
in-place hop cannot be labeled as successful obstacle crossing. Standing safely
does not automatically fulfill a forward-progress request.

The same pattern may occur in situations needing different skills. Retain such
conflicts explicitly; do not silently overwrite an earlier association. Unknown,
silent or ambiguous patterns should allow decoder abstention, with recovery or
fallback decided by the eventual supervisor and current skill eligibility.

## Validation milestones

1. Identical saved network/input/reset produces identical raster and pattern.
2. Saved and reloaded weights produce identical inference.
3. Record pattern collisions, silent outputs and within-scenario variation.
4. Keep all records from a scenario/context together when splitting training
   and evaluation data. Report unseen patterns and conflicting associations.
5. Evaluate task fulfillment, skill failures and decoder abstention separately.
6. Before integration, agree on skill initiation, termination, interruption and
   recovery, then test handoffs with persistent physics and existing torque guards.

The recorder uses the existing standing simulation to capture observations,
then ends before executing a skill. No random-output actuation, selector
retraining or hardware deployment is included in this milestone.
