# Goal count hysteresis second iteration

October 17, 2026, started after the first frozen noisy-pose cohort. The V1 model,
runtime, training records and ongoing 1000-scene evaluation stay unchanged.
V1 noise results become development evidence for this V2 readout iteration.

Observed failure: the first mild-noise goal settled 13.19 s after first entering
the goal region and switched eight times. Five of the first six noisy V1 goals
failed, while the rule control passed five. This motivates output commitment,
not another set of encoder or motor weights.

V2 reuses the exact V1 model and threshold 92. Hold commitment begins when a
fresh count packet indicates goal hold, then remains until the proximity count
falls below 25 on two fresh packets. The value 25 is the unchanged population's
response at the existing 0.16 m goal-resume boundary. The standard physical
handoff supervisor still debounces requested labels and checks braking/support.
External task changes and episode reset clear the goal-hold commitment. Neuron
membranes/counts continue to reset every packet. This is readout/supervisor
state, not recurrent SNN training.

The first single-packet entry candidate exposed premature commitment under
position noise. Its schema-1 checkpoints/results remain loadable. Before V2
final tests, schema 2 requires two fresh count packets for entry as well as two
for release; new validation seeds 15001 and 15051 replace the initial candidate
qualification. The action decoder's neuron weights and count thresholds remain
unchanged. Neither validation set is presented as a final holdout.

New validation seeds 15000 and 15050, 20 flat and 20 mixed terrain cases,
2 cm position and 0.02 rad yaw noise, with the existing obstacle perturbation.
Compare frozen V1, V2 and the rule control on identical cases. Promote only if
no collision/fall/torque violation and improved noisy strict goal completion;
use fresh seeds 15100 onward for V2 confirmation. Main V1 results are retained
and not labelled independent V2 evidence. No hardware or camera deployment.
# Additional frozen-model diagnostic

At 19:25 UTC, add the frozen V2 readout to the same 100 wider-goal/low-friction
OOD scenes already being evaluated with V1 and rule control. This is a paired
diagnostic prompted by the calf contact; it is not another 100 independent
environments and is not used to choose parameters. Preserve every refusal,
contact and missed goal. Use the same classical motor configuration and
unchanged observation settings. The contact replay is a repeat of case 14,
not a new test scene.

At 19:36 UTC, declare a separate fixed goal-vector bias diagnostic: 20 new flat
seed-15200 scenes, repeated with additive longitudinal errors of -8, -4, +4 and
+8 cm under frozen V1 and V2. Alter only the goal feature before inference;
leave lateral goal, geometry masks, low-level pose feedback and guards alone.
This isolates the stopping decoder and is not a full localization-stack test.
Save the injected features, unchanged model checkpoint and exact condition.
There are 20 independent generated geometries, not 80 or 160. Retain failures
under the unchanged strict goal contract and do not retune from this diagnostic.
