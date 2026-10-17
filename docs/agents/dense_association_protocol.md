# Dense-scan follow-up protocol

The expanded bank's training audit found 21 differently labeled examples with
the same legacy count vector. Coarse terrain inputs reduced this to four: three
flat controls labeled run and one narrow threshold labeled wait. The threshold
fell between forward samples 0.6 and 0.8 m, producing all-zero terrain inputs.
This is an observed training coverage defect, not a missing successful jump label.

Preserve the original 204 outcomes, coarse models, validation and evaluation.
For a separately versioned dense model, change forward sample spacing to 5 cm
over the same 0-1.4 m range and retain lateral offsets -0.2,0,0.2 m. The narrowest
threshold spans 11 cm, so this grid resolves it. Update the slope denominator
to .05 m; all eight feature names, scales, weights, biases, LIF parameters and
association neighbor settings stay fixed. Re-encode saved causal training
observations against their original static terrain geometry; no outcome labels
or validation examples are added to training. Archive this transformation.

Fit the same original 40 plus 54 new training examples. Score the existing 14
validation contexts without tuning. Because the coarse comparison is already
underway, use a separate fresh evaluation draw: curriculum seed 20261023, flat
seed 20261024, held-out variants only, 28 contexts, renamed with a fresh prefix.
Compare original, dense and diagnostic direct run: 84 episodes. Compare original
and dense on the same twelve exposed old regression contexts: 24 episodes.
Freeze dense artifacts before evaluating. Preserve coarse outcomes as development
evidence. No post-evaluation parameter changes are permitted in this follow-up.

Use the same task/primitive distinction, eligibility, guards and promotion
criteria as expanded_association_protocol.md. Diagnose training count conflicts
and missed surface samples explicitly. A dense-count association still cannot
make the current in-place hop clear an obstacle, nor fix weak terrain locomotion.
