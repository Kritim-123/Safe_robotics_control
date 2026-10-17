# Saved dense-terrain association model

This portable bundle contains `dense_encoder.npz` and `dense.json`: a frozen
22-input, 16-output random LIF encoder and 94 outcome-labeled associations.
The original 14 weight columns and biases are preserved; eight terrain-input
columns use seed 4242. Each observation resets the 16-step encoding window.
Weights are not trained. Association fitting uses original verified training
examples plus 54 new training contexts, never validation or evaluation labels.

Use `python -m experiments.spike_pattern_skills.run_learned --viewer` from the
project root. The CLI defaults to this bundle and saves all new recordings under
`output/spike_pattern_skills/learned-live/`; it does not train in the background.
`--video` creates finite MP4 recordings. `--cases` selects known learning or
fresh evaluation case names. `--bundle` allows an explicit different bundle.

`legacy_encoder.npz` and `original.json` support comparison. `fit_manifest.json`
links the verified source collection and frozen artifact hashes. Absolute source
paths are provenance information from the experiment machine, not requirements
for inference. `reencoded_labels.jsonl` includes training and validation records,
but only records with split `train` enter the bank. `evaluation/` contains the
fresh-draw and old regression comparison. `audit.json` records replay and
training-split checks. Complete raw trajectories/videos are in the ignored
workspace output folders documented in the milestone report.

See `../expanded_association_validation.md`, `../expanded_association_protocol.md`,
and `../dense_association_protocol.md` for the observation/skill contract and
remaining failures. The in-place hop is explicitly requested and is not an
obstacle-clearing jump policy. Hard terrain remains unsolved.
