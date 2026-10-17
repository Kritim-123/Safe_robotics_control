# Research brief: repeated SNN skill selection

October 17, 2026. This brief recommends experiments; measured outcomes belong
in the validation report. One agent applies researcher, developer and integrator
checklists sequentially. No external robot-policy code is imported here.

## Evidence and useful ideas

1. **Options with explicit initiation and termination.** Sutton, Precup and
   Singh formulate temporally extended actions and discuss interruption during
   execution. This supports treating a skill request separately from a permitted
   handoff; it does not justify replacing joint targets every sensor frame.
   [Original paper, 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521),
   [author paper copy](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf).

2. **Fixed spiking dynamics plus a learned readout.** Maass, Natschlaeger and
   Markram's liquid-state-machine work uses recurrent dynamics as a fading
   memory and a task-specific readout. A readout on our frozen random encoder
   is a compatible experiment; our feedforward constant-current encoder is
   not itself a liquid-state machine. Test a separate recurrent reservoir only
   when the task needs history.
   [Original paper, 2002](https://pubmed.ncbi.nlm.nih.gov/12433288/),
   [primary conference paper](https://proceedings.neurips.cc/paper/2002/file/6211080fa89981f66b1a0c9d55c61d0f-Paper.pdf).

3. **Spike timing as well as counts.** Van Rossum maps spike trains through a
   causal exponential filter and compares their distance. Total counts lose
   timing distinctions. We can compare complete rasters, filtered rasters
   (with a documented tail), and counts on identical observations. Internal
   encoding steps must not be presented as biological milliseconds or sensor
   frames. This paper does not establish a robot-control advantage.
   [Author manuscript, 2001](https://era.ed.ac.uk/bitstream/handle/1842/226/distance.pdf?sequence=1).

4. **Adaptive threshold neurons.** Bellec et al. study spike-triggered slow
   adaptation in trained recurrent SNNs on memory tasks. Merely adding
   adaptation to a frozen checkpoint need not improve it. Compare adaptation
   only with a refitted readout and matched data, and distinguish adaptation
   within an encoding packet from state retained across real observations.
   [NeurIPS paper, 2018](https://papers.nips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html).
   [Official LSNN code](https://github.com/IGITUGraz/LSNN-official) lists a
   BSD-3-Clause-Clear license and old TensorFlow dependencies. It is a reference,
   not a dependency of this project.

5. **Sequence training and eligibility traces.** E-prop combines forward
   eligibility traces with learning signals for recurrent spiking networks.
   It is a principled future online-learning direction, not a reason to update
   motor-affecting associations from unverified experience during evaluation.
   Our first comparison learns only versioned readouts/banks offline.
   [Original paper, 2020](https://www.nature.com/articles/s41467-020-17236-y).

6. **Surrogate gradients.** They make supervised training through spikes
   practical, but do not remove the need for suitable task labels and grouped
   sequence splits. The existing project already has a trained four-skill
   surrogate-gradient SNN. Replacing the user's reproducible random-pattern
   mechanism is not necessary to test the current handoff idea.
   [Author manuscript](https://nmi-lab.org/ielaam/79/8887548/8891809-aam.pdf),
   [author tutorial implementation](https://github.com/fzenke/spytorch).

## Local evidence that changes the recommendation

The older navigation memory experiment found identical actions/outcomes in
21 reset/retain pairs: see `integration_validation.md`. It is not new evidence
and does not reject sequence-trained memory. The current random-pattern demo
has a separate 22->16 frozen encoder and 94-entry count bank. Its 16-step
packet resets between observations; live packets are logged but do not trigger
handoffs. Most examples describe settled entry states. Count-only aliasing,
motion-induced distribution shift, and one-shot execution are distinct issues.

## Recommended order

Implement explicit repeat-selection/packet-reset/handoff semantics first, with
the frozen baseline unchanged. Compare readout/representation alternatives on
the same train/validation contexts. Add motion observations to the training
distribution and label their source honestly: causal teacher requests and
measured skill outcomes are different. Compare against the teacher and an
analog-feature diagnostic as well as the old SNN. Freeze candidates before
matched new physical episodes and a final confirmation split.

Start with counts, full/filtered rasters, larger fixed seeded populations and
refitted uncertainty-aware readouts. Test adaptive LIF or retained state as
ablations; recommend them only if physical evidence supports them. Include
noisy/delayed observations and situations with the same current observation but
different histories if making a memory claim. Keep all neural decisions
subordinate to eligibility, braking, jump commitment and recovery.

## Ideas excluded from immediate implementation

Unsupervised STDP has no supplied signal tying a firing pattern to successful
task completion. Randomly increasing leak or adaptation is not learning a
skill. Constantly clearing all state can erase necessary movement commitment;
retaining all state can accumulate stale evidence. Image/event-classification
benchmarks do not demonstrate robot performance. Long RL or imported motor
policy retraining would confound this selector milestone. None of these points
means those methods are universally useless; their immediate benefit here is
unestablished. Negative ablations and source references remain in the record.
