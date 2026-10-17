"""Render the project report from recorded cohorts and the evidence audit."""
import json
from pathlib import Path
import shutil
from .models import ROOT
from .report import generate
from .audit import paired

BASE=ROOT/'output/continual_snn'
DEST=ROOT/'docs/agents/continual_snn_v1'


def table(groups):
    lines=['| Arm | Tasks | Completed | Obstacle contacts | Falls |','|---|---:|---:|---:|---:|']
    for row in groups:
        lines.append(f'| {row["arm"]} | {row["tasks"]}/{row["trials"]} | {row["completed"]}/{row["trials"]} | {row["collisions"]} | {row["falls"]} |')
    return '\n'.join(lines)


def main():
    records,batches=generate(BASE/'report-v1');lookup={b['batch']:b for b in batches}
    DEST.mkdir(parents=True,exist_ok=True)
    for name in ('cohorts.csv','failures.csv','totals.json'):
        shutil.copyfile(BASE/'report-v1'/name,DEST/name)
    figures=ROOT/'docs/figures';figures.mkdir(parents=True,exist_ok=True)
    for name in ('outcomes','handoffs','coverage-v2'):
        path=BASE/'report-v1'/(name+'.png')
        if path.exists():shutil.copyfile(path,figures/('continual-snn-'+name+'.png'))
    freeze=json.loads((BASE/'frozen-evaluation-v1.json').read_text())
    model=json.loads((DEST/'model_card.json').read_text())
    totals=json.loads((DEST/'totals.json').read_text())
    session=json.loads((BASE/'session-v1/session.json').read_text())
    end=session.get('end_utc','in progress; requested end 2026-10-17 10:09:45 UTC')
    ablation=''
    for name,title in (('data-ablation-v1','Association-data ablation'),
                       ('data-ablation-stress-v1','Exploratory stress follow-up')):
        if name in lookup:ablation+='### '+title+'\n\n'+table(lookup[name]['groups'])+'\n\n'
    if ablation:
        ablation+=('The data-only comparison preserves the original 16 neurons, all 94 old '
            'examples, and the same legacy three-neighbor/radius-4 decoding algorithm. '
            'It adds the 19,286 training-stream teacher observations. Regular comparison '
            'seeds are 10100–10104. The stress follow-up uses 10200–10204 with the same '
            'models/settings and was chosen after the regular ablation; it is exploratory. '
            'This isolates a large contribution from motion/task-stream coverage. '
            'The extra examples remain teacher labels, not per-frame counterfactual motor outcomes.\n\n')
    coverage=''
    coverage_card=ROOT/'docs/agents/continual_snn_v2/model_card.json'
    if coverage_card.exists():
        card=json.loads(coverage_card.read_text())
        coverage=('### Updated default and fresh confirmation (V2)\n\n'
            'The selected default is the **80-neuron count-pattern association bank**, '
            'rather than the earlier ridge readout. It keeps the same frozen encoder, '
            '16 internal steps, decoder settings, motor skills and 24/28 N m bounds. '
            'The targeted addition is **4,598 observations from 32 successful oscillating/multiple-obstacle teacher episodes**. '
            'These remain teacher requests qualified by whole-episode success, not per-window counterfactual outcomes. '
            'V2 training has 23,978 rows in 254 groups; validation has 6,026 rows in 54 disjoint groups.\n\n')
        coverage+=('V1 stress failures motivated this separate iteration. V2 was selected on physical validation '
            '(40/40 for the prototype bank, 38/40 for the updated readout, 35/40 for the updated original-16 bank), '
            'then frozen at **09:11:08 UTC** before fresh confirmation. The newly expanded 16-neuron bank had '
            '59 mixed-label exact patterns, while the 80-neuron bank had none. That 16-neuron update was rejected; '
            'the useful earlier 16-neuron model remains available.\n\n')
        for phase,title in (('stress','Fresh stress: seeds 11200–11209'),
                            ('regular','Fresh regular streams: seeds 11300–11309'),
                            ('noise','Matched noisy repeats: first four regular seeds')):
            name='coverage-v2-confirm-'+phase
            if name not in lookup:continue
            groups=list(lookup[name]['groups'])+lookup.get('coverage-v2-control-'+phase,{}).get('groups',[])
            coverage+='**'+title+'**\n\n'+table(groups)+'\n\n'
        coverage+=('The same-architecture control uses the same 80-neuron encoder, one-neighbor/radius-12 decoder '
            'and old association data. On the fresh stress set it completes **60/80**, versus **70/80** after '
            'the coverage expansion. The rule controller also completes 70/80. All ten remaining failures are '
            'the continued head-on approach, where this three-action interface lacks an escape option. '
            'The bank grows from 7,159 to **10,446 unique patterns**; repeated identical vectors are deduplicated, '
            'so this comparison isolates added association coverage rather than duplicate-frame voting. '
            'The additional old-prototype arm is a matched control on the same cases, not extra independent scenes.\n\n')
        coverage+=('No fresh evaluation rows were used for training or tuning. Noise is obstacle-position sigma 0.03 m, '
            'velocity sigma 0.015 m/s and 100 ms delay, with unperturbed robot state/map and privileged hard stops. '
            'The updated model and the two V1 models were rebuilt from packaged training-only records with identical checksums.\n\n'
            '![Fresh updated-bank comparison](../figures/continual-snn-coverage-v2.png)\n\n'
            '[Current model/card](continual_snn_v2/model_card.json), '
            '[checkpoint](continual_snn_v2/primary.npz), '
            '[training qualification](continual_snn_v2/episode_qualification.json), '
            '[portable data hashes](continual_snn_v2/reproduction_data.json).\n\n')
        transfer=[]
        for phase in ('terrain','heading','severe_noise','stale'):
            groups=lookup.get('coverage-v2-transfer-'+phase,{}).get('groups',[])
            if groups:
                row=groups[0];transfer.append(f'| {phase} | {row["tasks"]}/{row["trials"]} | {row["collisions"]} | {row["falls"]} |')
        if transfer:
            coverage+=('**Major remaining limitation: terrain/goal transfer.** These are additional matched V1 cases, '
                'not independent fresh V2 scenes.\n\n| Check | Tasks | Obstacle contacts | Falls |\n|---|---:|---:|---:|\n'+'\n'.join(transfer)+'\n\n'
                'The updated bank fails all 60 terrain/longer-goal cases, including flat controls. '
                'Six fixed goal/bounds probes show a selector/contract shift even without rough terrain: '
                'the updated bank passes 2/6 versus 6/6 for the scripted rule. Narrow-room cases fail; '
                'in the open room it passes goals 2.4/3.2 m but stops short for 1.7 m. '
                'The rule follows timed progress requests rather than testing accurate arrival at a destination. '
                'These probes changed goals/bounds for diagnosis and do not repair the original failed transfer benchmark. '
                'Current success claims concern the specified flat-ground request-stream and obstacle families.\n\n')
    text=f'''# Continuous SNN selection: research and validation

October 17, 2026. One agent worked in this chat. Session: **{session['start_utc']} to {end}**.
The requested three hours refer to wall-clock work, not simulated robot time.

The working improvement is fresh count-based skill selection throughout an
episode, with a separately checked physical handoff. The selected encoder has
**64 sensory + 16 predictive LIF neurons** and a learned count-pattern bank.
The earlier ridge readout and compact original-16 bank remain comparisons. Encoder
weights remain fixed. Current observations use privileged MuJoCo state/map;
camera perception is future work. Jump selection currently uses an explicit
jump-request input and the existing short forward hop.

## Fresh results

'''
    text+=coverage
    if 'final-main-v1' in lookup:
        main_rows=[dict(arm=r['arm'],summary={k:v for k,v in r.items() if k not in ('batch','arm')})
            for r in records if r['batch']=='final-main-v1']
        text+='### Initial frozen study (V1)\n\nThe main procedural task-stream plan declares 20 new seed blocks, eight\nscene families, and the same motor skills/guards for all arms.\n\n'+table(lookup['final-main-v1']['groups'])+'\n\n'
        for before in ('old_repeat','new_once'):
            p=paired(main_rows,before,'new_repeat');lo,hi=p['seed_cluster_bootstrap_percentile95']
            text+=(f'Against `{before}`, paired success change is {100*p["mean_success_difference"]:.1f} '
                f'percentage points; descriptive seed-block bootstrap 95% interval '
                f'[{100*lo:.1f}, {100*hi:.1f}], {p["matched_episodes"]} matched episodes '
                f'in {p["independent_seed_blocks"]} seed blocks.\n\n')
        text+=('A zero-width bootstrap interval means the sampled seed blocks had identical '
               'success patterns. It does not quantify uncertainty over untested scene families.\n\n')
    for phase,title in (('stress','Unseen stress scenes'),('confirmation','Separate confirmation'),
            ('noise','Obstacle noise and 100 ms delay'),('severe_noise','Stronger noise and 200 ms delay'),
            ('heading','Broad-heading transfer'),('terrain','Fresh terrain geometry'),('stale','Stale-input diagnostic')):
        name='final-'+phase+'-v1'
        if name in lookup:
            text+='### '+title+'\n\n'+table(lookup[name]['groups'])+'\n\n'
            if phase in ('noise','severe_noise'):
                cfg=freeze['planned_batches'][phase]['sensor']
                text+=(f'Obstacle-position Gaussian sigma {cfg["position_noise"]} m; '
                       f'velocity sigma {cfg["velocity_noise"]} m/s; delay {cfg["delay"]} s. '
                       'Robot state and the terrain map remain unperturbed.\n\n')
    for phase,planned in freeze['planned_batches'].items():
        actual=lookup.get('final-'+phase+'-v1',{}).get('episodes',0)
        expected=len(planned['cases'])*len(planned['arms'])
        if actual<expected:
            text+=f'**Incomplete/running `{phase}`:** {actual}/{expected} planned episodes recorded.\n\n'
    text+=f'''![Measured task outcomes](../figures/continual-snn-outcomes.png)

There are **{totals['final_episodes']} V1 final matched physical episodes** in the
currently recorded batches and {totals['final_unique_scene_cases']} distinct
named final scene cases. Noise/stale runs reuse specified main scenes; they are
paired perturbation tests, not extra independent environments. All logged
development/training/evaluation cohorts total {totals['all_episodes']} episodes;
unit tests and the separately recorded demo are excluded from that tally.
V2 adds {totals.get('v2_fresh_confirmation_episodes',0)} fresh-confirmation episodes on
{totals.get('v2_unique_named_confirmation_cases',0)} named scene cases; noise repeats are paired perturbations.
V1 final obstacle contacts: {totals['final_contacts']}; final falls: {totals['final_falls']}.
These totals include failures. A completed episode can fail its requested task.

The main task criteria require 0.4 m progress (1 cm acceptance tolerance) where requested, the specified
number of verified hops, supported stable final hold, no failed/excess hops,
and no hold violation after a 2.5 s braking grace. Terrain transfer instead
requires **1.55 m**, with the same tolerance, to traverse the course. A hop requires
>=40 ms of all-feet flight, foot-site heights >2.5 cm, and stable recovery.
Episodes stop on obstacle contact, low clearance,
height/tilt limits, missing eligible fallback, excessive switching, or timeout.
Those guard stops remain failures, even when contact is avoided.
Hard stops retain privileged current MuJoCo geometry. Obstacle noise/delay
tests perturb the selector/request masks inside that guard system; they do not
validate a complete camera-based sensing and safety pipeline.

The rule/analog rows are essential controls. Gains over the old bank include
better observations, motion-state coverage, fitted associations and repeated
execution. They do not establish that spiking neurons outperform a non-spiking
selector, nor do they establish general navigation or hardware safety.

## What changed

```mermaid
flowchart LR
  O[Current robot and obstacle observations] --> F[22 sensory and 2 predictive features]
  F --> N[64 sensory and 16 predictive frozen LIF neurons]
  N --> C[Fresh 80-neuron spike-count vector]
  C --> A[Learned readout or association bank]
  A --> H[Eligibility and physical handoff supervisor]
  O --> H
  H --> P[Existing skill and joint PID]
  P --> R[Persistent MuJoCo robot]
  R --> O
```

Every 100 ms neural inference starts with fresh counts and zero membrane state,
runs 16 constant-input internal LIF updates, and records active IDs plus all
per-neuron counts. IDs form an activity set, not the chronological order of
spikes. Internal updates have no asserted biological duration. The first 64
neurons preserve the separately validated sensory population exactly; the first
16 also preserve the historical checkpoint. Added sensory seed: 20261031;
predictive seed: 20261101. All actual weights/biases and checksums are saved.

The two extra inputs are minimum predicted run/jump clearance in metres. They
use **all currently observed obstacles**, constant measured velocity, arena
bounds, a 1.6 s run horizon and 3 s hop reservation. No future scripted stopping
or reversal time is supplied. A unit counterexample proves that nearest-obstacle
features alone can be identical while a farther approaching obstacle changes
the feasible action. More neurons cannot reconstruct omitted information.

The V1 learned readout is a group/class-weighted ridge fit, alpha 0.0001. Its
softmax score is uncalibrated. Entirely silent spike packets abstain. The
V1 prototype bank has {model['association_patterns']} unique 80-neuron count
patterns, {model['conflicting_patterns']} exact training label conflicts,
one nearest neighbor and maximum count-vector distance 12. Exact conflicts,
silence and distant unknown patterns abstain; no associations update during
evaluation. No new motor policy was trained.

A supervisor requires two confirming packets, avoids restarting the same skill,
and rechecks the latest request after braking. Entry needs four-foot supported
stability (linear speed <0.04 m/s, angular speed <0.2 rad/s). A hop can cancel
before thrust; airborne motion retains landing/recovery commitment. A sustained
jump request triggers one hop until released. Stale input requests checked
wait/braking. Missing eligible fallback terminates the episode. Clearing a
spike packet never resets the robot, PID, telemetry or movement commitment.
Identical requests during active execution preserve the current primitive.
After a braking interruption, the same skill can be re-entered from stable
support; such re-entry is logged separately from a change of skill class.

Physics/PID remain 500 Hz, target updates 100 Hz, and selection 10 Hz. Bounds
remain **24 N m run/wait and 28 N m short hop**. The separate 72 N m obstacle
jump remains outside this selector. The new package is opt-in; the historical
navigation SNN and older one-shot demo remain available.

## Data, selection and rejected alternatives

Training: **19,380 rows in 222 context groups**, comprising 94 historical
measured-outcome contexts and 128 causal teacher trajectories (seeds 4100–4115).
Validation: **4,850 rows in 46 groups**, comprising 14 historical contexts and
32 teacher trajectories (5100–5103). Each whole trajectory/context stays in
one split. Training normalization and context/class balancing use training
data only. Teacher requests are explicitly labelled separately from measured
motor outcomes; they are not verified counterfactual outcomes at every frame.
This experiment learns the current rule/task contract through a spiking
representation. Learning obstacle affordances from raw images and collecting
verified outcomes for each possible skill remain separate milestones.

The final model/schema/runtime were frozen at **{freeze['frozen_utc']}**.
Main seeds: 6100–6119; confirmation: 7100–7109; stress: 8100–8109;
broad headings: 9200–9209; terrain generator seed: 20261117.
Development stress seeds 8200/8201 were exposed before V1 selection. V1 models
were never refitted after their freeze. V1 stress failures later motivated V2's
new training-only cases; V2 final results were not used for fitting or tuning.
Demonstrations replay confirmed cases and supply no independent extra evidence.
The planned 09:55 UTC cutoff stops before a new matched case group; any
unexecuted cases are disclosed in the batch's `stopped.json`.

The comparison evaluated 90 initial readout settings (counts, full/filtered
rasters, 16–256 neurons, LIF/ALIF/synaptic dynamics and analog features),
30 separately refitted reset/retained-state settings, 24 prototype settings,
and 16 predictive-population readout/prototype settings. Counts alone in the
old 16-neuron population achieved 20/32 physical validation tasks; full raster
16 achieved 31/32, filtered 16 achieved 30/32, while 64-neuron counts achieved
32/32. More elaborate timing was therefore unnecessary for the selected
default, though it helped the smaller encoder. Increasing to 256 neurons did
not consistently improve stress performance. Adaptive thresholds and retained
state added no consistent physical benefit; one retained-ALIF variant dropped
to 29/32 under noise while the matched reset alternatives completed 32/32.
These are bounded negative results, not universal rejections of memory or ALIF.

An exact-pattern audit quantified why counts matter: original 16-neuron active
ID sets have ambiguous labels across 11,940 training observations, versus 249
when full counts are retained. The 64-neuron count representation reduces that
to seven observations; the predictive 80-neuron representation has zero exact
conflicts in the recorded train/validation sets. Affected-row counts include
every row in a mixed-label bucket; they are not counts of actual classification errors.
Zero observed conflict does
not imply perfect unseen classification. Full tables and examples remain in
`output/continual_snn/pattern-aliasing.json`.

Keep the count-based readout, explicit handoff semantics and versioned
association alternative. Leave retained-state/ALIF/larger-population variants
as diagnostics. Prioritize outcome-labelled coverage of oscillating obstacles,
terrain and ambiguous motion states, and independently validate retreat/turn
skills for unavoidable approaches. Current-velocity extrapolation, explicit
task requests, privileged sensing and the limited motor inventory remain
material constraints. Failed logs are retained; unsupported ideas are removed
from the recommendation rather than erased from the record.

{ablation}

The compact 16-neuron alternative stores 1,335 unique count signatures in an
11.5 KB artifact. It preserves duplicate training-row multiplicities, all
distance ties, exact mixed-label conflicts and abstention. These multiplicities
are separate from the number of spikes within an observation packet. It matched
the source decoder on all 2,979 checked signatures/perturbations and produced
bit-identical 500 Hz physics traces in five paired scenes, including a failed
oscillating-gate case. The data-only comparison includes both added state/task
coverage and altered sampling multiplicity; it does not isolate those two
effects from each other.

## Research and verification

The implementation draws on **options/initiation/termination**
([Sutton et al., 1999](https://www.sciencedirect.com/science/article/pii/S0004370299000521)),
**fixed spiking dynamics plus a learned readout**
([Maass et al., 2002](https://pubmed.ncbi.nlm.nih.gov/12433288/)),
**filtered spike-train distances**
([van Rossum, 2001](https://era.ed.ac.uk/bitstream/handle/1842/226/distance.pdf?sequence=1)),
and **adaptive LIF/memory**
([Bellec et al., 2018](https://papers.nips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html)).
E-prop ([Bellec et al., 2020](https://www.nature.com/articles/s41467-020-17236-y))
and [surrogate gradients](https://nmi-lab.org/ielaam/79/8887548/8891809-aam.pdf)
remain future supervised-sequence learning options. Our constant-input
feedforward encoder is not a liquid-state machine. No external policy code,
new dependency or imported motor policy was added. See the
[research brief](continual_snn_research.md) for source interpretation.

The released runtime passed **176 project tests** with the project Python
environment and project-local TEMP/TMP; existing HTTP tests used loopback
permission. `full-suite-compact-release.log` records the exact run. Tests cover packet
reset, silent/stale/conflicting inputs, strict old/new schemas, latching,
braking/revalidation, no repeated-skill restart, pre-thrust cancellation,
exception cleanup, actuator-force consistency and persistent physical state.
The final audit checked **7,751 trace archives**, **19,246 handoff/re-entry
records** and **895,972 saved-model predictions** with no integrity, pairing
or source-snapshot failures. Recorded handoff linear/angular speed maxima were
0.01471 m/s and 0.02235 rad/s. All four historical checkpoint/bank files retain
their original hashes. The single aborted prelaunch folder is recorded
explicitly; all subsequent matched groups completed. Three early development
cohorts have source hashes without full source copies and are identified as
such. See [audit summary](continual_snn_v2/audit_summary.json) and the complete
`output/continual_snn/final-audit.json`.

For the updated bank, model-call medians were 0.12 ms on regular streams,
3.00 ms on stress and 2.81 ms with noise; p95 was <=3.46 ms and the recorded
maximum was 5.35 ms. Exact-pattern lookup is faster than a neighbor search.
These timings exclude feature extraction/predictive geometry and file I/O.
They fit the observed 100 ms selection interval on this computer; they do not
establish a wall-clock guarantee for the nominal 2 ms PID loop or hardware.

Four full-rate V1/V2 demonstrations were independently replayed using only their
recorded motor commands and obstacle mocap updates. All **35,559 physics steps**
reproduced position and velocity with zero numerical error on this environment.
In these replays, all 13 handoff/re-entry events had four externally loaded feet (about 149 N total normal
load). No SNN, PID, state correction, or external force was used in that replay.
This is a check of the recorded physical mechanism, not a hardware claim.

The first evaluation launch exposed Windows path lookup and strict old-input
adapter errors before a completed held-out trial. Those failed launch logs and
the prelaunch freeze remain archived. The adapter received a regression test;
the corrected runtime was frozen before the reported final comparisons.

A separate nine-episode approaching-obstacle probe changed the obstacle's
observed velocity during jump preparation while the explicit jump request
remained active. All nine logged cancellation before thrust; six completed
cancelled preparation with stable recovery, and three reached a clearance
guard stop sooner. No recorded trace sample lost all foot contact. All nine
remain failed tasks because the continued approach left no usable escape.
The frozen model/controller were unchanged. This checks cancellation and
documents a missing capability; it is not counted as successful navigation.

![Physical handoffs and count packets](../figures/continual-snn-handoffs.png)

The recorded run/jump/run/hold example performs four skill changes in one
persistent episode, one verified hop and stable recovery. Braking and recovery
take physical time: a fresh neural choice does not mean an instantaneous
joint-controller replacement.

## Files and commands

- [Earlier V1 model and provenance](continual_snn_v1/model_card.json),
  [V1 readout checkpoint](continual_snn_v1/primary.npz),
  [explicit association bank](continual_snn_v1/association_memory.npz).
- [All cohort totals](continual_snn_v1/cohorts.csv),
  [every failed task with failure signals](continual_snn_v1/failures.csv).
- [Compact original-16 count bank](continual_snn_v1/compact_motion_bank16.npz),
  [portable training-data hashes and label provenance](continual_snn_v1/reproduction_data.json).
- [Code/observation contract and usage](../../experiments/continual_snn/README.md),
  [three-hour protocol](snn_three_hour_protocol.md).
- Raw traces, per-packet counts, observed inputs, requests, handoffs, models,
  case manifests and source snapshots: `output/continual_snn/` (generated data).

Actual main-session commands (existing output directories are intentionally
protected from overwrite):

```powershell
.venv-win/Scripts/python.exe -m unittest discover -s tests -v
.venv-win/Scripts/python.exe -m experiments.continual_snn.evaluate --freeze --phases main stress confirmation noise severe_noise heading terrain stale
.venv-win/Scripts/python.exe -m experiments.continual_snn.audit --output output/continual_snn/final-audit.json
.venv-win/Scripts/python.exe -m experiments.continual_snn.report --output output/continual_snn/report-v1 --timeline output/continual_snn/demo-handoffs-v1
```

For a new finite live replay:

```powershell
.venv-win/Scripts/python.exe -m experiments.continual_snn.demo --viewer --video
```

This now defaults to the validated V2 count-pattern bank. Rebuild it from its
published training-only records with:

```powershell
.venv-win/Scripts/python.exe -m experiments.continual_snn.package_coverage --reproduce output/continual_snn/my-v2-reproduction
```

The original-16 alternative is available with
`--model docs/agents/continual_snn_v1/compact_motion_bank16.npz`.
All four released V1/V2 comparison/default models were rebuilt from packaged
training-only data with identical model checksums in the project environment.
Reproduce the V1 models with a new output directory:

```powershell
.venv-win/Scripts/python.exe -m experiments.continual_snn.reproduce --output output/continual_snn/my-reproduction
```
'''
    (ROOT/'docs/agents/continual_snn_validation.md').write_text(text,encoding='utf-8')
    print('Report updated:',totals,flush=True)


if __name__=='__main__':main()
