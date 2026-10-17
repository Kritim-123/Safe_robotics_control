"""Summarize actual completed cohorts without merging repeats into new scenes."""
import argparse,csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save


def rows_for(root,pattern):
    rows=[]
    for path in sorted(Path(root).glob(pattern+'/results.jsonl')):
        rows += [dict(json.loads(line),cohort=str(path.parent.relative_to(root))) for line in path.read_text().splitlines()]
    return rows


def aggregate(rows):
    result=[]
    for arm in dict.fromkeys(r['arm'] for r in rows):
        group=[r['summary'] for r in rows if r['arm']==arm]
        delays=[s['arrival_stopping_delay'] for s in group if s.get('arrival_stopping_delay') is not None]
        result.append(dict(arm=arm,trials=len(group),tasks=sum(s['task_fulfilled'] for s in group),
            completed=sum(s['episode_completed'] for s in group),contacts=sum(s['collision'] for s in group),
            falls=sum(s['fall'] for s in group),terrain_refusals=sum(s.get('terrain_refusal_latched',False) for s in group),
            bound_violations=sum(s['max_torque_limit_ratio']>1.+1e-9 for s in group),
            switch_count=sum(s['switches'] for s in group),verified_hops=sum(s['completed_jumps'] for s in group),
            median_arrival_delay=float(np.median(delays)) if delays else None,
            maximum_arrival_delay=max(delays) if delays else None))
    return result


def terrain_family(scenario):
    names={p['name'] for p in scenario.get('terrain',[])}
    if 'finish_threshold' in names:return 'mixed_course'
    if 'cross_slope' in names:return 'cross_slope'
    if 'reduced_friction' in names:return 'friction_changes'
    if any(n.startswith('block_') for n in names):return 'uneven_blocks'
    if any(n.startswith('step_') for n in names):return 'steps'
    if 'ramp_up' in names:return 'ramps'
    if 'threshold' in names:return 'thresholds'
    obstacles={o['name'] for o in scenario.get('obstacles',[])}
    if 'crossing' in obstacles:return 'moving_crossing'
    if obstacles:return 'static_passage'
    return 'flat'


def failure_signals(s):
    reasons=[]
    if not s['episode_completed']:reasons.append(s['reason'])
    if s['collision']:reasons.append('contact')
    if s['fall']:reasons.append('fall')
    if s.get('terrain_refusal_latched'):reasons.append('sampled_terrain_outside_motor_envelope')
    if s['progress']<s['required_progress']-.01:reasons.append('insufficient_progress')
    if s.get('final_goal_distance',0.)>.12 and s.get('adaptation_config',{}).get('goal_arrival'):reasons.append('goal_distance')
    if 'first_goal_arrival' in s and s.get('adaptation_config',{}).get('goal_arrival'):
        if s['first_goal_arrival'] is None:reasons.append('no_supported_goal_wait_during_progress')
        elif s['arrival_stopping_delay']>s['adaptation_config'].get('arrival_max_delay',6.):reasons.append('late_goal_wait')
    if s.get('post_arrival_departure_steps',0):reasons.append('departed_after_arrival')
    if s['hold_violation_packets']:reasons.append('hold_motion')
    if s['failed_jumps'] or s['completed_jumps']<s['required_jumps']:reasons.append('hop_requirement')
    if not s['stable_recovery']:reasons.append('unstable_terminal_hold')
    return ';'.join(dict.fromkeys(reasons)) or 'other_task_acceptance'


def collect(root,output):
    root=Path(root);output.mkdir(parents=True,exist_ok=True)
    phases={
        'baseline':rows_for(root,'baseline-v1'),
        'neural_only':rows_for(root,'final-neural-only-ablation-v1'),
        'original_benchmark':rows_for(root,'final-original-benchmark-v1'),
        'original_benchmark_v2':rows_for(root,'final-original-benchmark-v2'),
        'physical_parity':rows_for(root,'final-physical-parity-v2'),
        'v1_main':rows_for(root,'final-main-chunk-*'),
        'v1_mild':rows_for(root,'final-noise-v1'),
        'v1_severe':rows_for(root,'final-severe-noise-v1'),
        'v1_ood':rows_for(root,'final-ood-v1'),
        'regression':rows_for(root,'final-selector-regression-v1'),
        'gait_regression':rows_for(root,'final-gait-regression-v1'),
        'stale':rows_for(root,'final-stale-v1'),
        'negative':rows_for(root,'final-negative-controls-v1'),
        'data_ablation':rows_for(root,'representation-data-ablation-v6'),
        'v2_clean':rows_for(root,'hysteresis-final-clean-v2'),
        'v2_mild':rows_for(root,'hysteresis-final-mild-v2'),
        'v2_severe':rows_for(root,'hysteresis-final-severe-v2'),
        'v2_ood_diagnostic':rows_for(root,'frozen-v2-ood-diagnostic'),
        'refinement_validation':rows_for(root,'request-refinement-validation-v3'),
        'refinement_regression':rows_for(root,'request-refinement-regression-v3'),
        'refinement_benchmark':rows_for(root,'request-refinement-benchmark-v3'),
        'refinement_confirmation':rows_for(root,'request-refinement-confirmation-v3-*'),
        'refinement_noise':rows_for(root,'request-refinement-noise-v3')}
    for condition in ('minus8cm','minus4cm','plus4cm','plus8cm'):
        phases['goal_bias_'+condition]=rows_for(root,'fixed-goal-bias-v1/'+condition)
    # A single interrupted rule attempt has no terminal result. Its complete
    # matched replay replaces both arms for the same case; no duplicate scene.
    replay=rows_for(root,'ood-interruption-replay-v1')
    if replay:
        replaced={r['summary']['case'] for r in replay}
        phases['v1_ood']=[r for r in phases['v1_ood'] if r['summary']['case'] not in replaced]+replay+rows_for(root,'final-ood-continuation-v1')
    totals={name:aggregate(rows) for name,rows in phases.items()}
    save(output/'phase_summary.json',totals)
    with (output/'cohorts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['phase','arm','trials','tasks','completed','contacts','falls','terrain_refusals','bound_violations','switch_count','verified_hops','median_arrival_delay','maximum_arrival_delay'])
        writer.writeheader()
        for phase,groups in totals.items():
            for group in groups:writer.writerow(dict(phase=phase,**group))
    failures=[];all_count=0;exceptions=[]
    for path in sorted(root.rglob('results.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line);s=r['summary'];all_count+=1
            if not s['task_fulfilled']:
                failures.append(dict(cohort=str(path.parent.relative_to(root)),arm=r['arm'],case=s['case'],seed=s['seed'],
                    signals=failure_signals(s),progress=s['progress'],goal_distance=s.get('final_goal_distance'),
                    contact=s['collision'],fall=s['fall'],raw_reason=s['reason']))
    for path in root.rglob('error.json'):
        if 'source_snapshot' not in path.parts:exceptions.append(dict(path=str(path.relative_to(root)),error=json.loads(path.read_text())))
    with (output/'failed_tasks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['cohort','arm','case','seed','signals','progress','goal_distance','contact','fall','raw_reason'])
        writer.writeheader();writer.writerows(failures)
    save(output/'attempt_accounting.json',dict(completed_logged_episodes=all_count,failed_completed_logs=len(failures),aborted_episode_errors=exceptions,
        note='Episodes include paired controls, repetitions, training and development. They are not independent environments.'))
    by_family={}
    for phase in ('v1_main','v2_clean','v2_mild','v2_severe'):
        groups=defaultdict(list)
        cases={}
        for row in phases[phase]:
            cohort=root/row['cohort']
            if cohort not in cases:cases[cohort]={r['scenario']['name']:r['scenario'] for r in json.loads((cohort/'cases.json').read_text())}
            groups[terrain_family(cases[cohort][row['summary']['case']])].append(row)
        by_family[phase]={family:aggregate(rows) for family,rows in groups.items()}
    save(output/'terrain_family_summary.json',by_family)
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'output/terrain_goal')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)


if __name__=='__main__':main()
