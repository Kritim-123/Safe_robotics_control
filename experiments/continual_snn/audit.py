"""Recompute physical evidence, paired outcomes and archive integrity from files."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .models import ROOT,FrozenBankModel
from .loading import load_selector as load_model
from .study import save


def paired(rows,reference,comparison):
    arms=defaultdict(dict)
    for row in rows:
        s=row['summary'];arms[row['arm']][s['case']]=s
    common=sorted(set(arms[reference])&set(arms[comparison]))
    clusters=defaultdict(list);table=Counter()
    for case in common:
        a,b=arms[reference][case],arms[comparison][case]
        difference=int(b['task_fulfilled'])-int(a['task_fulfilled'])
        clusters[a['seed']].append(difference)
        table[str((bool(a['task_fulfilled']),bool(b['task_fulfilled'])))]+=1
    values=np.asarray([np.mean(v) for v in clusters.values()])
    rng=np.random.default_rng(20261017)
    # Families sharing a seed share initial random draws; resample seed blocks.
    boot=np.mean(rng.choice(values,size=(10000,len(values))),axis=1) if len(values) else np.zeros(1)
    return dict(reference=reference,comparison=comparison,matched_episodes=len(common),
        independent_seed_blocks=len(values),paired_table=dict(table),
        mean_success_difference=float(np.mean(values)) if len(values) else None,
        seed_cluster_bootstrap_percentile95=np.quantile(boot,[.025,.975]).tolist(),
        caution='Descriptive interval conditional on these procedural scene families; not hardware/general navigation evidence')


def audit_batch(folder,*,replay_cases=1):
    folder=Path(folder)
    rows=[json.loads(s) for s in (folder/'results.jsonl').read_text().splitlines()]
    failures=[];handoffs=packets=traces=replayed=0;max_linear=max_angular=0.
    telemetry=defaultdict(list);family=defaultdict(Counter);counts=Counter()
    loaded={};replay_count=Counter()
    expected_cases=json.loads((folder/'cases.json').read_text())
    expected_arms=[r['name'] for r in json.loads((folder/'arms.json').read_text())]
    recorded_keys=[(r['summary']['case'],r['arm']) for r in rows]
    if len(recorded_keys)!=len(set(recorded_keys)):failures.append('duplicate episode result records')
    seen=defaultdict(set)
    for case,arm in recorded_keys:seen[case].add(arm)
    complete_groups=sum(arms==set(expected_arms) for arms in seen.values())
    partial_groups={case:sorted(set(expected_arms)-arms) for case,arms in seen.items() if arms!=set(expected_arms)}
    unfinished=[str(p.parent.relative_to(folder)) for p in folder.glob('*/*/decisions.jsonl')
                if not (p.parent/'summary.json').exists()]
    orphaned=[str(p.parent.relative_to(folder)) for p in folder.glob('*/*/summary.json')
              if (p.parent.parent.name,p.parent.name) not in set(recorded_keys)]
    manifest=json.loads((folder/'manifest.json').read_text())
    source_failures=[]
    source_recorded=(folder/'source_snapshot').exists()
    for source,checksum in manifest['source_sha256'].items():
        snapshot=folder/'source_snapshot'/source
        if source_recorded and (not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=checksum):
            source_failures.append(source)
    for row in rows:
        arm=row['arm'];s=row['summary'];episode=folder/s['case']/arm
        family[arm][s['family']+'/trials']+=1
        family[arm][s['family']+'/successes']+=int(s['task_fulfilled'])
        counts[arm]+=1
        if s['max_torque_limit_ratio']>1+1e-10:failures.append(s['case']+'/'+arm+':torque ratio')
        with np.load(episode/'trace.npz',allow_pickle=False) as bundle:
            values=bundle['values'];columns=bundle['columns'].tolist()
        traces+=1
        if values.shape[1]!=len(columns) or not np.isfinite(values).all():failures.append(s['case']+'/'+arm+':trace schema/finiteness')
        if np.any(np.diff(values[:,0])<=0):failures.append(s['case']+'/'+arm+':nonmonotonic physics time')
        if np.max(np.abs(values[:,[columns.index('torque_'+str(i)) for i in range(12)]]))>28+1e-9:
            failures.append(s['case']+'/'+arm+':trace torque')
        if s['minimum_clearance']>min(values[:,columns.index('clearance')])+1e-9:
            failures.append(s['case']+'/'+arm+':clearance summary')
        for event in map(json.loads,(episode/'events.jsonl').read_text().splitlines()):
            if event['event']=='handoff':
                handoffs+=1
                linear=np.linalg.norm(event['qvel'][:3]);angular=np.linalg.norm(event['qvel'][3:6])
                max_linear=max(max_linear,float(linear));max_angular=max(max_angular,float(angular))
                if linear>=.04+1e-9 or angular>=.2+1e-9:
                    failures.append(s['case']+'/'+arm+':unstable handoff')
        decisions=[json.loads(x) for x in (episode/'decisions.jsonl').read_text().splitlines()]
        packets+=len(decisions)
        model_path=folder/'models'/(arm+'.npz')
        old_path=folder/'models'/(arm+'_encoder.npz')
        replay=(model_path.exists() or old_path.exists()) and replay_count[arm]<replay_cases
        if replay:
            if arm not in loaded:
                if model_path.exists():loaded[arm]=load_model(model_path)
                else:
                    from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
                    from experiments.spike_pattern_skills.associations import AssociationMemory
                    model=FrozenBankModel();model.encoder=DenseTerrainEncoder.load(old_path)
                    model.memory=AssociationMemory.load(folder/'models'/(arm+'_associations.json'),model.encoder.checksum)
                    model.checksum=model.encoder.checksum+':'+model.memory.checksum;loaded[arm]=model
            model=loaded[arm];model.reset()
            if model.checksum!=s['model_checksum']:failures.append(s['case']+'/'+arm+':model provenance')
            replay_count[arm]+=1
        for decision in decisions:
            if not decision['model_evaluated']:continue
            spike_counts=decision['proposal'].get('spike_counts')
            if spike_counts is not None:
                expected_ids=[i for i,c in enumerate(spike_counts) if c>0]
                if expected_ids!=decision['proposal'].get('active_neuron_ids'):
                    failures.append(s['case']+'/'+arm+':active IDs/counts differ')
            telemetry[arm].append(decision['inference_seconds']*1000)
            if replay:
                result=model.predict(decision['features'],now=decision['time'])
                for name in ('proposed_skill','spike_counts','active_neuron_ids'):
                    if result.get(name)!=decision['proposal'].get(name):
                        failures.append(s['case']+'/'+arm+':prediction replay '+name);break
                replayed+=1
    latency={arm:dict(packets=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),
                     maximum_ms=float(max(v))) for arm,v in telemetry.items()}
    pairs=[]
    if len(counts)>1:
        reference=next(iter(counts))
        pairs=[paired(rows,reference,arm) for arm in list(counts)[1:]]
    return dict(batch=folder.name,episodes=len(rows),trace_archives=traces,decision_packets=packets,
        planned_cases=len(expected_cases),planned_arms=len(expected_arms),complete_matched_case_groups=complete_groups,
        partial_matched_case_groups=partial_groups,unfinished_episode_folders=unfinished,
        summaries_without_result_record=orphaned,
        unexecuted_cases=[r['scenario']['name'] for r in expected_cases if r['scenario']['name'] not in seen],
        checked_handoffs=handoffs,maximum_handoff_linear_speed=max_linear,
        maximum_handoff_angular_speed=max_angular,exact_replayed_predictions=replayed,
        integrity_failures=failures,source_snapshot_failures=source_failures,
        source_snapshot_status='recorded' if source_recorded else 'earlier development cohort: hashes only, source copy unavailable',
        latency=latency,by_family={a:dict(v) for a,v in family.items()},
        paired_comparisons=pairs)


def main(root,output,replay_cases):
    root=Path(root);result=[]
    for path in sorted(root.glob('*/results.jsonl')):
        report=audit_batch(path.parent,replay_cases=replay_cases);result.append(report)
        print(path.parent.name,report['episodes'],'failures',len(report['integrity_failures']),flush=True)
    session=json.loads((root/'session-v1/session.json').read_text())
    unchanged={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==checksum for p,checksum in session['files'].items()}
    save(output,dict(historical_artifacts_unchanged=unchanged,batches=result,
                     complete=sum(r['episodes'] for r in result),report_type='raw evidence audit; no refitting'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'output/continual_snn')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--replay-cases',type=int,default=1)
    args=parser.parse_args();main(args.root,args.output,args.replay_cases)
