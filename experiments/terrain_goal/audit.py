"""Independent file, handoff, model-prediction and preserved-baseline audit."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save
from .loading import load_selector_any


def model_index(root):
    index={};errors=[]
    paths=list(root.glob('fit*/*.npz'))+[ROOT/'docs/agents/continual_snn_v2/primary.npz']
    for path in paths:
        try:
            model=load_selector_any(path);index[model.checksum]=(path,model)
        except Exception as e:errors.append(dict(path=str(path),error=str(e)))
    return index,errors


def audit(root,output,*,replay_cases=2):
    root=Path(root);index,model_errors=model_index(root);cohorts=[]
    for result in sorted(root.rglob('results.jsonl')):
        folder=result.parent
        if not (folder/'cases.json').exists():continue
        rows=[json.loads(line) for line in result.read_text().splitlines()]
        manifest=json.loads((folder/'manifest.json').read_text())
        source_errors=[]
        for name,digest in manifest['source_sha256'].items():
            p=folder/'source_snapshot'/name
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:source_errors.append(name)
        arms=manifest.get('arms')
        if arms is None:arms=json.loads((folder/'arms.json').read_text())
        names=[a['name'] for a in arms];expected={c['scenario']['name'] for c in json.loads((folder/'cases.json').read_text())}
        seen=defaultdict(set);failures=[];handoffs=replayed=packets=0;max_linear=max_angular=0.
        replayed_cases=Counter();latencies=defaultdict(list)
        for row in rows:
            arm=row['arm'];s=row['summary'];case=s['case'];episode=folder/case/arm
            if arm in seen[case]:failures.append(f'{case}/{arm}:duplicate result')
            seen[case].add(arm)
            with np.load(episode/'trace.npz',allow_pickle=False) as data:
                values=data['values'];columns=data['columns'].tolist()
            if not np.isfinite(values).all() or np.any(np.diff(values[:,0])<=0):failures.append(f'{case}/{arm}:trace finiteness/time')
            torque=values[:,[columns.index('torque_'+str(i)) for i in range(12)]]
            allowed=np.where(values[:,columns.index('skill')]==2,28.,24.)
            if np.any(np.abs(torque)>allowed[:,None]+1e-8) or s['max_torque_limit_ratio']>1+1e-9:
                failures.append(f'{case}/{arm}:motor limit')
            disk=json.loads((episode/'summary.json').read_text())
            if disk!=s:failures.append(f'{case}/{arm}:summary differs from results')
            for event in map(json.loads,(episode/'events.jsonl').read_text().splitlines()):
                if event['event']!='handoff':continue
                handoffs+=1;v=float(np.linalg.norm(event['qvel'][:3]));w=float(np.linalg.norm(event['qvel'][3:6]))
                max_linear=max(max_linear,v);max_angular=max(max_angular,w)
                if v>=.04+1e-8 or w>=.2+1e-8:failures.append(f'{case}/{arm}:unstable handoff')
            model=None
            if s['neural_evaluations'] and replayed_cases[arm]<replay_cases:
                checkpoint=folder/'models'/(arm+'.npz')
                if checkpoint.exists():model=load_selector_any(checkpoint)
                elif s['model_checksum'] in index:model=index[s['model_checksum']][1]
                else:failures.append(f'{case}/{arm}:missing saved inference model')
                if model is not None:
                    if model.checksum!=s['model_checksum']:failures.append(f'{case}/{arm}:model checksum')
                    model.reset();replayed_cases[arm]+=1
            for decision in map(json.loads,(episode/'decisions.jsonl').read_text().splitlines()):
                packets+=1
                if not decision['model_evaluated']:continue
                proposal=decision['proposal'];counts=proposal.get('spike_counts')
                if counts is not None:
                    if ([i for i,n in enumerate(counts) if n>0]!=proposal.get('active_neuron_ids')
                        or not np.isin(counts,np.arange(17)).all() or not decision['packet_counts_cleared']):
                        failures.append(f'{case}/{arm}:invalid spike packet')
                latencies[arm].append(decision['inference_seconds']*1000)
                if model is not None:
                    actual=model.predict(decision['features'],now=decision['time'])
                    if any(actual.get(k)!=proposal.get(k) for k in ('proposed_skill','spike_counts','active_neuron_ids')):
                        failures.append(f'{case}/{arm}:prediction mismatch at {decision["time"]}')
                    replayed+=1
        partial={case:sorted(set(names)-a) for case,a in seen.items() if a!=set(names)}
        unfinished=[str(p.parent.relative_to(folder)) for p in folder.glob('*/*/decisions.jsonl') if not (p.parent/'summary.json').exists()]
        cohort=dict(batch=str(folder.relative_to(root)),episodes=len(rows),planned_cases=len(expected),arms=names,
            complete_matched_groups=sum(a==set(names) for a in seen.values()),partial_groups=partial,
            unexecuted_cases=sorted(expected-set(seen)),unfinished_episode_folders=unfinished,
            integrity_failures=failures,source_snapshot_failures=source_errors,checked_handoffs=handoffs,
            max_handoff_linear_speed=max_linear,max_handoff_angular_speed=max_angular,
            decision_packets=packets,exact_prediction_replays=replayed,
            latency={a:dict(packets=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(max(v))) for a,v in latencies.items()})
        cohorts.append(cohort)
        print(cohort['batch'],len(rows),'integrity',len(failures),'source',len(source_errors),flush=True)
    session=json.loads((root/'session.json').read_text())
    preserved={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in session['preserved_sha256'].items()}
    report=dict(cohorts=cohorts,model_load_errors=model_errors,preserved_baselines=preserved,
        completed_physical_episodes=sum(c['episodes'] for c in cohorts),
        checked_handoffs=sum(c['checked_handoffs'] for c in cohorts),
        exact_prediction_replays=sum(c['exact_prediction_replays'] for c in cohorts),
        limitations=['Two episodes per arm/batch replayed unless configured otherwise; teacher requests are not counterfactual outcomes',
            'First gait-search process uses candidate-000 runtime source bundle; later per-candidate on-disk snapshots are not asserted to be live imports'])
    save(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replay-cases',type=int,default=2)
    a=p.parse_args();audit(a.root,a.output,replay_cases=a.replay_cases)
