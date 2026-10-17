"""A fresh approach during preparation: verify cancellation, not task success."""
from dataclasses import replace
import argparse
import json
import numpy as np
from .models import ROOT,load_model
from .scenes import make_case,Request
from hybrid.arena import Obstacle
from .study import run_batch,save


def main(audit_only=False):
    base=ROOT/'output/continual_snn';scenes=[]
    for seed,speed in ((8300,.6),(8301,.8),(8302,1.0)):
        original=make_case('jump_then_run',seed,'hazard_probe')
        heading=np.array([np.cos(original.scenario.start_yaw),np.sin(original.scenario.start_yaw)])
        start=np.array(original.scenario.start)
        obstacle=Obstacle('new_approach',tuple(start+heading*2.5),radius=.2,
            velocity=tuple(-heading*speed),start_after=2.8)
        spec=replace(original.scenario,name=f'hazard-preparation-{seed}',obstacles=(obstacle,))
        case=replace(original,scenario=spec,requests=(Request(2.,10.,'jump'),Request(10.,35.,'hold')),
            required_progress=0.,required_jumps=0,maximum_jumps=0,finish_after=15.,timeout=21.)
        scenes.append(case)
    arms=[('new_repeat','continuous',load_model(ROOT/'docs/agents/continual_snn_v1/primary.npz')),
          ('prototype80','continuous',load_model(ROOT/'docs/agents/continual_snn_v1/association_memory.npz')),
          ('rule','rule',None)]
    output=base/'hazard-preparation-probe-v1'
    if not audit_only:run_batch(output,scenes,arms)
    rows=[json.loads(s) for s in (output/'results.jsonl').read_text().splitlines()]
    violations=[];cancellations=0;requests=0;zero_contact_samples=0
    for row in rows:
        s=row['summary'];cancellations+=s['cancelled_jumps']
        if s['completed_jumps'] or s['failed_jumps'] or s['collision'] or s['fall']:
            violations.append((row['arm'],s['case']))
        if s['max_torque']>28.+1e-9:violations.append((row['arm'],s['case'],'torque'))
        episode=output/s['case']/row['arm']
        events=[json.loads(line) for line in (episode/'events.jsonl').read_text().splitlines()]
        starts=[e['time'] for e in events if e['event']=='handoff' and e['next']=='jump']
        cancelled=[e['time'] for e in events if e['event']=='jump_preparation_cancelled']
        requests+=len(cancelled)
        if len(starts)!=1 or len(cancelled)!=1 or not 0<=cancelled[0]-starts[0]<.55:
            violations.append((row['arm'],s['case'],'missing pre-thrust cancellation event'))
        with np.load(episode/'trace.npz') as trace:
            columns=trace['columns'].tolist()
            zero_contact_samples+=int(np.sum(trace['values'][:,columns.index('foot_contacts')]==0))
    report=dict(episodes=len(rows),pre_thrust_cancellation_requests=requests,
        cancelled_preparations_with_stable_recovery=cancellations,
        cancellation_recovery_incomplete_at_guard_stop=requests-cancellations,
        all_feet_no_contact_trace_samples=zero_contact_samples,violations=violations,
        task_fulfillment_claim=False,expectation='Cancel/refuse takeoff as a new approach invalidates the reserved corridor; subsequent missing-fallback stops remain task failures',
        model_or_parameter_updates=False)
    save(output/'probe_assertions.json',report)
    if violations or not cancellations:raise AssertionError(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true')
    main(parser.parse_args().audit_only)
