"""Fixed factorial diagnostic of goal/bounds transfer; never relabel benchmarks."""
from dataclasses import replace
import json
from .models import ROOT
from .loading import load_selector
from .transfer import terrain_cases
from .study import save,run_batch
from .evaluate import DEADLINE,verify


def main():
    verify();base=ROOT/'output/continual_snn';original=terrain_cases()[0];scenes=[]
    for room,bounds in (('narrow',original.scenario.bounds),('open',(-5.,10.,-5.,5.))):
        for goal in (1.7,2.4,3.2):
            spec=replace(original.scenario,name=f'goal-probe-{room}-{goal:g}',goal=(goal,0.),bounds=bounds)
            scenes.append(replace(original,scenario=spec,split='goal_bounds_diagnostic'))
    arms=[('v1_readout80','continuous',load_selector(ROOT/'docs/agents/continual_snn_v1/primary.npz')),
          ('v1_compact16','continuous',load_selector(ROOT/'docs/agents/continual_snn_v1/compact_motion_bank16.npz')),
          ('v2_selected','continuous',load_selector(ROOT/'docs/agents/continual_snn_v2/primary.npz')),
          ('rule','rule',None)]
    save(base/'goal-probe-protocol.json',dict(
        rationale='Compact16 stopped at 0.91 m in flat transfer controls; separate task/goal/bounds shift from terrain difficulty',
        factors=dict(goal_metres=[1.7,2.4,3.2],room=['narrow','open']),
        changed_model_parameters=False,physical_skill_and_torque_changes=False,
        original_transfer_failures_retained=True,interpretation='Six fixed post-hoc diagnostics; changed goals are not benchmark improvements',
        source_case=original.to_dict(),cases=[c.to_dict() for c in scenes]))
    output=base/'goal-bounds-probe-v1';run_batch(output,scenes,arms,deadline=DEADLINE)
    rows=[json.loads(s) for s in (output/'results.jsonl').read_text().splitlines()];examples=[]
    for row in rows:
        s=row['summary']
        if row['arm']=='rule':continue
        for line in (output/s['case']/row['arm']/'decisions.jsonl').read_text().splitlines():
            item=json.loads(line)
            if item['task']=='progress' and item['desired']=='wait' and item['time']>4.:
                examples.append(dict(case=s['case'],arm=row['arm'],time=item['time'],
                    goal_forward=item['features']['goal_forward'],footprint_clearance=item['features']['footprint_clearance'],
                    teacher_request=item['teacher_request'],proposal=item['proposal'],final_progress=s['progress']))
                break
    save(output/'first_wait_during_progress.json',examples)


if __name__=='__main__':main()
