"""Keep concise hard-case review evidence without silently relabelling training."""
from collections import defaultdict,Counter
import json
from pathlib import Path
from .models import ROOT
from .report import FINAL_BATCHES,failure_signals
from .study import save


def main():
    base=ROOT/'output/continual_snn';destination=base/'report-v1';destination.mkdir(exist_ok=True)
    queue=[];packets=[];categories=Counter()
    studies=[(batch,'new_repeat','v1') for batch in sorted(FINAL_BATCHES)]
    studies += [('coverage-v2-confirm-'+phase,'v2_selected','v2') for phase in ('stress','regular','noise')]
    studies += [('goal-bounds-probe-v1','v2_selected','v2_diagnostic')]
    for batch,selected_arm,version in studies:
        folder=base/batch;path=folder/'results.jsonl'
        if not path.exists():continue
        groups=defaultdict(dict)
        for row in map(json.loads,path.read_text().splitlines()):groups[row['summary']['case']][row['arm']]=row['summary']
        for name,arms in groups.items():
            if selected_arm not in arms or 'rule' not in arms:continue
            selected,reference=arms[selected_arm],arms['rule']
            if selected['task_fulfilled']:continue
            category='selector_coverage_review' if reference['task_fulfilled'] else 'shared_capability_or_environment_review'
            categories[category]+=1
            queue.append(dict(batch=batch,case=name,family=selected['family'],seed=selected['seed'],selector_version=version,
                category=category,failure_signals=failure_signals(selected),
                selected_outcome=selected,matched_arm_outcomes={arm:dict(task_fulfilled=s['task_fulfilled'],
                    reason=s['reason'],progress=s['progress'],hops=s['completed_jumps']) for arm,s in arms.items()},
                approved_for_training=False,
                caution='Different controller trajectories are episode contrasts, not same-state counterfactual labels'))
            seen=set();chosen=[]
            for line in (folder/name/selected_arm/'decisions.jsonl').read_text().splitlines():
                row=json.loads(line);actual=row['proposal']['proposed_skill'];suggested=row['teacher_request']
                signature=tuple(row['proposal'].get('spike_counts') or ())
                normalized=actual.removeprefix('dynamic/') if actual else actual
                if normalized==suggested or signature in seen:continue
                seen.add(signature)
                chosen.append(dict(batch=batch,case=name,time=row['time'],features=row['features'],
                    spike_counts=list(signature),raw_proposal=actual,causal_teacher_request=suggested,
                    teacher_request_is_verified_outcome=False,approved_for_training=False,
                    selector_version=version,source=str((folder/name/selected_arm/'decisions.jsonl').relative_to(ROOT))))
            # Keep a spread through the episode rather than the first repeated frames.
            if len(chosen)>12:
                indices=[round(i*(len(chosen)-1)/11) for i in range(12)];chosen=[chosen[i] for i in indices]
            packets.extend(chosen)
    save(destination/'review-cases.json',dict(cases=queue,categories=dict(categories),
        automatically_added_training_rows=0,review_packets=len(packets)))
    with (destination/'review-packets.jsonl').open('w',encoding='utf-8') as stream:
        for row in packets:stream.write(json.dumps(row,allow_nan=False)+'\n')
    print('Review cases',len(queue),dict(categories),'diverse packets',len(packets),flush=True)


if __name__=='__main__':main()
