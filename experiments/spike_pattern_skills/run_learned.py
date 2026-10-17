"""Run the saved dense-terrain association model with live telemetry and optional video."""
import argparse
from pathlib import Path
import uuid

from .execute_associations import execute_case
from .learn_associations import write_json
from .outcome_cases import learning_cases
from .record import ROOT
from .refine_associations import fresh_cases, models

MODEL_BUNDLE = ROOT/'docs/agents/expanded_association_v2'


def run(bundle=MODEL_BUNDLE, cases=None, output=None, viewer=False, video=False,jump_profile='smooth_forward'):
    bundle = Path(bundle)
    if not bundle.is_absolute():
        bundle = ROOT/bundle
    encoder, memory = models(bundle)['dense']
    names = cases or ['new-terrain-00-flat-00','new-terrain-02-ramps-00','new-requested_hop-00']
    inventory = {c.scenario.name:c for c in learning_cases()+fresh_cases()}
    if len(set(names)) != len(names) or any(name not in inventory for name in names):
        raise ValueError('Require unique known case names from the outcome-learning dataset')
    output = Path(output) if output else ROOT/'output/spike_pattern_skills/learned-live'/uuid.uuid4().hex
    if not output.is_absolute():
        output = ROOT/output
    output.mkdir(parents=True,exist_ok=False)
    rows = []
    for name in names:
        case = inventory[name]
        result = execute_case(case,memory,encoder,output/name,viewer=viewer,video=video,
                              duration=case.duration,timeout=case.timeout,jump_profile=jump_profile)
        rows.append(dict(context_id=name,skill=result['summary']['skill'],reason=result['summary']['reason'],
                         task_fulfilled=result['outcome']['task_fulfilled']))
        write_json(output/'summary.json',dict(encoder_checksum=encoder.checksum,
            association_checksum=memory.checksum,episodes=rows,training=False,jump_profile=jump_profile))
        print(name,rows[-1],flush=True)
        if result['summary']['reason']=='viewer closed':
            break
    print('Recordings:',output,flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=MODEL_BUNDLE)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--jump-profile',choices=('legacy','smooth_forward'),default='smooth_forward')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--video',action='store_true')
    run(**vars(parser.parse_args()))
