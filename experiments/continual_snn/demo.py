"""Finite live continuous SNN demo with packet counts, handoffs and outcome logs."""
import argparse
from pathlib import Path
import time
from .models import ROOT
from .loading import load_selector as load_model
from .runner import execute
from .scenes import FAMILIES,STRESS_FAMILIES,make_case
from .stress_scenes import make_stress_case


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES+STRESS_FAMILIES,default='run_jump_run')
    parser.add_argument('--seed',type=int,default=11300)
    parser.add_argument('--model',type=Path,default=ROOT/'docs/agents/continual_snn_v2/primary.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'output/continual_snn'/('demo-'+time.strftime('%Y%m%d-%H%M%S')))
    parser.add_argument('--mode',choices=('continuous','one_shot'),default='continuous')
    parser.add_argument('--viewer',action='store_true');parser.add_argument('--video',action='store_true')
    args=parser.parse_args()
    factory=make_case if args.family in FAMILIES else make_stress_case
    case=factory(args.family,args.seed,'demo')
    print(execute(case,args.output,model=load_model(args.model),mode=args.mode,
        viewer=args.viewer,video=args.video,full_trace=True))
