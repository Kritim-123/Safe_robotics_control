"""Fit and package the monotonic goal-count association on training rows only."""
import argparse,json
from pathlib import Path
from .models import Selector
from .factored import fit_threshold
from experiments.continual_snn.study import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    train=[json.loads(s) for s in (a.data/'training_records.jsonl').read_text().splitlines()]
    val=[json.loads(s) for s in (a.data/'validation_records.jsonl').read_text().splitlines()]
    model,report=fit_threshold(Selector.load(a.base),train)
    model.save(a.output/'factored.npz');save(a.output/'threshold_training.json',report)
    counts={}
    for split,rows in (('train',train),('validation',val)):
        checked=correct=0
        for r in rows:
            if not r['features'].get('goal_mode'):continue
            pred=model.predict(r['features'])['proposed_skill'];checked+=1;correct+=pred==r['label']
        counts[split]=dict(packets=checked,correct=correct,accuracy=correct/max(checked,1))
    save(a.output/'packet_metrics.json',counts)
    print('threshold',model.threshold,'checksum',model.checksum,'metrics',counts,flush=True)


if __name__=='__main__':main()
