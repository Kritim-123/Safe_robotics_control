"""Compare larger explicit pattern banks against learned readouts on the same data."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .encoding import PacketConfig,PacketEncoder
from .prototypes import PrototypeMemory
from .fit import metrics
from .study import save


def fit(output,data):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);data=Path(data)
    train=[json.loads(s) for s in (data/'training_records.jsonl').read_text().splitlines()]
    val=[json.loads(s) for s in (data/'validation_records.jsonl').read_text().splitlines()]
    ranking=[]
    with (output/'candidates.jsonl').open('x') as stream:
        for units in (16,64):
            encoder=PacketEncoder(PacketConfig(neurons=units))
            model=PrototypeMemory(encoder).fit(train)
            queries=[encoder.encode(r['features'])['spike_counts'] for r in val]
            for neighbors in (1,3,5):
                for radius in (2.,4.,8.,12.):
                    model.neighbors=neighbors;model.max_distance=radius;model._update_checksum()
                    proposals=[model.decode_counts(q) for q in queries]
                    pred=[('wait','run','jump').index(p['proposed_skill']) if p['proposed_skill'] else 0 for p in proposals]
                    metric=metrics([r['label'] for r in val],pred,[r['group'] for r in val])
                    row=dict(units=units,neighbors=neighbors,radius=radius,metrics=metric,
                        prototype_count=len(model.counts),conflicting_patterns=int(np.sum(~np.isin(model.label_bits,[1,2,4]))),
                        proposal_coverage=float(np.mean([p['proposed_skill'] is not None for p in proposals])),
                        reasons=dict(Counter(p['reason'] for p in proposals)))
                    checkpoint=f'n{units}-k{neighbors}-r{radius:g}.npz';model.save(output/'models'/checkpoint)
                    row['checkpoint']='models/'+checkpoint;ranking.append(row)
                    stream.write(json.dumps(row)+'\n');stream.flush()
                    save(output/'ranking.json',sorted(ranking,key=lambda r:r['metrics']['balanced_accuracy'],reverse=True))
                    print(units,neighbors,radius,round(metric['balanced_accuracy'],4),row['proposal_coverage'],flush=True)
    return ranking


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args();fit(args.output,args.data)
