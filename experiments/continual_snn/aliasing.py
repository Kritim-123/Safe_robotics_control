"""Measure label ambiguity in active IDs versus complete spike-count patterns."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .encoding import PacketEncoder,PacketConfig
from .predictive import PredictivePacketEncoder
from .models import ROOT
from .readout import LABELS
from .study import save


def main():
    folder=ROOT/'output/continual_snn/predictive-comparison-v1'
    result=[]
    for split in ('training','validation'):
        rows=[json.loads(s) for s in (folder/(split+'_records.jsonl')).read_text().splitlines()]
        for name,encoder in (('sensory16',PacketEncoder()),('sensory64',PacketEncoder(PacketConfig(neurons=64))),
                             ('predictive80',PredictivePacketEncoder())):
            patterns={kind:defaultdict(list) for kind in ('active_ids','counts')}
            for index,row in enumerate(rows):
                features={n:row['features'][n] for n in encoder.feature_names}
                packet=encoder.encode(features)
                patterns['active_ids'][tuple(packet['active_neuron_ids'])].append(index)
                patterns['counts'][tuple(packet['spike_counts'])].append(index)
            for kind,mapping in patterns.items():
                conflicts=[];affected=set()
                for signature,indices in mapping.items():
                    labels={rows[i]['label'].removeprefix('dynamic/') for i in indices}
                    if len(labels)>1:
                        affected.update(indices)
                        conflicts.append(dict(signature=list(signature),labels=sorted(labels),rows=len(indices),
                            contexts=sorted({rows[i]['group'] for i in indices})))
                result.append(dict(split=split,model=name,representation=kind,unique_signatures=len(mapping),
                    rows=len(rows),conflicting_signatures=len(conflicts),affected_rows=len(affected),
                    examples=conflicts[:10],
                    caution='Exact observed label ambiguity only; no claim of unseen-pattern accuracy'))
                print(split,name,kind,len(mapping),'conflicts',len(conflicts),'affected',len(affected),flush=True)
    save(ROOT/'output/continual_snn/pattern-aliasing.json',result)


if __name__=='__main__':main()
