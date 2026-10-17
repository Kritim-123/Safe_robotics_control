"""Refit readouts on ordered causal sequences for matched memory ablations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .encoding import PacketConfig
from .memory import MemoryConfig,TemporalPacketEncoder,TemporalReadout
from .fit import metrics
from .readout import labels_to_int
from .study import save


def fit(output,data_folder):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=Path(data_folder)
    train=[json.loads(s) for s in (source/'training_records.jsonl').read_text().splitlines()]
    val=[json.loads(s) for s in (source/'validation_records.jsonl').read_text().splitlines()]
    configurations=[]
    for neuron in ('lif','alif'):
        for retain,reset in ((False,True),(True,True),(True,False)):
            configurations.append((PacketConfig(neurons=64,neuron=neuron,adaptation_decay=.99),
                MemoryConfig(retain=retain,reset_on_command_change=reset)))
    save(output/'declared_configurations.json',[dict(packet=asdict(p),memory=asdict(m)) for p,m in configurations])
    ranking=[]
    with (output/'candidates.jsonl').open('x') as stream:
        for i,(packet,memory) in enumerate(configurations):
            encoder=TemporalPacketEncoder(packet,memory)
            xt=encoder.embed_records(train);xv=encoder.embed_records(val)
            winner=None
            for alpha in (.0001,.001,.01,.1,1.):
                model=TemporalReadout(encoder,alpha=alpha)
                model.fit_embeddings(xt,[r['label'] for r in train],[r['group'] for r in train])
                pred=np.argmax(model.predict_embeddings(xv),axis=1)
                score=metrics([r['label'] for r in val],pred,[r['group'] for r in val])
                row=dict(index=i,packet=asdict(packet),memory=asdict(memory),alpha=alpha,metrics=score,
                         model_checksum=model.checksum)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if winner is None or score['balanced_accuracy']>winner[0]['metrics']['balanced_accuracy']:
                    winner=(row,model)
                print(i,model.name,alpha,round(score['balanced_accuracy'],5),flush=True)
            row,model=winner;model.save(output/'models'/f'candidate-{i:02d}.npz')
            ranking.append(row);save(output/'ranking.json',sorted(ranking,key=lambda r:r['metrics']['balanced_accuracy'],reverse=True))
    return ranking


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args();fit(args.output,args.data)
