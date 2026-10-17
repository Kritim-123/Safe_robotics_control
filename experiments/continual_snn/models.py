"""Shared proposal interface. Neural variants are versioned separately."""
from pathlib import Path
from experiments.spike_pattern_skills.refine_associations import models

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'docs/agents/expanded_association_v2'


class FrozenBankModel:
    name='frozen_count_94'
    def __init__(self,bundle=BUNDLE):
        self.encoder,self.memory=models(Path(bundle))['dense']
        self.checksum=self.encoder.checksum+':'+self.memory.checksum

    def reset(self):
        pass

    def predict(self,features,now=None):
        packet=self.encoder.encode(features)
        result=self.memory.decode(packet['spike_counts'])
        return dict(result,spike_counts=packet['spike_counts'],
                    active_neuron_ids=packet['active_neuron_ids'],reset_state=packet['reset_state'])


def load_model(path):
    import json
    import numpy as np
    with np.load(path,allow_pickle=False) as bundle:
        kind=json.loads(str(bundle['metadata'].item()))['kind']
    if kind=='temporal_frozen_spike_ridge_readout':
        from .memory import TemporalReadout
        return TemporalReadout.load(path)
    if kind=='frozen_spike_prototype_memory':
        from .prototypes import PrototypeMemory
        return PrototypeMemory.load(path)
    from .readout import SpikeReadout
    return SpikeReadout.load(path)
