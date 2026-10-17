"""Fixed seeded packet encoders; original rows/weights are never modified."""
from dataclasses import asdict,dataclass
import hashlib
import json

import numpy as np

from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
from .models import BUNDLE


@dataclass(frozen=True)
class PacketConfig:
    neurons:int=16
    steps:int=16
    neuron:str='lif'
    representation:str='counts'
    seed:int=20261031
    adaptation_decay:float=.9
    adaptation_strength:float=.1
    synaptic_decay:float=.5
    filter_tau:float=2.

    def validate(self):
        if (type(self.neurons) is not int or not 16<=self.neurons<=512
                or type(self.steps) is not int or not 4<=self.steps<=64
                or self.neuron not in ('lif','alif','synaptic')
                or self.representation not in ('counts','raster','filtered')
                or type(self.seed) is not int or self.seed<0
                or not np.isfinite([self.adaptation_decay,self.adaptation_strength,self.synaptic_decay,self.filter_tau]).all()
                or not 0<=self.adaptation_decay<1 or self.adaptation_strength<0
                or not 0<=self.synaptic_decay<1 or self.filter_tau<=0):
            raise ValueError('Invalid packet encoder configuration')


def spike_embedding(raster,representation='counts',tau=2.):
    s=np.asarray(raster,dtype=np.float64)
    if s.ndim!=2 or not np.isfinite(s).all() or not np.isin(s,[0.,1.]).all():
        raise ValueError('Expected binary spike raster (steps, neurons)')
    if representation=='counts':
        return s.mean(axis=0)
    if representation=='raster':
        return s.ravel()
    if representation!='filtered' or not np.isfinite(tau) or tau<=0:
        raise ValueError('Invalid spike representation/filter')
    decay=float(np.exp(-1/tau));state=np.zeros(s.shape[1]);filtered=[]
    for spikes in s:
        state=decay*state+spikes;filtered.append(state.copy())
    # Discrete exponential-filter metric, including the complete post-window
    # tail. Its internal steps do NOT have an asserted biological duration.
    tail=state*decay/np.sqrt(1-decay**2)
    return np.r_[np.asarray(filtered).ravel(),tail]/np.sqrt(tau)


class PacketEncoder:
    def __init__(self,config=None,baseline=None,*,weights=None,biases=None):
        self.config=config or PacketConfig();self.config.validate()
        self.base=baseline or DenseTerrainEncoder.load(BUNDLE/'dense_encoder.npz')
        extra=self.config.neurons-16
        rng=np.random.default_rng(self.config.seed)
        scales=np.r_[np.full(14,1/np.sqrt(14)),np.full(8,1/np.sqrt(8))]
        added=rng.normal(size=(extra,22))*scales if extra else np.empty((0,22))
        bias=rng.normal(0.,.1,extra)
        if weights is None and biases is None:
            self.weights=np.vstack([self.base.weights,added]);self.biases=np.r_[self.base.biases,bias]
        elif weights is not None and biases is not None:
            self.weights=np.asarray(weights,dtype=np.float64).copy()
            self.biases=np.asarray(biases,dtype=np.float64).copy()
            if (self.weights.shape!=(self.config.neurons,22) or self.biases.shape!=(self.config.neurons,)
                    or not np.isfinite(self.weights).all() or not np.isfinite(self.biases).all()
                    or not np.array_equal(self.weights[:16],self.base.weights)
                    or not np.array_equal(self.biases[:16],self.base.biases)):
                raise ValueError('Stored encoder must preserve the original frozen rows and biases')
        else:
            raise ValueError('Supply both encoder weights and biases')
        self.weights.flags.writeable=self.biases.flags.writeable=False
        digest=hashlib.sha256(json.dumps(asdict(self.config),sort_keys=True).encode())
        digest.update(self.base.checksum.encode());digest.update(self.weights.astype('<f8').tobytes())
        digest.update(self.biases.astype('<f8').tobytes());self.checksum=digest.hexdigest()
        self.feature_names=self.base.feature_names

    def reset(self):
        pass  # Every call is an independent fresh packet by contract.

    def encode(self,features,now=None):
        raw=self.base._feature_vector(features)
        x=np.clip(raw/self.base.scales,-self.base.metadata['input_clip'],self.base.metadata['input_clip'])
        current=self.weights@x+self.biases
        c=self.config;v=np.zeros(c.neurons);adapt=np.zeros(c.neurons);syn=np.zeros(c.neurons)
        raster=[]
        for _ in range(c.steps):
            if c.neuron=='synaptic':
                syn=c.synaptic_decay*syn+(1-c.synaptic_decay)*current
                drive=syn
            else:drive=current
            baseline_threshold=self.base.metadata['threshold']
            threshold=baseline_threshold+c.adaptation_strength*adapt if c.neuron=='alif' else baseline_threshold
            v=self.base.metadata['beta_per_step']*v+drive
            spikes=(v>=threshold).astype(np.uint8)
            v-=threshold*spikes
            adapt=c.adaptation_decay*adapt+spikes
            raster.append(spikes)
        raster=np.asarray(raster,dtype=np.uint8)
        if not np.isfinite(v).all():raise ValueError('Nonfinite spiking state')
        return dict(spike_counts=raster.sum(axis=0).tolist(),raster=raster,
            embedding=spike_embedding(raster,c.representation,c.filter_tau),
            normalized_inputs=x,active_neuron_ids=np.flatnonzero(raster.sum(axis=0)).tolist(),
            reset_state='zero_membranes_each_packet',encoder_checksum=self.checksum)

    def embed_many(self,features):
        return np.asarray([self.encode(f)['embedding'] for f in features])
