"""Versioned 64 sensory + 16 predictive LIF population; no encoder learning."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from .encoding import PacketConfig,PacketEncoder,spike_embedding

PREDICTIVE_FEATURES=('predicted_run_clearance','predicted_jump_clearance')


class PredictivePacketEncoder(PacketEncoder):
    def __init__(self,config=None,*,weights=None,biases=None):
        self.config=config or PacketConfig(neurons=80)
        if self.config!=PacketConfig(neurons=80):
            raise ValueError('Predictive v1 requires its declared fixed LIF configuration')
        if (weights is None)!=(biases is None):raise ValueError('Require both stored arrays')
        prefix=PacketEncoder(PacketConfig(neurons=64),
            weights=None if weights is None else np.asarray(weights)[:64,:22],
            biases=None if biases is None else np.asarray(biases)[:64])
        self.base=prefix.base
        self.feature_names=tuple(prefix.feature_names)+PREDICTIVE_FEATURES
        self.input_scales=np.r_[self.base.scales,1.,1.]
        if weights is None:
            self.weights=np.zeros((80,24));self.weights[:64,:22]=prefix.weights
            rng=np.random.default_rng(20261101)
            self.weights[64:,22:]=rng.normal(size=(16,2))/np.sqrt(2)
            self.biases=np.r_[prefix.biases,rng.normal(0,.1,16)]
        else:
            self.weights=np.asarray(weights,dtype=np.float64).copy()
            self.biases=np.asarray(biases,dtype=np.float64).copy()
        if (self.weights.shape!=(80,24) or self.biases.shape!=(80,)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.biases).all()
                or np.any(self.weights[:64,22:]) or np.any(self.weights[64:,:22])):
            raise ValueError('Invalid predictive population block structure')
        self.weights.flags.writeable=self.biases.flags.writeable=False
        self.contract=dict(kind='predictive_population_v1',feature_names=list(self.feature_names),
            input_scales=self.input_scales.tolist(),predictive_seed=20261101,
            sensory_encoder_checksum=prefix.checksum,packet_config=asdict(self.config))
        digest=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode())
        digest.update(self.weights.astype('<f8').tobytes());digest.update(self.biases.astype('<f8').tobytes())
        self.checksum=digest.hexdigest()

    def encode(self,features,now=None):
        raw=np.asarray([features[n] for n in self.feature_names],dtype=np.float64)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite predictive inputs')
        x=np.clip(raw/self.input_scales,-3,3)
        current=self.weights@x+self.biases
        voltage=np.zeros(80);raster=[]
        for _ in range(16):
            voltage=self.base.metadata['beta_per_step']*voltage+current
            spikes=(voltage>=self.base.metadata['threshold']).astype(np.uint8)
            voltage-=self.base.metadata['threshold']*spikes;raster.append(spikes)
        raster=np.asarray(raster)
        return dict(spike_counts=raster.sum(axis=0).tolist(),raster=raster,
            embedding=spike_embedding(raster),normalized_inputs=x,
            active_neuron_ids=np.flatnonzero(raster.sum(axis=0)).tolist(),
            reset_state='zero_membranes_each_packet',encoder_checksum=self.checksum)


def encoder_metadata(encoder):
    return dict(input_contract=encoder.contract) if isinstance(encoder,PredictivePacketEncoder) else {}


def restore_encoder(meta,bundle):
    cfg=PacketConfig(**meta['packet_config'])
    args=dict(weights=bundle['encoder_weights'] if 'encoder_weights' in bundle else None,
              biases=bundle['encoder_biases'] if 'encoder_biases' in bundle else None)
    if 'input_contract' not in meta:return PacketEncoder(cfg,**args)
    encoder=PredictivePacketEncoder(cfg,**args)
    if encoder.contract!=meta['input_contract']:raise ValueError('Unsupported predictive input contract')
    return encoder
