"""Separately versioned retained-state ablation, with readouts refitted on sequences."""
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .encoding import PacketConfig,PacketEncoder,spike_embedding
from .readout import SpikeReadout,LABELS


@dataclass(frozen=True)
class MemoryConfig:
    retain:bool=True
    membrane_tau:float=.5
    adaptation_tau:float=1.
    maximum_gap:float=.35
    reset_on_command_change:bool=True

    def validate(self):
        if (type(self.retain) is not bool or type(self.reset_on_command_change) is not bool
                or not np.isfinite([self.membrane_tau,self.adaptation_tau,self.maximum_gap]).all()
                or min(self.membrane_tau,self.adaptation_tau,self.maximum_gap)<=0):
            raise ValueError('Invalid temporal-state configuration')


class TemporalPacketEncoder(PacketEncoder):
    def __init__(self,config=None,memory=None,baseline=None,*,weights=None,biases=None):
        super().__init__(config,baseline,weights=weights,biases=biases)
        self.memory=memory or MemoryConfig();self.memory.validate()
        self.packet_encoder_checksum=self.checksum
        self.checksum=hashlib.sha256((self.checksum+json.dumps(asdict(self.memory),sort_keys=True)).encode()).hexdigest()
        self.reset()

    def reset(self):
        self.membrane=np.zeros(self.config.neurons)
        self.adaptation=np.zeros(self.config.neurons)
        self.synapse=np.zeros(self.config.neurons)
        self.last_time=None;self.command=None;self.cached=None;self.reset_reason='episode'

    def encode(self,features,now=None):
        if now is None:
            raise ValueError('Temporal encoder requires a physical observation timestamp')
        if not math.isfinite(now) or now<0:
            self.reset();raise ValueError('Invalid temporal observation time')
        if self.last_time is not None and now<self.last_time:
            self.reset();raise ValueError('Temporal observation time moved backwards')
        if self.last_time is not None and now==self.last_time:
            return self.cached
        raw=self.base._feature_vector(features)
        command=(bool(features['jump_requested']),abs(features['goal_forward'])+abs(features['goal_left'])<1e-6)
        gap=now-self.last_time if self.last_time is not None else 0.
        reason=self.reset_reason
        if self.last_time is not None and gap>self.memory.maximum_gap:
            self.reset();reason='stale_gap'
        if self.memory.reset_on_command_change and self.command is not None and command!=self.command:
            self.reset();reason='command_changed'
        if not self.memory.retain:
            self.membrane[:]=self.adaptation[:]=self.synapse[:]=0.
            reason='packet_reset'
        else:
            self.membrane*=np.exp(-gap/self.memory.membrane_tau)
            self.adaptation*=np.exp(-gap/self.memory.adaptation_tau)
            self.synapse*=np.exp(-gap/self.memory.membrane_tau)
        x=np.clip(raw/self.base.scales,-self.base.metadata['input_clip'],self.base.metadata['input_clip'])
        current=self.weights@x+self.biases
        c=self.config;v=self.membrane;adapt=self.adaptation;syn=self.synapse
        raster=[]
        for _ in range(c.steps):
            if c.neuron=='synaptic':
                syn=c.synaptic_decay*syn+(1-c.synaptic_decay)*current;drive=syn
            else:drive=current
            threshold=self.base.metadata['threshold']+(c.adaptation_strength*adapt if c.neuron=='alif' else 0.)
            v=self.base.metadata['beta_per_step']*v+drive
            fired=(v>=threshold).astype(np.uint8);v-=threshold*fired
            adapt=c.adaptation_decay*adapt+fired;raster.append(fired)
        self.membrane,self.adaptation,self.synapse=v.copy(),adapt.copy(),syn.copy()
        if not np.isfinite(np.r_[v,adapt,syn]).all():
            self.reset();raise ValueError('Nonfinite temporal neuron state')
        raster=np.asarray(raster,dtype=np.uint8)
        self.last_time=float(now);self.command=command;self.reset_reason=None
        self.cached=dict(spike_counts=raster.sum(axis=0).tolist(),raster=raster,
            embedding=spike_embedding(raster,c.representation,c.filter_tau),normalized_inputs=x,
            active_neuron_ids=np.flatnonzero(raster.sum(axis=0)).tolist(),encoder_checksum=self.checksum,
            reset_state='membranes_retained_counts_cleared' if self.memory.retain else 'zero_membranes_each_packet',
            memory_reset_reason=reason)
        return self.cached

    def embed_records(self,records):
        values=[];previous=None
        self.reset()
        for row in records:
            if row['group']!=previous:self.reset()
            values.append(self.encode(row['features'],now=row.get('time',0.))['embedding'])
            previous=row['group']
        self.reset()
        return np.asarray(values)


class TemporalReadout(SpikeReadout):
    def __init__(self,encoder=None,**kwargs):
        super().__init__(encoder or TemporalPacketEncoder(),**kwargs)
        self.name='temporal_'+self.name+('_command_reset' if self.encoder.memory.reset_on_command_change else '_carry')

    def _metadata(self):
        value=super()._metadata()
        value.update(kind='temporal_frozen_spike_ridge_readout',memory_config=asdict(self.encoder.memory),
                     packet_reset=not self.encoder.memory.retain)
        return value

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['metadata'].item()))
            if meta['kind']!='temporal_frozen_spike_ridge_readout' or meta['labels']!=list(LABELS):
                raise ValueError('Invalid temporal readout artifact')
            encoder=TemporalPacketEncoder(PacketConfig(**meta['packet_config']),MemoryConfig(**meta['memory_config']),
                weights=bundle['encoder_weights'] if 'encoder_weights' in bundle else None,
                biases=bundle['encoder_biases'] if 'encoder_biases' in bundle else None)
            if encoder.checksum!=meta['encoder_checksum']:raise ValueError('Temporal encoder checksum mismatch')
            result=cls(encoder,alpha=meta['alpha'],confidence_threshold=meta['confidence_threshold'])
            for key in ('mean','scale','bias','coefficients'):
                value=bundle[key].copy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite temporal readout')
                setattr(result,key,value)
            dimensions=encoder.config.neurons
            if encoder.config.representation!='counts':
                dimensions*=encoder.config.steps+int(encoder.config.representation=='filtered')
            if (result.mean.shape!=(dimensions,) or result.scale.shape!=(dimensions,)
                    or result.bias.shape!=(3,) or result.coefficients.shape!=(dimensions,3) or np.any(result.scale<=0)):
                raise ValueError('Invalid temporal readout dimensions')
            result.training_groups=meta['training_groups'];result.training_rows=meta['training_rows']
            result.training_labels=meta['training_labels']
            if result._metadata()!=meta:raise ValueError('Temporal metadata differs from supported contract')
            result._update_checksum()
            if result.checksum!=str(bundle['checksum'].item()):raise ValueError('Temporal readout checksum mismatch')
        return result
