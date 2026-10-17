"""A learned association/readout on frozen spike representations, not motor RL."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .encoding import PacketConfig,PacketEncoder

LABELS=('wait','run','jump')


def labels_to_int(labels):
    return np.asarray([LABELS.index(v.removeprefix('dynamic/')) for v in labels],dtype=np.int64)


def grouped_weights(groups,labels,source_weights=None):
    groups=np.asarray(groups);labels=np.asarray(labels)
    unique,counts=np.unique(groups,return_counts=True)
    inverse=dict(zip(unique,counts))
    weights=np.asarray([1/inverse[g] for g in groups])
    if source_weights is not None:weights*=np.asarray(source_weights)
    totals=np.bincount(labels,weights=weights,minlength=3)
    weights*=np.asarray([1/max(totals[y],1e-12) for y in labels])
    return weights/np.mean(weights)


class SpikeReadout:
    def __init__(self,encoder=None,*,alpha=.01,confidence_threshold=0.,analog=False):
        if not np.isfinite([alpha,confidence_threshold]).all() or alpha<=0 or not 0<=confidence_threshold<=1:
            raise ValueError('Invalid readout hyperparameters')
        self.encoder=encoder or PacketEncoder()
        self.alpha=float(alpha);self.confidence_threshold=float(confidence_threshold);self.analog=bool(analog)
        c=self.encoder.config
        self.name=('analog_diagnostic' if analog else f'{c.neuron}_{c.neurons}_{c.representation}_a{alpha:g}')
        self.checksum=None

    def _embedding(self,features):
        if self.analog:
            return np.asarray([features[n] for n in self.encoder.feature_names])/getattr(self.encoder,'input_scales',self.encoder.base.scales)
        return self.encoder.encode(features)['embedding']

    def fit(self,features,labels,groups,*,source_weights=None):
        if len(features)!=len(labels) or len(features)!=len(groups) or len(set(groups))<3:
            raise ValueError('Mismatched or insufficient grouped training data')
        x=np.asarray([self._embedding(f) for f in features],dtype=np.float64)
        return self.fit_embeddings(x,labels,groups,source_weights=source_weights)

    def fit_embeddings(self,x,labels,groups,*,source_weights=None):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or len(x)!=len(labels) or len(labels)!=len(groups) or len(set(groups))<3:
            raise ValueError('Mismatched grouped embeddings')
        y=labels_to_int(labels);weights=grouped_weights(groups,y,source_weights)
        if not np.isfinite(x).all():raise ValueError('Invalid readout features')
        self.mean=np.average(x,axis=0,weights=weights)
        self.scale=np.maximum(np.sqrt(np.average((x-self.mean)**2,axis=0,weights=weights)),.05)
        z=(x-self.mean)/self.scale
        targets=np.eye(3)[y]
        self.bias=np.average(targets,axis=0,weights=weights)
        weighted=z*np.sqrt(weights[:,None]/weights.sum())
        rhs=weighted.T@((targets-self.bias)*np.sqrt(weights[:,None]/weights.sum()))
        covariance=weighted.T@weighted
        covariance.flat[::len(covariance)+1]+=self.alpha
        self.coefficients=np.linalg.solve(covariance,rhs)
        self.training_groups=sorted(set(groups));self.training_rows=len(x)
        self.training_labels={name:int(np.sum(y==i)) for i,name in enumerate(LABELS)}
        self._update_checksum()
        return self

    def _metadata(self):
        from .predictive import encoder_metadata
        return dict(schema_version=1,kind='analog_feature_diagnostic' if self.analog else 'frozen_spike_ridge_readout',
            packet_config=asdict(self.encoder.config),encoder_checksum=self.encoder.checksum,
            alpha=self.alpha,confidence_threshold=self.confidence_threshold,labels=list(LABELS),
            training_groups=self.training_groups,training_rows=self.training_rows,
            training_labels=self.training_labels,normalization='train_group_and_class_weighted',
            learned_encoder_weights=False,learned_readout=True,packet_reset=True,**encoder_metadata(self.encoder))

    def _update_checksum(self):
        digest=hashlib.sha256(json.dumps(self._metadata(),sort_keys=True).encode())
        for value in (self.mean,self.scale,self.bias,self.coefficients):digest.update(np.asarray(value,dtype='<f8').tobytes())
        self.checksum=digest.hexdigest()

    def reset(self):
        self.encoder.reset()

    def predict(self,features,now=None):
        if self.checksum is None:raise ValueError('Readout is not fitted')
        packet=(dict(spike_counts=None,active_neuron_ids=None,reset_state='analog_diagnostic') if self.analog
                else self.encoder.encode(features,now=now))
        embedding=(np.asarray([features[n] for n in self.encoder.feature_names])/getattr(self.encoder,'input_scales',self.encoder.base.scales)
                   if self.analog else packet['embedding'])
        scores=((embedding-self.mean)/self.scale)@self.coefficients+self.bias
        probabilities=np.exp(np.clip((scores-scores.max())*3.,-100,0))
        probabilities/=probabilities.sum()
        chosen=int(np.argmax(scores));confidence=float(probabilities[chosen])
        silent=not self.analog and not any(packet['spike_counts'])
        accepted=not silent and confidence>=self.confidence_threshold
        return dict(proposed_skill=LABELS[chosen] if accepted else None,
            reason='silent_pattern' if silent else 'learned_frozen_spike_readout' if accepted else 'readout_low_confidence',
            scores=scores.tolist(),confidence=confidence,spike_counts=packet['spike_counts'],
            active_neuron_ids=packet['active_neuron_ids'],reset_state=packet['reset_state'])

    def predict_embeddings(self,embeddings):
        embeddings=np.asarray(embeddings,dtype=np.float64)
        scores=((embeddings-self.mean)/self.scale)@self.coefficients+self.bias
        if not np.isfinite(scores).all():raise ValueError('Nonfinite readout scores')
        return scores

    def save(self,path):
        if self.checksum is None:raise ValueError('Readout is not fitted')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=np.asarray(json.dumps(self._metadata(),sort_keys=True)),
                mean=self.mean,scale=self.scale,bias=self.bias,coefficients=self.coefficients,
                encoder_weights=self.encoder.weights,encoder_biases=self.encoder.biases,
                checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['metadata'].item()))
            if (meta['schema_version']!=1 or meta['labels']!=list(LABELS) or meta['learned_encoder_weights'] is not False
                    or meta['kind'] not in ('analog_feature_diagnostic','frozen_spike_ridge_readout')):
                raise ValueError('Unsupported readout artifact')
            from .predictive import restore_encoder
            encoder=restore_encoder(meta,bundle)
            if encoder.checksum!=meta['encoder_checksum']:raise ValueError('Encoder differs from readout provenance')
            result=cls(encoder,alpha=meta['alpha'],confidence_threshold=meta['confidence_threshold'],
                       analog=meta['kind']=='analog_feature_diagnostic')
            for key in ('mean','scale','bias','coefficients'):
                setattr(result,key,bundle[key].copy())
                if not np.isfinite(getattr(result,key)).all():raise ValueError('Nonfinite readout artifact')
            c=encoder.config
            dimensions=(len(encoder.feature_names) if result.analog else c.neurons if c.representation=='counts' else
                        c.neurons*(c.steps+int(c.representation=='filtered')))
            if (result.mean.shape!=(dimensions,) or result.scale.shape!=(dimensions,)
                    or result.bias.shape!=(3,) or result.coefficients.shape!=(dimensions,3)
                    or np.any(result.scale<=0)):
                raise ValueError('Invalid readout array shapes or scales')
            result.training_groups=meta['training_groups'];result.training_rows=meta['training_rows']
            result.training_labels=meta['training_labels']
            if result._metadata()!=meta:raise ValueError('Readout metadata differs from supported contract')
            result._update_checksum()
            if result.checksum!=str(bundle['checksum'].item()):raise ValueError('Readout checksum mismatch')
        return result
