"""Expanded count-pattern association memory, retaining explicit conflict refusal."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .encoding import PacketConfig,PacketEncoder
from .readout import LABELS,labels_to_int


class PrototypeMemory:
    def __init__(self,encoder=None,*,neighbors=3,max_distance=8.):
        self.encoder=encoder or PacketEncoder(PacketConfig(neurons=64))
        if (self.encoder.config.representation!='counts' or type(neighbors) is not int or neighbors<1
                or not np.isfinite(max_distance) or max_distance<0):
            raise ValueError('Invalid prototype-memory contract')
        self.neighbors,self.max_distance=neighbors,float(max_distance)
        self.name=f'count_memory_{self.encoder.config.neurons}_k{neighbors}_r{max_distance:g}'
        self.checksum=None

    def reset(self):self.encoder.reset()

    def fit(self,records):
        if not records:raise ValueError('Require labeled training records')
        patterns={}
        for row in records:
            counts=tuple(self.encoder.encode(row['features'])['spike_counts'])
            label=LABELS.index(row['label'].removeprefix('dynamic/'))
            patterns[counts]=patterns.get(counts,0)|(1<<label)
        self.counts=np.asarray(list(patterns),dtype=np.int16)
        self.label_bits=np.asarray(list(patterns.values()),dtype=np.uint8)
        self.training_groups=sorted({r['group'] for r in records});self.training_rows=len(records)
        self._prepare();self._update_checksum();return self

    def _prepare(self):
        self.exact={tuple(int(v) for v in row):i for i,row in enumerate(self.counts)}
        self.floating=self.counts.astype(np.float64)

    def _metadata(self):
        from .predictive import encoder_metadata
        return dict(schema_version=1,kind='frozen_spike_prototype_memory',labels=list(LABELS),
            packet_config=asdict(self.encoder.config),encoder_checksum=self.encoder.checksum,
            neighbors=self.neighbors,max_distance=self.max_distance,
            training_groups=self.training_groups,training_rows=self.training_rows,
            learned_encoder_weights=False,duplicate_conflicts='preserve_labels_and_abstain',**encoder_metadata(self.encoder))

    def _update_checksum(self):
        digest=hashlib.sha256(json.dumps(self._metadata(),sort_keys=True).encode())
        digest.update(self.counts.astype('<i2').tobytes());digest.update(self.label_bits.tobytes())
        self.checksum=digest.hexdigest()

    def decode_counts(self,counts):
        raw=np.asarray(counts)
        if (raw.shape!=(self.encoder.config.neurons,) or not np.isfinite(raw).all()
                or np.any(raw<0) or np.any(raw>self.encoder.config.steps) or np.any(raw!=np.floor(raw))):
            raise ValueError('Invalid count vector')
        query=raw.astype(np.int16)
        if not query.any():return dict(proposed_skill=None,reason='silent_pattern',distance=None)
        exact=self.exact.get(tuple(int(v) for v in query))
        if exact is not None:
            bits=int(self.label_bits[exact]);distance=0.;selected=[exact]
        else:
            distances=np.linalg.norm(self.floating-query,axis=1)
            near=np.flatnonzero(distances<=self.max_distance)
            if not len(near):return dict(proposed_skill=None,reason='unknown_pattern',distance=float(min(distances)))
            ordered=near[np.argsort(distances[near],kind='stable')]
            cutoff=distances[ordered[min(self.neighbors,len(ordered))-1]]
            selected=ordered[distances[ordered]<=cutoff+1e-12]
            bits=0
            for index in selected:bits|=int(self.label_bits[index])
            distance=float(distances[selected[0]])
        conflict=bits not in (1,2,4)
        return dict(proposed_skill=None if conflict else LABELS[{1:0,2:1,4:2}[bits]],
            reason='conflicting_associations' if conflict else 'exact_association' if distance==0 else 'neighbor_agreement',
            distance=distance,matched_patterns=len(selected))

    def predict(self,features,now=None):
        packet=self.encoder.encode(features)
        return dict(self.decode_counts(packet['spike_counts']),spike_counts=packet['spike_counts'],
            active_neuron_ids=packet['active_neuron_ids'],reset_state=packet['reset_state'])

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=np.asarray(json.dumps(self._metadata(),sort_keys=True)),
                counts=self.counts,label_bits=self.label_bits,
                encoder_weights=self.encoder.weights,encoder_biases=self.encoder.biases,
                checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['metadata'].item()))
            if (meta['kind']!='frozen_spike_prototype_memory' or meta['labels']!=list(LABELS)
                    or meta['schema_version']!=1 or meta['learned_encoder_weights'] is not False):
                raise ValueError('Unsupported pattern memory artifact')
            from .predictive import restore_encoder
            encoder=restore_encoder(meta,bundle)
            if encoder.checksum!=meta['encoder_checksum']:raise ValueError('Pattern encoder checksum mismatch')
            result=cls(encoder,neighbors=meta['neighbors'],max_distance=meta['max_distance'])
            result.counts=bundle['counts'].copy();result.label_bits=bundle['label_bits'].copy()
            if (result.counts.ndim!=2 or result.counts.shape[1]!=encoder.config.neurons
                    or result.label_bits.shape!=(len(result.counts),) or not len(result.counts)
                    or not np.isfinite(result.counts).all() or np.any(result.counts!=np.floor(result.counts)) or np.any(result.counts<0)
                    or np.any(result.counts>encoder.config.steps) or not np.isin(result.label_bits,range(1,8)).all()):
                raise ValueError('Invalid prototype arrays')
            result.training_groups=meta['training_groups'];result.training_rows=meta['training_rows']
            if result._metadata()!=meta:raise ValueError('Prototype metadata differs from supported contract')
            result._prepare();result._update_checksum()
            if result.checksum!=str(bundle['checksum'].item()):raise ValueError('Pattern memory checksum mismatch')
        return result
