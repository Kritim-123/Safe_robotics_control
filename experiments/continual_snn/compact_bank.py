"""Lossless compaction of the legacy count bank, including neighbor multiplicity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spike_pattern_skills.associations import SKILLS,count_vector
from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder
from .models import BUNDLE


class CompactLegacyBank:
    def __init__(self,encoder,counts,label_bits,multiplicity,*,neighbors=3,max_distance=4.,
                 source_memory_checksum='',source_label_sources=None):
        self.encoder=encoder
        raw=np.asarray(counts);bits=np.asarray(label_bits);mult=np.asarray(multiplicity)
        if (raw.ndim!=2 or raw.shape[1]!=16 or not len(raw) or not np.isfinite(raw).all()
                or np.any(raw<0) or np.any(raw>16) or np.any(raw!=np.floor(raw))
                or bits.shape!=(len(raw),) or not np.isin(bits,range(1,8)).all()
                or mult.shape!=(len(raw),) or not np.isfinite(mult).all() or np.any(mult<1)
                or np.any(mult>np.iinfo(np.uint32).max) or np.any(mult!=np.floor(mult))
                or type(neighbors) is not int or neighbors<1 or not np.isfinite(max_distance) or max_distance<0
                or not isinstance(source_memory_checksum,str) or len(source_memory_checksum)!=64):
            raise ValueError('Invalid compact legacy bank')
        self.counts=raw.astype(np.int16);self.label_bits=bits.astype(np.uint8)
        self.multiplicity=mult.astype(np.uint32)
        self.exact={tuple(int(v) for v in row):i for i,row in enumerate(self.counts)}
        if len(self.exact)!=len(self.counts):raise ValueError('Duplicate compact signatures')
        self.floating=self.counts.astype(np.float64)
        for array in (self.counts,self.label_bits,self.multiplicity,self.floating):array.flags.writeable=False
        self.neighbors,self.max_distance=neighbors,float(max_distance)
        self.source_memory_checksum=source_memory_checksum
        self.source_label_sources=source_label_sources or {}
        self.name=f'compact_legacy16_{len(self.counts)}_patterns'
        self._update_checksum()

    @classmethod
    def from_legacy(cls,encoder,memory):
        if encoder.checksum!=memory.encoder_checksum:raise ValueError('Legacy encoder/bank mismatch')
        patterns={};sources=Counter()
        for row in memory.examples:
            signature=tuple(row['spike_counts']);label=SKILLS.index(row['skill'])
            bits,mult=patterns.get(signature,(0,0));patterns[signature]=(bits|(1<<label),mult+1)
            sources[row.get('label_source','historical_original_bank')]+=1
        return cls(encoder,list(patterns),[v[0] for v in patterns.values()],
            [v[1] for v in patterns.values()],neighbors=memory.neighbors,max_distance=memory.max_distance,
            source_memory_checksum=memory.checksum,source_label_sources=dict(sources))

    def _metadata(self):
        return dict(schema_version=1,kind='compact_legacy_count_bank',labels=list(SKILLS),
            encoder_checksum=self.encoder.checksum,neighbors=self.neighbors,max_distance=self.max_distance,
            source_memory_checksum=self.source_memory_checksum,source_label_sources=self.source_label_sources,
            training_rows=int(self.multiplicity.astype(np.uint64).sum()),
            compaction='unique counts and label union; preserve original row multiplicity for k-neighbor cutoff',
            encoder_weights_trained=False,evaluation_learning=False)

    def _update_checksum(self):
        digest=hashlib.sha256(json.dumps(self._metadata(),sort_keys=True).encode())
        for array,dtype in ((self.counts,'<i2'),(self.label_bits,'u1'),(self.multiplicity,'<u4')):
            digest.update(array.astype(dtype).tobytes())
        self.checksum=digest.hexdigest()

    def reset(self):pass

    def decode_counts(self,counts):
        query=count_vector(counts)
        if not query.any():return dict(proposed_skill=None,reason='silent pattern',matched_associations=0)
        exact=self.exact.get(tuple(query))
        if exact is not None:
            indices=np.array([exact]);reason='exact association'
        else:
            distances=np.linalg.norm(self.floating-query,axis=1)
            eligible=np.flatnonzero(distances<=self.max_distance)
            if not len(eligible):return dict(proposed_skill=None,reason='unknown pattern',matched_associations=0)
            ordered=eligible[np.argsort(distances[eligible],kind='stable')]
            cumulative=np.cumsum(self.multiplicity[ordered],dtype=np.uint64)
            boundary=min(int(np.searchsorted(cumulative,self.neighbors)),len(ordered)-1)
            cutoff=distances[ordered[boundary]]
            indices=ordered[distances[ordered]<=cutoff+1e-12];reason='neighbor agreement'
        bits=0
        for index in indices:bits|=int(self.label_bits[index])
        conflict=bits not in (1,2,4)
        return dict(proposed_skill=None if conflict else SKILLS[{1:0,2:1,4:2}[bits]],
            reason='conflicting associations' if conflict else reason,
            matched_associations=int(self.multiplicity[indices].astype(np.uint64).sum()))

    def predict(self,features,now=None):
        packet=self.encoder.encode(features)
        return dict(self.decode_counts(packet['spike_counts']),spike_counts=packet['spike_counts'],
            active_neuron_ids=packet['active_neuron_ids'],reset_state=packet['reset_state'])

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=np.asarray(json.dumps(self._metadata(),sort_keys=True)),
                counts=self.counts,label_bits=self.label_bits,multiplicity=self.multiplicity,
                encoder_weights=self.encoder.weights,encoder_biases=self.encoder.biases,
                checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['metadata'].item()))
            if meta['schema_version']!=1 or meta['kind']!='compact_legacy_count_bank' or meta['labels']!=list(SKILLS):
                raise ValueError('Unsupported compact bank artifact')
            encoder=DenseTerrainEncoder.load(BUNDLE/'dense_encoder.npz')
            if (encoder.checksum!=meta['encoder_checksum']
                    or not np.array_equal(bundle['encoder_weights'],encoder.weights)
                    or not np.array_equal(bundle['encoder_biases'],encoder.biases)):
                raise ValueError('Compact bank frozen encoder mismatch')
            result=cls(encoder,bundle['counts'],bundle['label_bits'],bundle['multiplicity'],
                neighbors=meta['neighbors'],max_distance=meta['max_distance'],
                source_memory_checksum=meta['source_memory_checksum'],source_label_sources=meta['source_label_sources'])
            if result._metadata()!=meta or result.checksum!=str(bundle['checksum'].item()):
                raise ValueError('Compact bank metadata/checksum mismatch')
        return result
