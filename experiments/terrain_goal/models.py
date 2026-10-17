"""Portable frozen ridge/prototype models with explicit task-population schema."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.continual_snn.readout import SpikeReadout
from experiments.continual_snn.prototypes import PrototypeMemory
from .encoder import TaskEncoder,add_task


class Selector:
    def __init__(self,encoder,decoder='ridge',alpha=.001,max_distance=12.):
        if decoder not in ('ridge','prototype'):raise ValueError('Unknown decoder')
        self.encoder,self.decoder=encoder,decoder
        self.inner=SpikeReadout(encoder,alpha=alpha) if decoder=='ridge' else PrototypeMemory(encoder,neighbors=1,max_distance=max_distance)
        self.name=f'terrain_goal_{getattr(encoder,"kind","sensory80")}_{decoder}'

    def augment_features(self,features,*,task,geometry,observed):return add_task(features,task)

    def fit(self,records):
        if self.decoder=='ridge':self.inner.fit([r['features'] for r in records],[r['label'] for r in records],[r['group'] for r in records])
        else:self.inner.fit(records)
        self._checksum();return self

    def _checksum(self):
        self.checksum=hashlib.sha256((self.encoder.checksum+self.inner.checksum+self.name).encode()).hexdigest()

    def reset(self):self.inner.reset()
    def predict(self,features,now=None):return self.inner.predict(features,now)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        meta=dict(schema_version=1,kind='terrain_goal_selector',encoder_kind=self.encoder.kind,
            encoder_checksum=self.encoder.checksum,decoder=self.decoder,inner=self.inner._metadata(),checksum=self.checksum,
            contract=self.encoder.contract)
        arrays=dict(encoder_weights=self.encoder.weights,encoder_biases=self.encoder.biases,
            metadata=np.asarray(json.dumps(meta,sort_keys=True)))
        fields=('mean','scale','bias','coefficients') if self.decoder=='ridge' else ('counts','label_bits')
        arrays.update({n:getattr(self.inner,n) for n in fields})
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['metadata'].item()))
            if meta['schema_version']!=1 or meta['kind']!='terrain_goal_selector':raise ValueError('Unsupported selector')
            encoder=TaskEncoder(meta['encoder_kind'],bundle['encoder_weights'],bundle['encoder_biases'])
            if encoder.checksum!=meta['encoder_checksum'] or encoder.contract!=meta['contract']:raise ValueError('Encoder changed')
            m=meta['inner'];result=cls(encoder,meta['decoder'],alpha=m.get('alpha',.001),max_distance=m.get('max_distance',12.))
            fields=('mean','scale','bias','coefficients') if result.decoder=='ridge' else ('counts','label_bits')
            for name in fields:
                value=bundle[name].copy()
                if not np.isfinite(value).all():raise ValueError('Nonfinite saved model')
                setattr(result.inner,name,value)
            result.inner.training_groups=m['training_groups'];result.inner.training_rows=m['training_rows']
            if result.decoder=='ridge':result.inner.training_labels=m['training_labels']
            else:result.inner._prepare()
            result.inner._update_checksum();result._checksum()
            if result.checksum!=meta['checksum']:raise ValueError('Model checksum mismatch')
        return result
