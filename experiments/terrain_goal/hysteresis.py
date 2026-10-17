"""V2: persistent goal-hold commitment driven by fresh spike-count packets."""
import hashlib,io,json
from pathlib import Path
import numpy as np
from .factored import FactoredGoalSelector


class HystereticGoalSelector:
    def __init__(self,base,base_bytes,*,release_threshold=25,release_confirmations=2,entry_confirmations=2,schema_version=2):
        if not isinstance(base,FactoredGoalSelector):raise ValueError('Require the frozen V1 goal selector')
        if type(release_threshold) is not int or not 0<=release_threshold<base.threshold:
            raise ValueError('Release must require fewer proximity spikes than hold entry')
        if type(release_confirmations) is not int or not 1<=release_confirmations<=10:raise ValueError('Invalid release confirmation')
        if type(entry_confirmations) is not int or not 1<=entry_confirmations<=10 or schema_version not in (1,2):raise ValueError('Invalid entry confirmation')
        self.base=base;self.base_bytes=bytes(base_bytes);self.encoder=base.encoder
        self.release_threshold=release_threshold;self.release_confirmations=release_confirmations
        self.entry_confirmations=entry_confirmations;self.schema_version=schema_version
        self.name='hysteretic_goal_count_selector_v2'
        self.checksum=hashlib.sha256(json.dumps(self.metadata(),sort_keys=True).encode()).hexdigest()
        self.reset()

    def metadata(self):
        result=dict(kind='goal_count_hysteresis_v2',schema_version=self.schema_version,base_checksum=self.base.checksum,
            base_sha256=hashlib.sha256(self.base_bytes).hexdigest(),release_count_threshold=self.release_threshold,
            release_confirmations=self.release_confirmations,hold_count_threshold=self.base.threshold,
            state='Goal hold commitment only; zero neuron membranes and counts each packet',
            release_calibration='25 spikes is the frozen proximity code at the existing 0.16 m resume boundary',
            learned_encoder_or_motor_changes=False)
        if self.schema_version==2:result['entry_confirmations']=self.entry_confirmations
        return result

    @classmethod
    def from_path(cls,path,**kwargs):
        payload=Path(path).read_bytes();return cls(FactoredGoalSelector.load(io.BytesIO(payload)),payload,**kwargs)

    def reset(self):
        self.base.reset();self.held=False;self.release_count=0;self.entry_count=0;self.last_time=None;self.last_proposal=None

    def augment_features(self,*args,**kwargs):return self.base.augment_features(*args,**kwargs)

    def predict(self,features,now=None):
        if now is not None:
            if not np.isfinite(now) or (self.last_time is not None and now<self.last_time):raise ValueError('Invalid neural packet clock')
            if self.last_time is not None and now==self.last_time:return dict(self.last_proposal)
        r=self.base.predict(features,now)
        eligible=bool(features.get('goal_mode') and features.get('task_progress'))
        count=r['goal_count_sum']
        if not eligible:
            self.held=False;self.release_count=0;self.entry_count=0
        else:
            if self.held:
                self.release_count=self.release_count+1 if count<self.release_threshold else 0
                if self.release_count>=self.release_confirmations:
                    self.held=False;self.release_count=0
            if not self.held:
                self.entry_count=self.entry_count+1 if count>=self.base.threshold and r['base_proposed_skill']=='run' else 0
                if self.entry_count>=self.entry_confirmations:
                    self.held=True;self.entry_count=0
            if self.held and r['base_proposed_skill']=='run':
                r['proposed_skill']='wait';r['reason']='learned_goal_count_hold_association'
        r.update(goal_latch_active=self.held,release_confirmation_count=self.release_count,
            entry_confirmation_count=self.entry_count,
            goal_release_count_threshold=self.release_threshold)
        self.last_time=now;self.last_proposal=dict(r)
        return r

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=np.asarray(json.dumps(self.metadata(),sort_keys=True)),
                base_model=np.frombuffer(self.base_bytes,dtype=np.uint8),checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as b:
            meta=json.loads(str(b['metadata'].item()))
            if meta['kind']!='goal_count_hysteresis_v2' or meta['schema_version'] not in (1,2):raise ValueError('Unsupported goal latch')
            payload=b['base_model'].tobytes()
            result=cls(FactoredGoalSelector.load(io.BytesIO(payload)),payload,
                release_threshold=meta['release_count_threshold'],release_confirmations=meta['release_confirmations'],
                entry_confirmations=meta.get('entry_confirmations',1),schema_version=meta['schema_version'])
            if result.metadata()!=meta or result.checksum!=str(b['checksum'].item()):raise ValueError('Goal latch changed')
            return result
