"""Monotonic learned count-to-hold association above a frozen task decoder."""
import hashlib,json
from pathlib import Path
import numpy as np
from .encoder import TaskEncoder,add_task
from .models import Selector


class FactoredGoalSelector:
    def __init__(self,base,threshold,*,training_groups,training_rows):
        if base.encoder.kind!='factor32' or base.decoder!='ridge':raise ValueError('Require frozen factor32 ridge')
        if type(threshold) is not int or not 1<=threshold<=512:raise ValueError('Invalid learned goal-count threshold')
        self.base=base;self.encoder=TaskEncoder('goal_lattice64');self.threshold=threshold
        if not np.array_equal(base.encoder.weights,self.encoder.weights[:32]) or not np.array_equal(base.encoder.biases,self.encoder.biases[:32]):
            raise ValueError('Goal encoder must preserve the actual frozen task population')
        self.training_groups=sorted(training_groups);self.training_rows=training_rows
        self.name='factored_monotonic_goal_count_association'
        self.checksum=hashlib.sha256(json.dumps(self.metadata(),sort_keys=True).encode()).hexdigest()

    def metadata(self):
        return dict(kind='factored_goal_count_association',schema_version=1,base_checksum=self.base.checksum,
            encoder_checksum=self.encoder.checksum,learned_goal_count_threshold=self.threshold,
            training_groups=self.training_groups,training_rows=self.training_rows,
            goal_rule='Learned nonnegative count-sum threshold; closer goals cannot revoke an accepted hold',
            learned_encoder_weights=False,motor_training=False)

    def augment_features(self,features,*,task,geometry,observed):return add_task(features,task)
    def reset(self):self.base.reset()

    def predict(self,features,now=None):
        packet=self.encoder.encode(features);counts=packet['spike_counts']
        prefix=self.base.predict(features,now)
        stop_count=int(sum(counts[32:]))
        stop=stop_count>=self.threshold and prefix['proposed_skill']=='run'
        return dict(proposed_skill='wait' if stop else prefix['proposed_skill'],
            reason='learned_goal_count_hold_association' if stop else prefix['reason'],
            base_proposed_skill=prefix['proposed_skill'],goal_count_sum=stop_count,goal_count_threshold=self.threshold,
            spike_counts=counts,active_neuron_ids=packet['active_neuron_ids'],reset_state=packet['reset_state'])

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream,metadata=np.asarray(json.dumps(self.metadata(),sort_keys=True)),
                base_metadata=np.asarray(json.dumps(self.base.inner._metadata(),sort_keys=True)),
                base_encoder_weights=self.base.encoder.weights,base_encoder_biases=self.base.encoder.biases,
                mean=self.base.inner.mean,scale=self.base.inner.scale,bias=self.base.inner.bias,
                coefficients=self.base.inner.coefficients,checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as b:
            meta=json.loads(str(b['metadata'].item()));inner=json.loads(str(b['base_metadata'].item()))
            if meta['kind']!='factored_goal_count_association' or meta['schema_version']!=1:raise ValueError('Unsupported factored goal model')
            base=Selector(TaskEncoder('factor32',b['base_encoder_weights'],b['base_encoder_biases']),'ridge',alpha=inner['alpha'])
            for name in ('mean','scale','bias','coefficients'):setattr(base.inner,name,b[name].copy())
            for name in ('training_groups','training_rows','training_labels'):setattr(base.inner,name,inner[name])
            base.inner._update_checksum();base._checksum()
            if base.checksum!=meta['base_checksum']:raise ValueError('Base decoder changed')
            result=cls(base,meta['learned_goal_count_threshold'],training_groups=meta['training_groups'],training_rows=meta['training_rows'])
            if result.checksum!=str(b['checksum'].item()) or result.metadata()!=meta:raise ValueError('Factored model changed')
            return result


def fit_threshold(base,records):
    encoder=TaskEncoder('goal_lattice64');examples=[]
    for r in records:
        f=r['features']
        if not f.get('goal_mode') or not f['task_progress'] or not r.get('run_eligible',f['predicted_run_clearance']>.15):continue
        if np.hypot(f['goal_forward'],f['goal_left'])>.3:continue
        if r['label'] not in ('run','wait'):continue
        count=int(sum(encoder.encode(f)['spike_counts'][32:]))
        examples.append((count,r['label']=='wait',r['group']))
    if not examples or len({r[1] for r in examples})!=2:raise ValueError('Need moving and hold goal labels')
    return fit_count_examples(base,examples)


def fit_count_examples(base,examples):
    """Rebuild from the published minimal count/teacher-label/group records."""
    from experiments.continual_snn.readout import grouped_weights
    x=np.asarray([r[0] for r in examples]);y=np.asarray([r[1] for r in examples],dtype=int)
    groups=[r[2] for r in examples];w=grouped_weights(groups,y)
    ranking=[dict(threshold=t,weighted_error=float(np.average((x>=t)!=y,weights=w)),
        missed_hold=int(np.sum((x<t)&(y==1))),early_hold=int(np.sum((x>=t)&(y==0)))) for t in range(1,513)]
    best=min(ranking,key=lambda r:(r['weighted_error'],r['missed_hold'],r['threshold']))
    model=FactoredGoalSelector(base,best['threshold'],training_groups=set(groups),training_rows=len(examples))
    return model,dict(selected=best,candidates=ranking,examples=len(examples),groups=len(set(groups)),
        qualification='Causal teacher labels at near-goal observed states; no per-window counterfactual outcome claim')
