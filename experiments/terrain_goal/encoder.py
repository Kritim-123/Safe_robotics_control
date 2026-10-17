"""Explicit task populations; comparison against the unchanged 80-neuron code."""
from dataclasses import asdict
import hashlib,json
import numpy as np
from experiments.continual_snn.encoding import PacketConfig,spike_embedding
from experiments.continual_snn.predictive import PredictivePacketEncoder

TASK_FEATURES=('task_progress','task_hold','task_jump')
FACTOR_FEATURES=TASK_FEATURES+('predicted_run_clearance','predicted_jump_clearance','footprint_clearance')


def add_task(features,task=None):
    result=dict(features)
    if task is None:
        # Historical packets use a zeroed goal vector for hold. No labels used.
        task='jump' if features['jump_requested'] else 'progress' if np.hypot(features['goal_forward'],features['goal_left'])>1e-8 else 'hold'
    if task not in ('progress','hold','jump'):raise ValueError('Unknown task')
    result.update({n:float(task==n.removeprefix('task_')) for n in TASK_FEATURES})
    return result


class TaskEncoder:
    def __init__(self,kind='task96',weights=None,biases=None):
        if kind not in ('task96','factor32','goal64','goal_lattice64'):raise ValueError('Unknown encoder')
        self.kind=kind;self.prefix=PredictivePacketEncoder();self.base=self.prefix.base
        rng=np.random.default_rng(2026101701)
        n,dimensions=(16,3) if kind=='task96' else (64,18) if kind=='goal64' else (64,6) if kind=='goal_lattice64' else (32,6)
        if kind=='goal64' and weights is None:
            prefix=TaskEncoder('factor32')
            added=np.zeros((64,18));added[:32,:6]=prefix.weights
            added[32:,6:]=rng.normal(size=(32,12))/np.sqrt(12)
            weights=added;biases=np.r_[prefix.biases,rng.normal(0,.1,32)]
        if kind=='goal_lattice64' and weights is None:
            prefix=TaskEncoder('factor32')
            added=np.zeros((64,6));added[:32]=prefix.weights
            weights=added;biases=np.r_[prefix.biases,rng.normal(0,.01,32)]
        self.weights=np.asarray(weights if weights is not None else rng.normal(size=(n,dimensions))/np.sqrt(dimensions),dtype=np.float64)
        self.biases=np.asarray(biases if biases is not None else rng.normal(0,.1,n),dtype=np.float64)
        if self.weights.shape!=(n,dimensions) or self.biases.shape!=(n,) or not np.isfinite(self.weights).all() or not np.isfinite(self.biases).all():
            raise ValueError('Invalid task population')
        self.weights.flags.writeable=self.biases.flags.writeable=False
        self.feature_names=tuple(self.prefix.feature_names)+TASK_FEATURES
        if kind in ('goal64','goal_lattice64'):self.feature_names+=('goal_mode','run_active','braking_active')
        self.config=PacketConfig(neurons=96 if kind=='task96' else 64 if kind in ('goal64','goal_lattice64') else 32)
        self.contract=dict(kind=kind,seed=2026101701,prefix_checksum=self.prefix.checksum,
            features=list(TASK_FEATURES if kind=='task96' else FACTOR_FEATURES),steps=16,
            note='factor32 intentionally isolates immediate task/clearance decision; terrain stays in motor adapter')
        if kind=='goal64':self.contract.update(goal_centres=[0.,.04,.08,.12,.2,.4,.8,1.6],
            goal_widths=[.04,.04,.04,.06,.1,.2,.4,.8],
            goal_features='goal_mode, gated Gaussian distance population, speed, current run and braking state')
        if kind=='goal_lattice64':self.contract.update(goal_thresholds=[.02,.04,.06,.08,.10,.12,.16,.24],
            goal_gains=[.8,1.2,1.6,2.],goal_features='mode-gated monotonic proximity channels; no encoder learning')
        digest=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode())
        digest.update(self.weights.astype('<f8').tobytes());digest.update(self.biases.astype('<f8').tobytes())
        self.checksum=digest.hexdigest()

    def reset(self):pass

    def encode(self,features,now=None):
        f=add_task(features) if not all(n in features for n in TASK_FEATURES) else features
        names=TASK_FEATURES if self.kind=='task96' else FACTOR_FEATURES
        x=np.clip(np.asarray([f[n] for n in names],dtype=np.float64),-3,3)
        if self.kind=='goal64':
            mode=float(f.get('goal_mode',0.))
            distance=float(np.hypot(f['goal_forward'],f['goal_left']))
            centres=np.asarray(self.contract['goal_centres']);widths=np.asarray(self.contract['goal_widths'])
            x=np.r_[x,mode,mode*np.exp(-.5*((distance-centres)/widths)**2),
                    mode*f['speed'],mode*f.get('run_active',0.),mode*f.get('braking_active',0.)]
        if not np.isfinite(x).all():raise ValueError('Nonfinite packet')
        current=self.weights@x+self.biases;v=np.zeros(len(current));raster=[]
        if self.kind=='goal_lattice64':
            distance=float(np.hypot(f['goal_forward'],f['goal_left']))
            mode=float(f.get('goal_mode',0.))
            levels=np.maximum(0.,1-distance/np.repeat(self.contract['goal_thresholds'],4))
            current[32:]=mode*levels*np.tile(self.contract['goal_gains'],8)+self.biases[32:]
        for _ in range(16):
            v=self.base.metadata['beta_per_step']*v+current
            spikes=(v>=self.base.metadata['threshold']).astype(np.uint8)
            v-=self.base.metadata['threshold']*spikes;raster.append(spikes)
        raster=np.asarray(raster)
        if self.kind=='task96':raster=np.column_stack((self.prefix.encode(f)['raster'],raster))
        counts=raster.sum(axis=0)
        return dict(spike_counts=counts.tolist(),raster=raster,embedding=spike_embedding(raster),
            active_neuron_ids=np.flatnonzero(counts).tolist(),reset_state='zero_membranes_each_packet',
            normalized_inputs=x,encoder_checksum=self.checksum)

    def embed_many(self,features):return np.asarray([self.encode(f)['embedding'] for f in features])
