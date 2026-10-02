"""NumPy inference for a real LIF spiking selector trained with PyTorch.

Analog observations provide constant input current over a finite 16-step
encoding window. Both hidden layers emit binary spikes and reset membrane
voltage. A linear readout decodes mean second-layer spike counts.
"""
from pathlib import Path
import numpy as np

from hybrid.perception import FEATURE_NAMES
from skills.avoidance import SKILL_NAMES

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'skills' / 'policies' / 'selector_snn' / 'model.npz'


class SpikingSelector:
    def __init__(self, path=DEFAULT_MODEL):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as bundle:
            self.weights = {key: bundle[key].copy() for key in bundle.files}
        w = self.weights
        if tuple(w['feature_names']) != FEATURE_NAMES or tuple(w['skill_names']) != SKILL_NAMES:
            raise ValueError('SNN observation/skill schema does not match the runtime')
        self.steps, self.beta = int(w['steps']), float(w['beta'])
        if not 1 <= self.steps <= 128 or not 0 <= self.beta < 1:
            raise ValueError('Invalid SNN temporal parameters')
        if w['w1'].shape != (64, len(FEATURE_NAMES)) or w['w2'].shape != (32, 64) or w['w3'].shape != (4, 32):
            raise ValueError('Unexpected SNN architecture')
        for key, shape in [('b1', (64,)), ('b2', (32,)), ('b3', (4,)),
                           ('mean', (len(FEATURE_NAMES),)), ('std', (len(FEATURE_NAMES),))]:
            if w[key].shape != shape:
                raise ValueError(f'Unexpected SNN parameter shape: {key}')
        for key in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'mean', 'std'):
            if not np.all(np.isfinite(w[key])):
                raise ValueError(f'Nonfinite SNN parameters: {key}')
        if np.any(w['std'] <= 0):
            raise ValueError('SNN normalization requires positive standard deviations')

    def predict(self, features, *, trace=False):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            raise ValueError('Expected a finite feature vector matching FEATURE_NAMES')
        w = self.weights
        x = np.clip((features - w['mean']) / w['std'], -5, 5)
        current = w['w1'] @ x + w['b1']
        membrane1 = np.zeros(64, dtype=np.float32)
        membrane2 = np.zeros(32, dtype=np.float32)
        counts = np.zeros(32, dtype=np.float32)
        spike_count = 0
        raster = []
        for _ in range(self.steps):
            membrane1 = self.beta * membrane1 + current
            spike1 = (membrane1 >= 1).astype(np.float32)
            membrane1 -= spike1
            membrane2 = self.beta * membrane2 + w['w2'] @ spike1 + w['b2']
            spike2 = (membrane2 >= 1).astype(np.float32)
            membrane2 -= spike2
            counts += spike2
            spike_count += int(spike1.sum() + spike2.sum())
            if trace:
                raster.append(np.r_[spike1, spike2])
        scores = w['w3'] @ (counts / self.steps) + w['b3']
        diagnostics = dict(spikes=spike_count, opportunities=self.steps*96,
                           spike_fraction=spike_count / (self.steps*96))
        if trace:
            diagnostics['raster'] = np.asarray(raster)
        return scores, diagnostics
