"""Experimental skill policy with one binary-spiking output LIF per skill."""
from pathlib import Path
import numpy as np

SKILLS = ('wait', 'run', 'jump')
FEATURES = ('obstacle_present', 'obstacle_forward', 'obstacle_left',
            'obstacle_v_forward', 'obstacle_v_left', 'obstacle_radius',
            'obstacle_height', 'obstacle_width', 'footprint_clearance',
            'goal_forward', 'goal_left', 'speed', 'base_height', 'jump_requested')
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'output'/'skill_lif'/'model'/'model.npz'


def decode_counts(counts, allowed):
    counts, allowed = np.asarray(counts), np.asarray(allowed, dtype=bool)
    if counts.shape != (len(SKILLS),) or allowed.shape != counts.shape or not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError('Require finite nonnegative spike counts and one eligibility per skill')
    if not allowed.any():
        return None, 'no eligible skill'
    if np.max(counts[allowed]) <= 0:
        return (0, 'zero-spike wait fallback') if allowed[0] else (None, 'no eligible spiking output')
    return int(np.argmax(np.where(allowed, counts, -1))), 'spike-count winner'


class SkillLIF:
    def __init__(self, path=DEFAULT_MODEL):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.w = {k: data[k].copy() for k in data.files}
        w = self.w
        if tuple(w['feature_names']) != FEATURES or tuple(w['skill_names']) != SKILLS:
            raise ValueError('Policy feature/action schema mismatch')
        self.steps, self.beta = int(w['steps']), float(w['beta'])
        if not 1 <= self.steps <= 128 or not 0 <= self.beta < 1:
            raise ValueError('Invalid LIF parameters')
        shapes = {'w1': (32,len(FEATURES)), 'b1': (32,), 'w2': (len(SKILLS),32),
                  'b2': (len(SKILLS),), 'mean': (len(FEATURES),), 'std': (len(FEATURES),)}
        for key, shape in shapes.items():
            if w[key].shape != shape or not np.isfinite(w[key]).all():
                raise ValueError(f'Invalid parameter {key}')
        if np.any(w['std'] <= 0):
            raise ValueError('Normalization scales must be positive')

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
            raise ValueError('Require finite causal feature vector')
        w = self.w
        x = np.clip((x-w['mean'])/w['std'], -5, 5)
        current = w['w1']@x+w['b1']
        v1, v2 = np.zeros(32,np.float32), np.zeros(len(SKILLS),np.float32)
        outputs, hidden = [], []
        for _ in range(self.steps):
            v1 = self.beta*v1+current
            s1 = (v1 >= 1).astype(np.float32)
            v1 -= s1
            v2 = self.beta*v2+w['w2']@s1+w['b2']
            s2 = (v2 >= 1).astype(np.float32)
            v2 -= s2
            hidden.append(s1)
            outputs.append(s2)
        raster = np.asarray(outputs)
        return raster.sum(0), {'output_spikes': raster, 'hidden_spikes': np.asarray(hidden)}

    def choose(self, features, allowed):
        counts, diagnostics = self.predict(features)
        choice, reason = decode_counts(counts, allowed)
        return choice, reason, counts, diagnostics
