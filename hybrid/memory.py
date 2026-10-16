"""Frozen-checkpoint, across-observation LIF membrane retention ablation.

Counts reset every window. This is not a sequence-trained policy. The optional
wall-time decay multiplies the existing per-microstep beta, not replaces it.
"""
import numpy as np

from hybrid.snn import SpikingSelector, DEFAULT_MODEL


class TemporalSelector:
    def __init__(self, path=DEFAULT_MODEL, *, retain=True, tau=.5, max_gap=.5):
        if not np.isfinite(tau) or tau <= 0 or not np.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('Temporal constants must be finite and positive')
        self.network = SpikingSelector(path)
        self.path = self.network.path
        self.retain, self.tau, self.max_gap = bool(retain), tau, max_gap
        self.updates = self.resets = 0
        self.reset('initial')

    def reset(self, reason='episode'):
        self.membranes = (np.zeros(64, np.float32), np.zeros(32, np.float32))
        self.last_time = self.identity = self.cached = None
        if reason == 'episode':
            self.updates = self.resets = 0
        self.reset_reason = reason
        self.resets += 1

    def observe(self, features, now, identity):
        if not np.isfinite(now) or now < 0:
            self.reset('invalid time')
            raise ValueError('Observation time must be finite and nonnegative')
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (12,) or not np.all(np.isfinite(features)):
            self.reset('invalid features')
            raise ValueError('Invalid observation features')
        if self.last_time is not None and now < self.last_time:
            self.reset('out of order')
            raise ValueError('Out-of-order observation')
        if identity != self.identity:
            self.reset('obstacle changed')
        if self.last_time is not None and now == self.last_time:
            return self.cached
        if self.last_time is not None and now-self.last_time > self.max_gap:
            self.reset('stale gap')
        elapsed = 0 if self.last_time is None else now-self.last_time
        state = tuple(v*np.exp(-elapsed/self.tau) for v in self.membranes)
        if not self.retain:
            state = tuple(np.zeros_like(v) for v in state)
        try:
            scores, diagnostics = self.network.predict(features, membrane_state=state)
        except Exception:
            self.reset('selector failure')
            raise
        self.membranes = diagnostics.pop('_membrane_state')
        diagnostics.update(memory_retained=self.retain, memory_reset=self.reset_reason,
                           memory_updates=self.updates+1)
        self.reset_reason = None
        self.last_time, self.identity = now, identity
        self.updates += 1
        self.cached = scores, diagnostics
        return self.cached
