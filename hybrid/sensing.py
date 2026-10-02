"""Optional reproducible imperfections on privileged obstacle observations.

This adapter is a sensor stress model, not an implementation of LiDAR/vision.
It delays noisy snapshots; it never exposes a future trajectory to the controller.
"""
from collections import deque
import numpy as np


class ObstacleSensor:
    def __init__(self, *, position_noise=0.0, velocity_noise=0.0, delay=0.0, filter_tau=0.0, seed=0):
        values = (position_noise, velocity_noise, delay, filter_tau)
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError('Sensor noise and delay must be finite and nonnegative')
        self.position_noise, self.velocity_noise, self.delay, self.filter_tau = values
        self.rng = np.random.default_rng(seed)
        self.history = deque()
        self.filtered, self.last_time = None, None

    def observe(self, now, obstacles):
        snapshot = []
        for center, radius, velocity in obstacles:
            center = np.asarray(center).copy()
            velocity = np.asarray(velocity).copy()
            if self.position_noise:
                center = center + self.rng.normal(0, self.position_noise, 2)
            if self.velocity_noise:
                velocity = velocity + self.rng.normal(0, self.velocity_noise, 2)
            snapshot.append((center, radius, velocity))
        if self.filter_tau and self.filtered is not None:
            dt = now - self.last_time
            alpha = dt / (self.filter_tau + dt)
            # Motion-compensated exponential smoothing, with no map or script.
            # A production detector also needs object association; this simulated
            # sensor has stable obstacle indices and known radii.
            snapshot = [(old_c + old_v*dt + alpha*(c-old_c-old_v*dt), r,
                         old_v + alpha*(v-old_v))
                        for (c, r, v), (old_c, _, old_v) in zip(snapshot, self.filtered)]
        self.filtered, self.last_time = snapshot, now
        self.history.append((now, snapshot))
        threshold = now - self.delay + 1e-8
        while len(self.history) > 1 and self.history[1][0] <= threshold:
            self.history.popleft()
        # At startup the first available snapshot is held until the delay fills.
        return [(c.copy(), r, v.copy()) for c, r, v in self.history[0][1]]
