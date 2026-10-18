"""Reset LIF count patterns associated with measured low-bar outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'docs/agents/obstacle_v4/primary.npz'
V3 = ROOT / 'docs/agents/terrain_goal_v3/primary.npz'
NAMES = ('present', 'front', 'side', 'height', 'depth', 'span', 'target_speed',
         'alignment', 'landing_clearance', 'stable', 'goal_distance', 'robot_speed', 'friction')
SCALES = np.array([1., .5, .3, .08, .08, 1.5, .3, .3, 1., 1., 2., .2, 1.])


def observe(spec, position, yaw, speed, stable, now):
    """Current geometry only; no mission request, elapsed time or outcome feature."""
    from skills.avoidance import path_clearance
    position = np.asarray(position)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-forward[1], forward[0]])
    candidates = [(i, o, o.at(now)-position) for i, o in enumerate(spec.obstacles)
                  if (o.at(now)-position) @ forward > -.02]
    item = min(candidates, key=lambda entry: np.linalg.norm(entry[2]), default=None)
    values = dict.fromkeys(NAMES, 0.)
    values.update(stable=float(stable), goal_distance=float(np.linalg.norm(np.array(spec.goal)-position)),
                  robot_speed=float(speed), friction=float(spec.friction))
    target = None
    if item is not None:
        target, obstacle, delta = item
        angle = float(np.arctan2(np.sin(obstacle.yaw-yaw), np.cos(obstacle.yaw-yaw)))
        others = [(o.at(now), o.planning_radius, o.speed_at(now))
                  for i, o in enumerate(spec.obstacles) if i != target]
        corridor = path_clearance(position, [position+forward*1.2], others, spec.bounds)
        values.update(present=1., front=float(delta@forward), side=float(delta@left),
                      height=float(obstacle.height),
                      depth=float(2*(obstacle.half_extents[0] if obstacle.shape == 'box' else obstacle.radius)),
                      span=float(2*(obstacle.half_extents[1] if obstacle.shape == 'box' else obstacle.radius)),
                      target_speed=float(np.linalg.norm(obstacle.speed_at(now))), alignment=angle,
                      landing_clearance=float(min(corridor, 2.)))
    return values, target


class JumpHead:
    """Seeded LIF encoder, learned 1-nearest count association, eligibility separate."""
    def __init__(self, weights, biases, bank, labels, metadata):
        self.weights, self.biases = np.asarray(weights), np.asarray(biases)
        self.bank, self.labels = np.asarray(bank), np.asarray(labels)
        self.metadata = metadata
        n = len(self.biases)
        if (self.weights.shape != (n, len(NAMES)) or self.bank.ndim != 2
                or self.bank.shape[1] != n or self.labels.shape != (len(self.bank),)
                or not len(self.bank) or not np.isin(self.labels, [0, 1]).all()
                or not all(np.isfinite(a).all() for a in (self.weights, self.biases, self.bank))
                or not ((self.bank >= 0) & (self.bank <= 16)).all()
                or not np.equal(self.bank, np.floor(self.bank)).all()
                or metadata.get('features') != list(NAMES)):
            raise ValueError('Invalid V4 head arrays/contract')
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for array in (self.weights, self.biases, self.bank, self.labels):
            digest.update(np.asarray(array, dtype='<f8').tobytes())
        self.checksum = digest.hexdigest()

    @staticmethod
    def seeded_arrays():
        rng = np.random.default_rng(20261017)
        # Opposed level-sensitive cells retain magnitude, plus seeded mixtures.
        rows, biases = [], []
        for i in range(len(NAMES)):
            for sign in (-1., 1.):
                for threshold in (-.5, 0., .5, 1.):
                    row = np.zeros(len(NAMES)); row[i] = sign
                    rows.append(row); biases.append(.25-threshold)
        rows.extend(rng.normal(0., .4, (64, len(NAMES))))
        biases.extend(rng.normal(.2, .1, 64))
        return np.array(rows), np.array(biases)

    def encode(self, features):
        x = np.array([features[k] for k in NAMES], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite observation')
        current = self.weights @ np.clip(x/SCALES, -3., 3.) + self.biases
        membrane = np.zeros(len(current)); counts = np.zeros(len(current), dtype=np.int16)
        for _ in range(16):
            membrane = .9*membrane+current
            fired = membrane >= 1.
            counts += fired
            membrane -= fired
        return counts

    def predict(self, features):
        counts = self.encode(features)
        distances = np.mean(np.abs(self.bank-counts), axis=1)
        minimum = float(distances.min())
        matches = self.labels[np.isclose(distances, minimum, atol=1e-12, rtol=0)]
        # Conflicting closest outcomes or a distant packet abstain.
        jump = bool(np.all(matches == 1) and minimum <= 1.5)
        return dict(proposed_skill='low_bar_jump' if jump else 'wait',
                    spike_counts=counts.tolist(), active_neuron_ids=np.flatnonzero(counts).tolist(),
                    distance=minimum, matched_rows=int(len(matches)),
                    reason='learned_obstacle_count_association' if jump else 'negative_conflicting_or_unknown_pattern')

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, weights=self.weights, biases=self.biases,
                                bank=self.bank, labels=self.labels,
                                metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
                                checksum=np.array(self.checksum))

    @classmethod
    def load(cls, path=MODEL):
        with np.load(path, allow_pickle=False) as data:
            model = cls(data['weights'], data['biases'], data['bank'], data['labels'],
                        json.loads(str(data['metadata'].item())))
            if model.checksum != str(data['checksum'].item()):
                raise ValueError('V4 checksum mismatch')
            if hashlib.sha256(V3.read_bytes()).hexdigest() != model.metadata['v3_sha256']:
                raise ValueError('Inherited V3 model changed')
            return model
