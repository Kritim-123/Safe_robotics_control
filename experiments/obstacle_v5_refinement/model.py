"""Preserved V4 LIF cells plus fine geometry/posture cells and outcome memory."""
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.obstacle_v4.model import ROOT, V3, JumpHead, NAMES as OLD_NAMES, SCALES as OLD_SCALES
from experiments.obstacle_v4.model import observe as old_observe

ASSETS = ROOT/'docs/agents/obstacle_v5_refinement'
PROFILE = ASSETS/'profile.json'
MODEL = ASSETS/'primary.npz'
NAMES = OLD_NAMES+('body_pitch', 'body_height', 'angular_speed')
SCALES = np.r_[OLD_SCALES, .02, .02, .1]
CENTERS = np.r_[np.zeros(len(OLD_NAMES)), 0., .267, 0.]


def observe(spec, position, yaw, speed, stable, now, *, robot):
    features, index = old_observe(spec, position, yaw, speed, stable, now)
    rot = robot.data.body('base').xmat.reshape(3, 3)
    features.update(body_pitch=float(np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2]))),
                    body_height=float(robot.data.qpos[2]), angular_speed=float(np.linalg.norm(robot.data.qvel[3:6])))
    return features, index


class RefinedHead:
    def __init__(self, weights, biases, bank, labels, metadata, scales=SCALES, centers=CENTERS):
        self.weights, self.biases, self.bank, self.labels, self.scales, self.centers = [np.array(x, copy=True) for x in
            (weights, biases, bank, labels, scales, centers)]
        self.metadata = dict(metadata)
        n = len(self.biases)
        if (metadata.get('features') != list(NAMES) or self.weights.shape != (n, len(NAMES))
                or self.bank.ndim != 2 or self.bank.shape[1] != n or not len(self.bank)
                or self.labels.shape != (len(self.bank),) or not np.isin(self.labels, (0, 1)).all()
                or self.scales.shape != (len(NAMES),) or self.centers.shape != (len(NAMES),)
                or not (self.scales > 0).all() or not np.equal(self.bank, np.floor(self.bank)).all()
                or not ((self.bank >= 0) & (self.bank <= 16)).all()
                or not all(np.isfinite(x).all() for x in (self.weights, self.biases, self.bank, self.scales, self.centers))):
            raise ValueError('Invalid refined observation/array contract')
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        for a in (self.weights, self.biases, self.bank, self.labels, self.scales, self.centers):
            digest.update(np.asarray(a, dtype='<f8').tobytes())
        self.checksum = digest.hexdigest()
        for a in (self.weights, self.biases, self.bank, self.labels, self.scales, self.centers):
            a.flags.writeable = False

    @staticmethod
    def encoder_arrays():
        old_w, old_b = JumpHead.seeded_arrays()
        rows = list(np.pad(old_w, ((0, 0), (0, 3))))
        biases = list(old_b)
        ranges = {'friction': (.8, .9), 'height': (.02, .04), 'depth': (.012, .02),
                  'front': (.23, .29), 'body_pitch': (-.015, .015), 'body_height': (.25, .28),
                  'robot_speed': (0., .04), 'angular_speed': (0., .1)}
        for name, (lo, hi) in ranges.items():
            i = NAMES.index(name)
            for sign in (-1., 1.):
                for threshold in np.linspace(lo, hi, 6):
                    row = np.zeros(len(NAMES))
                    row[i] = sign*2*SCALES[i]/(hi-lo)
                    rows.append(row)
                    biases.append(.25+sign*2*(CENTERS[i]-threshold)/(hi-lo))
        return np.array(rows), np.array(biases)

    def encode(self, features):
        values = np.array([features[k] for k in NAMES], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite observation')
        current = self.weights @ np.clip((values-self.centers)/self.scales, -3., 3.)+self.biases
        membrane = np.zeros(len(current)); counts = np.zeros(len(current), dtype=np.int16)
        for _ in range(16):
            membrane = .9*membrane+current
            fired = membrane >= 1.; counts += fired; membrane -= fired
        return counts

    def predict(self, features):
        counts = self.encode(features)
        distances = np.mean(np.abs(self.bank-counts), axis=1)
        distance = float(distances.min())
        matches = self.labels[np.isclose(distances, distance, atol=1e-12, rtol=0)]
        jump = bool(np.all(matches == 1) and distance <= 1.5)
        return dict(proposed_skill='low_bar_jump' if jump else 'wait', spike_counts=counts.tolist(),
                    active_neuron_ids=np.flatnonzero(counts).tolist(), distance=distance, matched_rows=len(matches),
                    reason='fine_count_outcome_association' if jump else 'negative_conflicting_or_unknown_pattern')

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, weights=self.weights, biases=self.biases, bank=self.bank, labels=self.labels,
                                scales=self.scales, centers=self.centers,
                                metadata=np.array(json.dumps(self.metadata, sort_keys=True)), checksum=np.array(self.checksum))

    @classmethod
    def load(cls, path=MODEL):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data['weights'], data['biases'], data['bank'], data['labels'],
                         json.loads(str(data['metadata'].item())), data['scales'], data['centers'])
            if result.checksum != str(data['checksum'].item()):
                raise ValueError('Refined head checksum mismatch')
        if hashlib.sha256(V3.read_bytes()).hexdigest() != result.metadata['v3_sha256']:
            raise ValueError('Inherited V3 changed')
        return result
