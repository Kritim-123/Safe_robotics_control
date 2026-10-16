"""Frozen seeded random LIF encoder, independent of skills and robot actuation."""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .observations import FEATURES, UNITS, SCALES, feature_vector


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class RandomLIFEncoder:
    """One dense input -> pattern-neuron layer, constant-current encoding.

    Each call starts at zero membrane state. Output neuron IDs have no action
    meanings. A 16-step window is an internal encoding window, not 16 sensor
    frames or a claimed physical/biological duration.
    """
    def __init__(self, metadata, weights, biases, scales):
        self.metadata = json.loads(_json(metadata))
        m = self.metadata
        if (m.get('schema_version') != 1 or m.get('neuron_model') != 'LIF'
                or m.get('feature_names') != list(FEATURES) or m.get('feature_units') != list(UNITS)
                or m.get('reset_between_observations') is not True
                or m.get('encoding') != 'constant_current'
                or m.get('reset_rule') != 'subtract_threshold_once_per_step'):
            raise ValueError('Unsupported encoder/observation contract')
        for name, maximum in (('pattern_neurons', 4096), ('window_steps', 4096)):
            if type(m.get(name)) is not int or not 1 <= m[name] <= maximum:
                raise ValueError(f'Invalid {name}')
        if (not np.isfinite([m['beta_per_step'], m['threshold'], m['input_clip']]).all()
                or not 0 <= m['beta_per_step'] < 1 or min(m['threshold'], m['input_clip']) <= 0):
            raise ValueError('Invalid LIF or normalization parameters')
        self.weights, self.biases, self.scales = [np.array(v, dtype=np.float64, copy=True)
                                                for v in (weights, biases, scales)]
        if (self.weights.shape != (m['pattern_neurons'], len(FEATURES))
                or self.biases.shape != (m['pattern_neurons'],) or self.scales.shape != (len(FEATURES),)
                or not all(np.isfinite(v).all() for v in (self.weights, self.biases, self.scales))
                or np.any(self.scales <= 0)):
            raise ValueError('Invalid encoder array shape, value or input scales')
        for array in (self.weights, self.biases, self.scales):
            array.flags.writeable = False
        self.checksum = self._checksum()

    @classmethod
    def initialize(cls, seed=42, pattern_neurons=16, window_steps=16, beta_per_step=.9, threshold=1.):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('Seed must be an integer in [0, 2**64)')
        if type(pattern_neurons) is not int or not 1 <= pattern_neurons <= 4096:
            raise ValueError('Invalid pattern neuron count')
        rng = np.random.Generator(np.random.PCG64(seed))
        weight_std = 1/np.sqrt(len(FEATURES))
        weights = rng.normal(0., weight_std, (pattern_neurons, len(FEATURES)))
        biases = rng.normal(0., .1, pattern_neurons)
        metadata = dict(schema_version=1, neuron_model='LIF', seed=seed, rng='PCG64',
                        feature_names=list(FEATURES), feature_units=list(UNITS),
                        pattern_neurons=pattern_neurons, window_steps=window_steps,
                        beta_per_step=beta_per_step, threshold=threshold, input_clip=3.,
                        encoding='constant_current', reset_between_observations=True,
                        reset_rule='subtract_threshold_once_per_step',
                        initialization=dict(weights_mean=0., weights_std=float(weight_std),
                                            biases_mean=0., biases_std=.1, distribution='normal',
                                            draw_order='all weights row-major, then biases'),
                        runtime=dict(python=platform.python_version(), numpy=np.__version__),
                        dtype='float64', learned_weights=False)
        return cls(metadata, weights, biases, SCALES)

    def _checksum(self):
        digest = hashlib.sha256(_json(self.metadata).encode('utf-8'))
        for array in (self.weights, self.biases, self.scales):
            # Canonical little-endian bytes, independent of archive timestamps.
            digest.update(np.asarray(array, dtype='<f8').tobytes(order='C'))
        return digest.hexdigest()

    def save(self, path):
        if self._checksum() != self.checksum:
            raise ValueError('Frozen encoder contents changed')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, metadata=np.asarray(_json(self.metadata)),
                                weights=self.weights, biases=self.biases, scales=self.scales,
                                checksum=np.asarray(self.checksum))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {'metadata', 'weights', 'biases', 'scales', 'checksum'}:
                raise ValueError('Unexpected model archive contents')
            result = cls(json.loads(str(bundle['metadata'].item())), bundle['weights'],
                         bundle['biases'], bundle['scales'])
            if result.checksum != str(bundle['checksum'].item()):
                raise ValueError('Encoder checksum mismatch')
        return result

    def encode(self, features):
        if self._checksum() != self.checksum:
            raise ValueError('Frozen encoder contents changed')
        raw = feature_vector(features)
        x = np.clip(raw/self.scales, -self.metadata['input_clip'], self.metadata['input_clip'])
        current = self.weights @ x + self.biases
        if not np.isfinite(current).all():
            raise ValueError('Nonfinite input current')
        membrane = np.zeros(len(self.biases), dtype=np.float64)
        spikes, before, after = [], [], []
        beta, threshold = self.metadata['beta_per_step'], self.metadata['threshold']
        for _ in range(self.metadata['window_steps']):
            membrane = beta*membrane+current
            if not np.isfinite(membrane).all():
                raise ValueError('Nonfinite membrane state')
            before.append(membrane.copy())
            fired = (membrane >= threshold).astype(np.uint8)
            membrane = membrane-threshold*fired
            spikes.append(fired)
            after.append(membrane.copy())
        raster = np.asarray(spikes, dtype=np.uint8)
        counts = raster.sum(axis=0, dtype=np.int64)
        return dict(encoder_checksum=self.checksum, active_neuron_ids=np.flatnonzero(counts).tolist(),
                    spike_counts=counts.tolist(), output_raster=raster.tolist(),
                    normalized_inputs=x.tolist(), input_current=current.tolist(),
                    membrane_before_reset=np.asarray(before).tolist(),
                    membrane_after_reset=np.asarray(after).tolist(),
                    silent=not bool(counts.any()), reset_state='zero_membranes')

    def record(self, observation):
        for key in ('observation_id', 'context_id', 'source'):
            if not isinstance(observation.get(key), str) or not observation[key].strip():
                raise ValueError(f'Missing or invalid {key}')
        timestamp = observation.get('time_s')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Require finite nonnegative observation time_s')
        # Metadata and any future labels never enter the encoder.
        result = self.encode(observation['features'])
        return dict(schema_version=1, observation=json.loads(_json(observation)),
                    recording=result, skill_label=None, decoder_status='not_implemented')
