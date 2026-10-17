"""Versioned privileged surface observations and frozen random LIF extension."""
import numpy as np

from .encoder import RandomLIFEncoder
from .observations import FEATURES, UNITS, SCALES, feature_vector, observation_from_simulation

TERRAIN_FEATURES = ('terrain_present', 'surface_rise', 'largest_up_step', 'largest_down_step',
                    'longitudinal_slope', 'cross_slope', 'surface_roughness', 'friction_loss')
TERRAIN_UNITS = ('binary', 'm', 'm', 'm', 'ratio', 'ratio', 'm', 'coefficient')
TERRAIN_SCALES = (1., .06, .04, .04, .15, .15, .025, .4)
FORWARD_SAMPLES = (0., .2, .4, .6, .8, 1., 1.2, 1.4)
SIDE_SAMPLES = (-.2, 0., .2)


def surface_features(spec, observation):
    position, yaw = np.asarray(observation['position']), observation['yaw']
    forward, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    heights, friction, present = [], [], False
    for distance in FORWARD_SAMPLES:
        row = []
        for side in SIDE_SAMPLES:
            point = position+forward*distance+left*side
            supports = [(p.surface_height(*point), p.friction) for p in spec.terrain]
            supports = [(h, f) for h, f in supports if h is not None]
            height = max([0.]+[h for h, _ in supports])
            row.append(height)
            friction.append(min([f for h, f in supports if abs(h-height) < 1e-8] or [spec.friction]))
            present |= bool(supports)
        heights.append(row)
    heights = np.asarray(heights)
    changes = np.diff(heights, axis=0)
    center = heights[:, 1]
    return dict(terrain_present=float(present), surface_rise=float(np.max(heights)-heights[0, 1]),
        largest_up_step=float(max(0., changes.max())), largest_down_step=float(max(0., -changes.min())),
        longitudinal_slope=float(np.mean(np.diff(center)/.2)),
        cross_slope=float(np.mean((heights[:, 2]-heights[:, 0])/.4)),
        surface_roughness=float(np.std(np.diff(heights, n=2, axis=0))),
        friction_loss=float(max(0., spec.friction-min(friction))))


def terrain_observation(spec, observation, jump_requested, **identifiers):
    item = observation_from_simulation(spec, observation, jump_requested, **identifiers)
    item['features'].update(surface_features(spec, observation))
    item['metadata']['observation_schema'] = 'terrain_v2_22_inputs'
    item['metadata']['surface_source'] = 'current privileged static terrain map; not camera perception'
    return item


class TerrainAwareEncoder(RandomLIFEncoder):
    feature_names = FEATURES+TERRAIN_FEATURES
    feature_units = UNITS+TERRAIN_UNITS
    input_scales = SCALES+TERRAIN_SCALES
    observation_adapter = staticmethod(terrain_observation)
    observation_schema = 'terrain_v2_22_inputs'
    forward_samples = FORWARD_SAMPLES

    def __init__(self, metadata, weights, biases, scales):
        if (metadata.get('observation_schema') != self.observation_schema
                or metadata.get('terrain_extension', {}).get('forward_samples') != list(self.forward_samples)):
            raise ValueError('Terrain encoder observation/sampling schema mismatch')
        super().__init__(metadata, weights, biases, scales)

    def _feature_vector(self, features):
        if not isinstance(features, dict) or set(features) != set(self.feature_names):
            raise ValueError('Require exactly the versioned 22 named inputs')
        original = feature_vector({k: features[k] for k in FEATURES})
        extra = np.asarray([features[k] for k in TERRAIN_FEATURES], dtype=np.float64)
        if extra.shape != (8,) or not np.isfinite(extra).all() or extra[0] not in (0., 1.):
            raise ValueError('Invalid terrain inputs')
        return np.concatenate([original, extra])

    @classmethod
    def extend(cls, legacy, seed=4242):
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError('Use a valid extension seed')
        rng = np.random.Generator(np.random.PCG64(seed))
        extra = rng.normal(0., 1/np.sqrt(8), (16, 8))
        metadata = dict(legacy.metadata, feature_names=list(cls.feature_names), feature_units=list(cls.feature_units),
            observation_schema='terrain_v2_22_inputs', legacy_encoder_checksum=legacy.checksum,
            terrain_extension=dict(seed=seed, distribution='normal', std=float(1/np.sqrt(8)),
                existing_columns_and_biases_preserved=True, forward_samples=list(FORWARD_SAMPLES),
                lateral_samples=list(SIDE_SAMPLES)), learned_weights=False)
        return cls(metadata, np.column_stack([legacy.weights, extra]), legacy.biases, cls.input_scales)
