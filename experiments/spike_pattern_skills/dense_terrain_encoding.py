"""Five-centimetre terrain sampling, versioned separately from the coarse encoder."""
import numpy as np

from .observations import observation_from_simulation
from .terrain_encoding import TerrainAwareEncoder

DENSE_FORWARD = tuple(float(v/20) for v in range(29))


def dense_surface_features(spec, observation):
    position, yaw = np.asarray(observation['position']), observation['yaw']
    forward, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    heights, friction, present = [], [], False
    for distance in DENSE_FORWARD:
        row = []
        for side in (-.2, 0., .2):
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
    return dict(terrain_present=float(present), surface_rise=float(np.max(heights)-heights[0, 1]),
        largest_up_step=float(max(0., changes.max())), largest_down_step=float(max(0., -changes.min())),
        longitudinal_slope=float(np.mean(np.diff(heights[:, 1])/.05)),
        cross_slope=float(np.mean((heights[:, 2]-heights[:, 0])/.4)),
        surface_roughness=float(np.std(np.diff(heights, n=2, axis=0))),
        friction_loss=float(max(0., spec.friction-min(friction))))


def dense_observation(spec, observation, jump_requested, **identifiers):
    item = observation_from_simulation(spec, observation, jump_requested, **identifiers)
    item['features'].update(dense_surface_features(spec, observation))
    item['metadata']['observation_schema'] = 'terrain_v3_dense_22_inputs'
    item['metadata']['surface_source'] = 'current privileged static terrain map; not camera perception'
    return item


class DenseTerrainEncoder(TerrainAwareEncoder):
    observation_adapter = staticmethod(dense_observation)
    observation_schema = 'terrain_v3_dense_22_inputs'
    forward_samples = DENSE_FORWARD

    def __init__(self, metadata, weights, biases, scales):
        if metadata.get('observation_schema') != 'terrain_v3_dense_22_inputs':
            raise ValueError('Dense encoder requires the dense observation schema')
        if metadata.get('terrain_extension', {}).get('forward_samples') != list(DENSE_FORWARD):
            raise ValueError('Dense encoder sampling contract mismatch')
        super().__init__(metadata, weights, biases, scales)

    @classmethod
    def extend(cls, legacy, seed=4242):
        coarse = TerrainAwareEncoder.extend(legacy, seed)
        metadata = dict(coarse.metadata, observation_schema='terrain_v3_dense_22_inputs',
            terrain_extension=dict(coarse.metadata['terrain_extension'], forward_samples=list(DENSE_FORWARD)))
        return cls(metadata, coarse.weights, coarse.biases, coarse.scales)
