"""Versioned, named physical observations; no outcome or skill-label inputs."""
import numpy as np

# Reuse the established 14-feature meanings, but keep this experiment's schema
# explicit and independently versioned. No normalization is fitted on outcomes.
FEATURES = ('obstacle_present', 'obstacle_forward', 'obstacle_left',
            'obstacle_v_forward', 'obstacle_v_left', 'obstacle_radius',
            'obstacle_height', 'obstacle_width', 'footprint_clearance',
            'goal_forward', 'goal_left', 'speed', 'base_height', 'jump_requested')
UNITS = ('binary', 'm', 'm', 'm/s', 'm/s', 'm', 'm', 'm', 'm', 'm', 'm', 'm/s', 'm', 'binary')
SCALES = (1., 3., 3., .5, .5, .5, 1., 1., 3., 3., 3., .5, .35, 1.)


def feature_vector(features):
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError('Supply exactly the named observation features')
    try:
        x = np.asarray([features[k] for k in FEATURES], dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError('Observation features must be numeric scalars') from error
    if x.shape != (len(FEATURES),) or not np.isfinite(x).all():
        raise ValueError('Observation features must be finite scalars')
    if x[0] not in (0, 1) or x[-1] not in (0, 1):
        raise ValueError('Presence and task-request features must be binary')
    return x


def observation_from_simulation(spec, observation, jump_requested, *, observation_id, context_id):
    """Use the existing causal feature builder at the current settled state."""
    from hybrid.behavior_dataset import observation_features
    from hybrid.skill_lif import FEATURES as existing_features
    if FEATURES != existing_features:
        raise ValueError('Existing feature schema changed; review the observation adapter')
    values = observation_features(spec, observation, jump_requested)
    return dict(observation_id=observation_id, context_id=context_id,
                time_s=float(observation['time']), source='mujoco_state',
                features={name: float(v) for name, v in zip(FEATURES, values)},
                metadata=dict(robot_observation=observation, scenario=spec.to_dict(),
                              jump_request_is_explicit=True))
