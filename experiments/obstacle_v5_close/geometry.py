"""Conservative shape-aware landing corridor with the original disc margin."""
import numpy as np

from experiments.obstacle_v5_refinement.model import observe as parent_observe
from experiments.terrain_goal.geometry import scene_clearance


def landing_clearance(spec, target_index, position, yaw, now):
    """Lower bound on clearance over a straight 1.2 m corridor and 3 s horizon.

    Current target is the committed crossing; every other obstacle is included.
    Space and time sampling gaps are bounded using the distance function's
    Lipschitz constant and the maximum speed of each declared motion type.
    """
    others = tuple(o for i, o in enumerate(spec.obstacles) if i != target_index)
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    points = np.asarray(position)+np.linspace(0., 1.2, 33)[:, None]*direction
    spatial_margin = 1.2/32/2
    max_speed = max((float(np.linalg.norm(o.velocity)) if o.motion == 'linear'
                     else float(np.linalg.norm(o.amplitude))*2*np.pi/o.period
                     for o in others), default=0.)
    temporal_margin = max_speed*(3./16)/2
    minimum = float('inf')
    # Stationary geometry has no temporal sampling error and needs one pass.
    for elapsed in ([0.] if max_speed == 0. else np.linspace(0., 3., 17)):
        obstacles = [(o.at(now+elapsed), o.planning_radius, o.speed_at(now+elapsed)) for o in others]
        for point in points:
            minimum = min(minimum, scene_clearance(point, obstacles, spec.bounds, others))
    return float(minimum-spatial_margin-temporal_margin)


def observe(spec, position, yaw, speed, stable, now, *, robot):
    features, index = parent_observe(spec, position, yaw, speed, stable, now, robot=robot)
    if index is not None:
        features['landing_clearance'] = min(2., landing_clearance(spec, index, position, yaw, now))
    return features, index
