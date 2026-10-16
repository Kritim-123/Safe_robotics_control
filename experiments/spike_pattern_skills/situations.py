"""Fixed 10 x 10 scene protocol, specified before observing neural outputs."""
from dataclasses import dataclass

import numpy as np

from hybrid.arena import Scenario, Obstacle

FAMILIES = ('clear', 'static_cylinder', 'static_box', 'low_bar',
            'crossing_left_to_right', 'crossing_right_to_left',
            'approaching', 'receding', 'oscillating', 'mixed_obstacles')


@dataclass(frozen=True)
class Situation:
    family: str
    variant: int
    scene_seed: int
    jump_requested: bool
    scenario: Scenario


def situations(seed=20261016):
    if type(seed) is not int or not 0 <= seed < 2**63-100:
        raise ValueError('Require a nonnegative integer study seed')
    cases = []
    for family_index, family in enumerate(FAMILIES):
        for variant in range(10):
            scene_seed = seed+10*family_index+variant
            rng = np.random.default_rng(scene_seed)
            yaw = float(rng.uniform(-np.pi, np.pi))
            cs, sn = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cs, -sn], [sn, cs]])
            start = rng.uniform(-.2, .2, 2)
            def point(x, y):
                return tuple(float(v) for v in start+rotation @ [x, y])
            def velocity(x, y):
                return tuple(float(v) for v in rotation @ [x, y])
            radius = float(rng.uniform(.15, .28))
            side = float(rng.uniform(-.35, .35))
            height = float(rng.uniform(.25, .9))
            obstacles = ()
            if family == 'static_cylinder':
                obstacles = (Obstacle('cylinder', point(rng.uniform(1., 2.8), side), radius, height=height),)
            elif family == 'static_box':
                obstacles = (Obstacle('box', point(rng.uniform(1.3, 2.8), side), shape='box',
                    half_extents=(float(rng.uniform(.15, .35)), float(rng.uniform(.12, .3))),
                    height=height, yaw=yaw+float(rng.uniform(-.7, .7))),)
            elif family == 'low_bar':
                obstacles = (Obstacle('bar', point(rng.uniform(1.5, 2.8), side), shape='box',
                    half_extents=(float(rng.uniform(.06, .15)), float(rng.uniform(.5, .85))),
                    height=float(rng.uniform(.04, .12)), yaw=yaw),)
            elif family.startswith('crossing_'):
                direction = -1 if family == 'crossing_left_to_right' else 1
                obstacles = (Obstacle('crossing', point(rng.uniform(1.1, 2.6), -direction*rng.uniform(1.1, 1.9)),
                    radius, velocity=velocity(0., direction*rng.uniform(.12, .4)), height=height),)
            elif family in ('approaching', 'receding'):
                approaching = family == 'approaching'
                obstacles = (Obstacle('moving', point(rng.uniform(2.1, 3.3) if approaching else rng.uniform(1.1, 2.), side),
                    radius, velocity=velocity((-1 if approaching else 1)*rng.uniform(.12, .32), 0.), height=height),)
            elif family == 'oscillating':
                obstacles = (Obstacle('oscillator', point(rng.uniform(1.2, 2.6), side), radius,
                    motion='oscillate', amplitude=velocity(0., rng.uniform(.4, .9)),
                    period=float(rng.uniform(4., 10.)), height=height),)
            elif family == 'mixed_obstacles':
                obstacles = (Obstacle('box', point(rng.uniform(1.4, 2.5), -.5), shape='box',
                                half_extents=(.23, .18), height=height, yaw=yaw+.35),
                             Obstacle('side_cylinder', point(1., 1.5), radius, height=.7),
                             Obstacle('crossing_far', point(2.9, -1.5), .18,
                                velocity=velocity(0., rng.uniform(.12, .25)), height=.6))
            spec = Scenario(f'{family}-{variant:02d}', start=tuple(float(v) for v in start),
                goal=point(rng.uniform(2.2, 3.1), rng.uniform(-.3, .3)), start_yaw=yaw,
                bounds=(-4., 4., -4., 4.), obstacles=obstacles,
                friction=float(rng.uniform(.65, 1.05)))
            spec.validate()
            cases.append(Situation(family, variant, scene_seed, bool(variant % 2), spec))
    return cases
