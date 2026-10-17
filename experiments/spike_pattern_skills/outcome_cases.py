"""Predeclared contexts for expanded association learning, not outcome-derived scenes."""
from dataclasses import dataclass, replace

import numpy as np

from hybrid.arena import Obstacle, Scenario
from .curriculum import curriculum


@dataclass(frozen=True)
class LearningCase:
    family: str
    variant: int
    scene_seed: int
    split: str
    scenario: Scenario
    jump_requested: bool = False
    traversal_x: float | None = None
    duration: float = 4.
    timeout: float = 12.
    stage: int = -1


def learning_cases(terrain_seed=20261019, flat_seed=20261020):
    cases = []
    for c in curriculum(terrain_seed):
        cases.append(LearningCase(c.family, c.variant, c.scene_seed,
            'train' if c.split == 'development' else c.split,
            replace(c.scenario, name='new-'+c.scenario.name), traversal_x=c.traversal_x,
            duration=8., timeout=18., stage=c.stage))
    lengths = (1.3, 1.7, 2.2, 3., 3.6, 1.7, 1.55, 2.6, 3.3)
    headings = (0., 0., -.7, .7, np.pi, -np.pi/2, 0., -1.1, 1.1)
    for group, family in enumerate(('flat_forward', 'requested_hop', 'static_blocker', 'moving_crossing')):
        for variant in range(9):
            seed = flat_seed+group*9+variant
            rng = np.random.default_rng(seed)
            yaw = headings[variant]
            forward, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            start = rng.uniform(-.15, .15, 2)
            goal = start+forward*lengths[variant]
            obstacles = ()
            if family == 'static_blocker':
                center = start+forward*rng.uniform(1.2, 2.8)+left*rng.uniform(-.15, .15)
                obstacles = (Obstacle('blocker', tuple(center), radius=float(rng.uniform(.17, .27))),)
            elif family == 'moving_crossing':
                center = start+forward*rng.uniform(1.4, 2.5)-left*1.4
                obstacles = (Obstacle('crossing', tuple(center), radius=.2,
                                      velocity=tuple(left*rng.uniform(.13, .3)), stop_after=16.),)
            spec = Scenario(f'new-{family}-{variant:02d}', start=tuple(start), goal=tuple(goal),
                start_yaw=yaw, obstacles=obstacles, bounds=(-6., 6., -6., 6.),
                friction=float(rng.uniform(.7, .95)))
            spec.validate()
            cases.append(LearningCase(family, variant, seed,
                'train' if variant < 6 else 'validation' if variant == 6 else 'held_out',
                spec, jump_requested=family == 'requested_hop'))
    return cases
