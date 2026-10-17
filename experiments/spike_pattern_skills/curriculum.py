"""Ten ordered environment levels; six deterministic variants per level."""
from dataclasses import dataclass

import numpy as np

from hybrid.arena import Obstacle, Scenario
from hybrid.terrain import TerrainPatch

STAGES = ('flat', 'thresholds', 'ramps', 'steps', 'uneven_blocks', 'cross_slope',
          'friction_changes', 'static_passage', 'moving_crossing', 'mixed_course')


@dataclass(frozen=True)
class CurriculumCase:
    stage: int
    family: str
    variant: int
    scene_seed: int
    split: str
    scenario: Scenario
    traversal_x: float = 1.55
    jump_requested: bool = False


def curriculum(seed=20261017):
    if type(seed) is not int or seed < 0 or seed >= 2**63-100:
        raise ValueError('Use an integer curriculum seed in [0,2**63-100)')
    result = []
    for stage, family in enumerate(STAGES):
        for variant in range(6):
            scene_seed = seed+stage*6+variant
            rng = np.random.default_rng(scene_seed)
            difficulty = variant/5
            offset = float(rng.uniform(-.025, .025))
            patches, obstacles = [], []
            def patch(name, x, hx, height, *, hy=.85, y=0., kind='box', yaw=0., friction=.8):
                patches.append(TerrainPatch(name, (x+offset, y), (hx, hy), float(height),
                                            kind=kind, yaw=yaw, friction=friction))
            if family == 'thresholds':
                patch('threshold', .72, .055, .008+.012*difficulty)
            elif family == 'ramps':
                rise = .025+.04*difficulty
                patch('ramp_up', .65, .25, rise, kind='ramp')
                patch('ramp_down', 1.15, .25, rise, kind='ramp', yaw=np.pi)
            elif family == 'steps':
                rise = .012+.018*difficulty
                for i, multiplier in enumerate((1, 2, 1)):
                    patch('step_'+str(i), .55+i*.3, .15, rise*multiplier)
            elif family == 'uneven_blocks':
                for i in range(5):
                    for side in (-1, 1):
                        patch(f'block_{i}_{"left" if side > 0 else "right"}', .48+i*.18, .088,
                            rng.uniform(.005, .012+.023*difficulty), hy=.3, y=side*.30)
            elif family == 'cross_slope':
                patch('cross_slope', .88, .85, .055+.09*difficulty, hy=.48, kind='ramp', yaw=np.pi/2)
            elif family == 'friction_changes':
                patch('reduced_friction', .82, .4, .008, friction=.6-.25*difficulty)
            elif family in ('static_passage', 'moving_crossing'):
                half_width = .95-.12*difficulty
                obstacles.extend(Obstacle('side_'+str(i), (.9+offset, side*half_width), radius=.16,
                    shape='box', half_extents=(.4, .12), height=.5) for i, side in enumerate((-1, 1)))
                if family == 'moving_crossing':
                    speed = .12+.06*difficulty
                    obstacles.append(Obstacle('crossing', (1.1+offset, -1.2), radius=.17,
                        velocity=(0., speed), start_after=2., stop_after=16.))
            elif family == 'mixed_course':
                rise = .02+.025*difficulty
                patch('ramp_up', .6, .2, rise, kind='ramp')
                patch('ramp_down', 1., .2, rise, kind='ramp', yaw=np.pi)
                patch('finish_threshold', 1.31, .05, .01+.01*difficulty, friction=.55)
                obstacles.append(Obstacle('crossing', (1.1+offset, -1.3), radius=.16,
                    velocity=(0., .13+.04*difficulty), start_after=2., stop_after=18.))
            spec = Scenario(f'terrain-{stage:02d}-{family}-{variant:02d}', goal=(1.7, 0),
                bounds=(-1.5, 6., -2.4, 2.4), timeout=18., terrain=tuple(patches),
                obstacles=tuple(obstacles))
            spec.validate()
            result.append(CurriculumCase(stage, family, variant, scene_seed,
                'development' if variant < 3 else 'validation' if variant == 3 else 'held_out', spec))
    return result
