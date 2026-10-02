"""Small reproducible obstacle encounters, with explicitly privileged sensing."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Obstacle:
    name: str
    center: tuple[float, float]
    radius: float = 0.23
    velocity: tuple[float, float] = (0.0, 0.0)
    stop_after: float = 40.0

    def at(self, time):
        return np.asarray(self.center) + np.asarray(self.velocity) * min(max(time, 0), self.stop_after)

    def speed_at(self, time):
        return np.asarray(self.velocity) if time < self.stop_after else np.zeros(2)


@dataclass(frozen=True)
class Scenario:
    name: str
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (3.0, 0.0)
    obstacles: tuple[Obstacle, ...] = ()
    bounds: tuple[float, float, float, float] = (-1.5, 4.5, -2.0, 2.0)
    start_yaw: float = 0.0
    timeout: float = 150.0
    friction: float = 0.8
    push_time: float = -1.0
    push_force: tuple[float, float] = (0.0, 0.0)
    push_duration: float = 0.2
    position_noise: float = 0.0
    velocity_noise: float = 0.0
    sensor_delay: float = 0.0
    sensor_filter_tau: float = 0.0
    sensing_seed: int = 0

    def validate(self):
        if (np.shape(self.start) != (2,) or np.shape(self.goal) != (2,)
                or np.shape(self.bounds) != (4,) or np.shape(self.push_force) != (2,)):
            raise ValueError('Start, goal and force must be 2D; bounds must have four values')
        values = [*self.start, *self.goal, *self.bounds, self.start_yaw,
                  self.timeout, self.friction, self.push_time, *self.push_force, self.push_duration,
                  self.position_noise, self.velocity_noise, self.sensor_delay, self.sensor_filter_tau]
        if not np.all(np.isfinite(values)) or self.timeout <= 0 or self.friction <= 0:
            raise ValueError('Scenario values must be finite and timeout/friction positive')
        x0, x1, y0, y1 = self.bounds
        if x0 >= x1 or y0 >= y1 or self.push_duration <= 0:
            raise ValueError('Invalid arena bounds or push duration')
        if min(self.position_noise, self.velocity_noise, self.sensor_delay, self.sensor_filter_tau) < 0:
            raise ValueError('Sensor noise and delay must be nonnegative')
        if not isinstance(self.sensing_seed, int) or self.sensing_seed < 0:
            raise ValueError('Sensing seed must be a nonnegative integer')
        for label, point in [('start', self.start), ('goal', self.goal)]:
            if not x0 + .42 < point[0] < x1 - .42 or not y0 + .42 < point[1] < y1 - .42:
                raise ValueError(f'{label} must leave room for the robot inside the arena')
        if len(set(o.name for o in self.obstacles)) != len(self.obstacles):
            raise ValueError('Obstacle names must be unique')
        for o in self.obstacles:
            if np.shape(o.center) != (2,) or np.shape(o.velocity) != (2,):
                raise ValueError('Obstacle center and velocity must be 2D')
            if not np.all(np.isfinite([*o.center, o.radius, *o.velocity, o.stop_after])) or o.radius <= 0 or o.stop_after < 0:
                raise ValueError('Invalid obstacle geometry or motion')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        """Load an editable JSON scenario, including files saved with each run."""
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        values['obstacles'] = tuple(Obstacle(**o) for o in values.get('obstacles', []))
        result = cls(**values)
        result.validate()
        return result


def scenario(name='crossing', seed=0):
    """Seed varies geometry/motion, never a pre-recorded action sequence."""
    rng = np.random.default_rng(seed)
    x = 1.35 + rng.uniform(-0.12, 0.12)
    radius = rng.uniform(0.20, 0.27)
    if name == 'clear':
        return Scenario(name)
    if name == 'static':
        return Scenario(name, obstacles=(Obstacle('blocker', (x, rng.uniform(-0.12, 0.12)), radius),))
    if name in ('crossing', 'crossing_reverse'):
        direction = 1 if name == 'crossing' else -1
        speed = rng.uniform(0.07, 0.10) * direction
        return Scenario(name, obstacles=(Obstacle('crossing', (x, -1.3 * direction), radius,
                        (0, speed), 2.6 / abs(speed)),))
    if name in ('left_blocked', 'right_blocked'):
        side = 1 if name == 'left_blocked' else -1
        return Scenario(name, obstacles=(Obstacle('blocker', (x, 0), radius),
                        Obstacle('side_blocker', (x - 0.2, side * 1.1), 0.37)))
    if name == 'narrow_crossing':
        speed = rng.uniform(0.07, 0.10)
        return Scenario(name, obstacles=(Obstacle('crossing', (x, -1.0), radius, (0, speed), 2.0 / speed),),
                        bounds=(-1.5, 4.5, -1.35, 1.35))
    if name == 'push':
        return Scenario(name, obstacles=(Obstacle('blocker', (x, 0), radius),),
                        push_time=28.0, push_force=(0.0, 25.0))
    if name == 'head_on':
        # Approaches the robot then stops; exercises back-off/replanning limits.
        return Scenario(name, obstacles=(Obstacle('approaching', (2.1, 0), radius,
                        (-.06, 0), 15.0),))
    if name == 'blocked':
        return Scenario(name, obstacles=(Obstacle('blocker', (x, 0), .30),),
                        bounds=(-1.5, 4.5, -.8, .8), timeout=90.0)
    if name == 'low_friction':
        return Scenario(name, obstacles=(Obstacle('blocker', (x, 0), radius),), friction=.25)
    if name == 'angled':
        return Scenario(name, goal=(2.7, .8), start_yaw=-.6,
                        obstacles=(Obstacle('blocker', (x, .4), radius),), timeout=180.0)
    if name == 'rotated_static':
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        lateral = np.array([-direction[1], direction[0]])
        center = direction*x + lateral*rng.uniform(-.12, .12)
        return Scenario(name, goal=tuple(3*direction), start_yaw=float(rng.uniform(-np.pi, np.pi)),
                        obstacles=(Obstacle('blocker', tuple(center), radius),),
                        bounds=(-4.5, 4.5, -4.5, 4.5), timeout=180.0)
    if name == 'late_crossing':
        # The second obstacle moves into an initially clear bypass after the
        # first skill has been selected. This exposes the lack of replanning.
        return Scenario(name, obstacles=(Obstacle('blocker', (x, 0), radius),
                        Obstacle('late_crossing', (.9, 1.9), .15, (0, -.045), 60.0)))
    raise ValueError(f'Unknown scenario: {name}')


SCENARIOS = ('clear', 'static', 'crossing', 'crossing_reverse', 'left_blocked',
             'right_blocked', 'narrow_crossing', 'push')
STRESS_SCENARIOS = ('head_on', 'blocked', 'low_friction', 'angled', 'rotated_static', 'late_crossing')


def build_model(spec):
    spec.validate()
    # Expand the supplied robot XML so file loading is independent of cwd.
    root = ET.parse(ROOT / 'unitree_go2' / 'go2_mjx.xml').getroot()
    root.set('model', f'Go2 hybrid {spec.name}')
    root.find('compiler').set('meshdir', str(ROOT / 'unitree_go2' / 'assets'))
    # The Menagerie foot geoms have priority=1, so their contact parameters
    # override the floor's priority=0. Change that default as well, preserving
    # its contact dimensions and softness while making friction experiments real.
    root.find(".//default[@class='foot']/geom").set('friction', f'{spec.friction} 0.02 0.01')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='640')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', pos='1 -1 5', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='arena_floor', type='plane', size='8 5 0.1',
                  rgba='0.78 0.80 0.82 1', friction=f'{spec.friction} 0.02 0.01', condim='3')
    x0, x1, y0, y1 = spec.bounds
    walls = [('west', (x0 - .08, (y0+y1)/2), (.08, (y1-y0)/2+.16)),
             ('east', (x1 + .08, (y0+y1)/2), (.08, (y1-y0)/2+.16)),
             ('south', ((x0+x1)/2, y0-.08), ((x1-x0)/2, .08)),
             ('north', ((x0+x1)/2, y1+.08), ((x1-x0)/2, .08))]
    for name, pos, size in walls:
        ET.SubElement(world, 'geom', name=f'wall_{name}', type='box',
                      pos=f'{pos[0]} {pos[1]} .25', size=f'{size[0]} {size[1]} .25',
                      rgba='.38 .44 .52 1', friction='0.8 .02 .01')
    for i, obstacle in enumerate(spec.obstacles):
        body = ET.SubElement(world, 'body', name=f'obstacle_{i}', mocap='true',
                             pos=f'{obstacle.center[0]} {obstacle.center[1]} .4')
        ET.SubElement(body, 'geom', name=f'obstacle_geom_{i}', type='cylinder',
                      size=f'{obstacle.radius} .4', rgba='.88 .29 .20 1', friction='.8 .02 .01')
    for name, point, color in [('task_start', spec.start, '.15 .7 .4 .8'),
                                ('task_goal', spec.goal, '1 .72 .1 .9')]:
        ET.SubElement(world, 'site', name=name, type='cylinder', pos=f'{point[0]} {point[1]} .006',
                      size='.14 .006', rgba=color)
    ET.SubElement(world, 'site', name='skill_target', type='sphere', pos='0 0 -1',
                  size='.065', rgba='.15 .45 1 .9')
    # Mocap defaults added to the keyframe are compiled from body positions.
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class Arena:
    def __init__(self, spec, model):
        self.spec, self.model = spec, model
        self.mocap_ids = [int(model.body(f'obstacle_{i}').mocapid[0]) for i in range(len(spec.obstacles))]
        self.hazard_geoms = {model.geom(f'obstacle_geom_{i}').id for i in range(len(spec.obstacles))}
        self.hazard_geoms.update(model.geom(f'wall_{side}').id for side in ('west', 'east', 'north', 'south'))
        base_id = model.body('base').id
        robot_bodies = {base_id}
        for i in range(base_id + 1, model.nbody):
            if model.body_parentid[i] in robot_bodies:
                robot_bodies.add(i)
        self.robot_geoms = set(np.flatnonzero(np.isin(model.geom_bodyid, list(robot_bodies))))

    def update(self, data):
        for i, obstacle in enumerate(self.spec.obstacles):
            data.mocap_pos[self.mocap_ids[i], :2] = obstacle.at(data.time)

    def sensed_obstacles(self, data):
        """Ground-truth positions and scripted velocities; not camera/LiDAR perception."""
        return [(data.mocap_pos[mid, :2].copy(), obstacle.radius,
                 obstacle.speed_at(data.time).copy())
                for mid, obstacle in zip(self.mocap_ids, self.spec.obstacles)]

    def collided(self, data):
        return any((c.geom1 in self.hazard_geoms and c.geom2 in self.robot_geoms)
                   or (c.geom2 in self.hazard_geoms and c.geom1 in self.robot_geoms)
                   for c in data.contact)
