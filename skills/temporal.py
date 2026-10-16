"""Optional classical retreat/hold options; no new learned-policy outputs.

The supervisor supplies current all-obstacle route clearance to ``update`` and
owns physics, dynamic clearance stops and stable recovery. These objects never
write robot state, motor commands, or simulator time.
"""
from dataclasses import dataclass

import numpy as np

from skills.avoidance import Skill, clearance, path_clearance, segment_distance, ROBOT_RADIUS

TEMPORAL_SKILL_NAMES = ('yield_hold', 'retreat_hold')
CATALOG_VERSION = 2


@dataclass
class TemporalSkill(Skill):
    hold_duration: float = 1.5
    holding: bool = False
    status: str = 'running'
    failure_reason: str | None = None
    clear_since: float | None = None
    last_update: float | None = None

    def target(self, position):
        return None if self.holding or self.status != 'running' else super().target(position)

    def update(self, position, now, route_clear):
        """Advance lifecycle from a fresh sensing tick (metres, seconds, bool).

        ``route_clear`` must include every observed obstacle, not only the one
        which triggered selection. A new threat resets the entire dwell.
        """
        position = np.asarray(position, dtype=float)
        if (position.shape != (2,) or not np.all(np.isfinite(position))
                or not np.isfinite(now) or now < self.started
                or (self.last_update is not None and now < self.last_update)
                or not isinstance(route_clear, (bool, np.bool_))):
            raise ValueError('Require finite position, monotonic time and boolean route_clear')
        if self.status != 'running':
            return self.status
        self.last_update = float(now)
        if self.expired(now):
            self.status, self.failure_reason = 'timeout', 'skill timeout'
            return self.status
        if not self.holding and np.linalg.norm(position-self.points[-1]) < .14:
            self.holding = True
        if self.holding:
            self.clear_since = (float(now) if self.clear_since is None else self.clear_since) if route_clear else None
            if self.clear_since is not None and now-self.clear_since >= self.hold_duration-1e-9:
                self.status = 'complete'
        return self.status

    def complete(self, position, now):
        return self.status == 'complete'

    def expired(self, now):
        return now-self.started >= self.timeout

    def interrupt(self):
        if self.status == 'running':
            self.status, self.failure_reason = 'interrupted', 'skill interrupted'
        return self.status


def build_temporal_skills(position, goal, obstacle, now, obstacles, bounds, *,
                          hold_duration=1.5, timeout=60.0, retreat_distance=.55):
    """Return yield, retreat, yield eligibility, retreat eligibility.

    ``obstacle`` is accepted for compatibility with encounter factories; all
    obstacle tuples (world centre m, radius m, world velocity m/s) are checked.
    Eligibility requires >8 cm inflated footprint clearance. Yield additionally
    requires >8 cm throughout a three-second constant-velocity forecast.
    Retreat's swept segment is checked against current geometry; the runtime
    must continue checking moving hazards. Invalid inputs raise ValueError.
    """
    position, goal, bounds = (np.asarray(x, dtype=float) for x in (position, goal, bounds))
    if (position.shape != (2,) or goal.shape != (2,) or bounds.shape != (4,)
            or not all(np.all(np.isfinite(x)) for x in (position, goal, bounds))
            or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
            or not np.all(np.isfinite([now, hold_duration, timeout, retreat_distance]))
            or min(hold_duration, timeout, retreat_distance) <= 0 or timeout <= hold_duration
            or np.linalg.norm(goal-position) < 1e-9):
        raise ValueError('Require finite geometry, nonzero goal direction and positive durations/distance')
    checked = []
    for center, radius, velocity in obstacles:
        center, velocity = np.asarray(center, dtype=float), np.asarray(velocity, dtype=float)
        if (center.shape != (2,) or velocity.shape != (2,)
                or not np.all(np.isfinite(center)) or not np.all(np.isfinite(velocity))
                or not np.isfinite(radius) or radius <= 0):
            raise ValueError('Obstacle must have finite 2D centre/velocity and positive radius')
        checked.append((center, radius, velocity))
    forward = (goal-position)/np.linalg.norm(goal-position)
    point = position-forward*retreat_distance
    yield_skill = TemporalSkill('yield_hold', [], float(now), timeout,
                                hold_duration=hold_duration, holding=True)
    retreat_skill = TemporalSkill('retreat_hold', [point.copy()], float(now), timeout,
                                  hold_duration=hold_duration)
    current_gap = clearance(position, checked, bounds)
    wait_gap = min([clearance(position, [], bounds)] + [
        segment_distance(position, center, center+3*velocity)-radius-ROBOT_RADIUS
        for center, radius, velocity in checked])
    return (yield_skill, retreat_skill, bool(current_gap > .08 and wait_gap > .08),
            bool(current_gap > .08 and path_clearance(position, [point], checked, bounds) > .08))


def build_temporal_skill(name, position, goal, obstacles, bounds, now, **parameters):
    """Single-option factory which rejects ineligible initiation explicitly."""
    if name not in TEMPORAL_SKILL_NAMES:
        raise ValueError(f'Unknown temporal skill: {name}')
    hold, retreat, hold_allowed, retreat_allowed = build_temporal_skills(
        position, goal, None, now, obstacles, bounds, **parameters)
    skill, allowed = (hold, hold_allowed) if name == 'yield_hold' else (retreat, retreat_allowed)
    if not allowed:
        raise ValueError(f'{name}: initiation clearance failed')
    return skill
