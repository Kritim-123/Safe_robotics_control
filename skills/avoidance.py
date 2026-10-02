"""Closed-loop options used by both the rule and spiking skill selectors.

These are classical movement skills, not downloaded locomotion networks.
The SNN selects an option; joint PID continues throughout its execution.
"""
from dataclasses import dataclass
import numpy as np

SKILL_NAMES = ('bypass_left', 'bypass_right', 'wait', 'back_off')
ROBOT_RADIUS = 0.42


def segment_distance(point, start, end):
    vector = np.asarray(end) - start
    t = np.clip(np.dot(np.asarray(point) - start, vector) / max(np.dot(vector, vector), 1e-12), 0, 1)
    return float(np.linalg.norm(np.asarray(point) - (start + t * vector)))


def clearance(point, obstacles, bounds, radius=ROBOT_RADIUS):
    x0, x1, y0, y1 = bounds
    x, y = point
    gaps = [x-x0-radius, x1-x-radius, y-y0-radius, y1-y-radius]
    gaps += [np.linalg.norm(point - center) - obstacle_radius - radius
             for center, obstacle_radius, _ in obstacles]
    return float(min(gaps))


def path_clearance(start, points, obstacles, bounds):
    minimum = float('inf')
    for end in points:
        # Cylinder distances are exact for the conservative circular footprint.
        minimum = min(minimum, clearance(start, [], bounds), clearance(end, [], bounds))
        for center, radius, _ in obstacles:
            minimum = min(minimum, segment_distance(center, np.asarray(start), np.asarray(end)) - radius - ROBOT_RADIUS)
        start = end
    return float(minimum)


@dataclass
class Skill:
    name: str
    points: list
    started: float
    timeout: float
    index: int = 0

    def target(self, position):
        while self.index < len(self.points) - 1 and np.linalg.norm(position - self.points[self.index]) < 0.14:
            self.index += 1
        return self.points[self.index] if self.points else None

    def complete(self, position, now):
        if self.name == 'wait':
            return False  # Supervisor requires persistent clear route.
        return bool(self.points and self.index == len(self.points)-1
                    and np.linalg.norm(position - self.points[-1]) < 0.14)

    def expired(self, now):
        return now - self.started > self.timeout


def build_skills(position, goal, obstacle, now):
    forward = np.asarray(goal) - position
    forward /= max(np.linalg.norm(forward), 1e-12)
    left = np.array([-forward[1], forward[0]])
    center, radius, _ = obstacle
    along = float(np.dot(center-position, forward))
    # Room for the swept feet and a small tracking error, beyond collision radius.
    offset = radius + ROBOT_RADIUS + 0.28
    end_along = max(along + radius + ROBOT_RADIUS + 0.3, 0.8)
    skills = []
    for sign, name in [(1, 'bypass_left'), (-1, 'bypass_right')]:
        lateral = np.dot(center-position, left) + sign * offset
        points = [position + left*lateral,
                  position + forward*end_along + left*lateral]
        skills.append(Skill(name, points, now, 100.0))
    skills.append(Skill('wait', [], now, 25.0))
    skills.append(Skill('back_off', [position - forward*0.55], now, 30.0))
    return skills
