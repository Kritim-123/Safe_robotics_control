"""Geometric observation contract shared by data generation and simulation.

This first experiment assumes obstacle position/radius/velocity observations
are provided by the simulator. It does not infer them from images or scans.
"""
from dataclasses import dataclass
import numpy as np

from skills.avoidance import ROBOT_RADIUS, build_skills, clearance, path_clearance, segment_distance

FEATURE_NAMES = ('obstacle_forward', 'obstacle_left', 'obstacle_v_forward',
                 'obstacle_v_left', 'obstacle_radius', 'current_clearance',
                 'left_path_clearance', 'right_path_clearance', 'back_path_clearance',
                 'predicted_wait_clearance', 'goal_distance', 'heading_error')


@dataclass
class Encounter:
    features: np.ndarray
    allowed: np.ndarray
    skills: list
    obstacle_index: int


def threatening_obstacle(position, goal, obstacles, horizon=3.0, lookahead=1.15):
    delta = np.asarray(goal)-position
    distance = np.linalg.norm(delta)
    if distance < 0.12:
        return None
    end = position + delta / distance * min(distance, lookahead)
    if not obstacles:
        return None
    centers = np.asarray([o[0] for o in obstacles])
    radii = np.asarray([o[1] for o in obstacles])
    velocities = np.asarray([o[2] for o in obstacles])
    # Same seven prediction samples, batched over obstacles and times.
    samples = centers[:, None, :] + velocities[:, None, :]*np.linspace(0, horizon, 7)[None, :, None]
    vector = end-position
    projected = np.clip(np.sum((samples-position)*vector, axis=-1)/max(np.dot(vector, vector), 1e-12), 0, 1)
    gaps = np.linalg.norm(samples-(position+projected[..., None]*vector), axis=-1).min(axis=1)-radii-ROBOT_RADIUS-.10
    candidates = [(np.linalg.norm(centers[i]-position), int(i)) for i in np.flatnonzero(gaps < 0)]
    return min(candidates)[1] if candidates else None


def observe_encounter(position, yaw, goal, obstacles, bounds, obstacle_index, now=0.0):
    position, goal = np.asarray(position, dtype=float), np.asarray(goal, dtype=float)
    delta = goal-position
    distance = float(np.linalg.norm(delta))
    forward = delta / max(distance, 1e-12)
    left = np.array([-forward[1], forward[0]])
    center, radius, velocity = obstacles[obstacle_index]
    relative = center-position
    skills = build_skills(position, goal, obstacles[obstacle_index], now)
    left_gap, right_gap, back_gap = [path_clearance(position, skills[i].points, obstacles, bounds) for i in (0, 1, 3)]
    centers = np.asarray([o[0] for o in obstacles])
    radii = np.asarray([o[1] for o in obstacles])
    velocities = np.asarray([o[2] for o in obstacles])
    future = centers[:, None, :] + velocities[:, None, :]*np.linspace(0, 3.0, 7)[None, :, None]
    wait_gap = min(clearance(position, [], bounds),
                   float(np.min(np.linalg.norm(future-position, axis=-1)-radii[:, None]-ROBOT_RADIUS)))
    heading = np.arctan2(delta[1], delta[0]) - yaw
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    features = np.array([np.dot(relative, forward), np.dot(relative, left),
                         np.dot(velocity, forward), np.dot(velocity, left), radius,
                         clearance(position, obstacles, bounds), left_gap, right_gap,
                         back_gap, wait_gap, distance, heading], dtype=np.float32)
    allowed = np.array([left_gap > 0.045, right_gap > 0.045, wait_gap > 0.07, back_gap > 0.025])
    return Encounter(features, allowed, skills, obstacle_index)


def teacher_scores(features):
    """Heuristic demonstration teacher; these scores are not learned rewards."""
    f = np.asarray(features)
    x, y, vx, vy, radius, gap, left, right, back, wait, distance, heading = np.moveaxis(f, -1, 0)
    left_score = 1.2 + np.minimum(left, 0.8)*2 - 0.35*y - 3.0*vy
    right_score = 1.2 + np.minimum(right, 0.8)*2 + 0.35*y + 3.0*vy
    # Waiting is efficient for a crossing obstacle, not a permanent blocker.
    crossing = (np.abs(vy) > 0.035) & (np.abs(vx) < 0.045)
    wait_score = np.where(crossing & (wait > 0.13), 3.4, -0.8)
    back_score = np.where((gap < 0.18) | (vx < -0.04), 3.8, -0.4) + np.minimum(back, 0.5)
    return np.stack([left_score, right_score, wait_score, back_score], axis=-1)


def choose_masked(scores, allowed):
    scores, allowed = np.asarray(scores), np.asarray(allowed, dtype=bool)
    if scores.shape != (4,) or allowed.shape != (4,) or not np.all(np.isfinite(scores)):
        raise ValueError('Selector must return four finite skill scores and four mask values')
    if not np.any(allowed):
        return None
    return int(np.argmax(np.where(allowed, scores, -np.inf)))
