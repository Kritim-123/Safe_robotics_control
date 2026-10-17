"""Conservative robot-disc clearance to actual oriented obstacle rectangles."""
import numpy as np
from skills.avoidance import clearance as circle_clearance

ROBOT_RADIUS=.42


def point_box_distance(point,center,half_extents,yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    local=np.array([[c,s],[-s,c]])@(np.asarray(point)-center)
    return float(np.linalg.norm(np.maximum(np.abs(local)-half_extents,0.)))


def scene_clearance(position,obstacles,bounds,specifications):
    if len(obstacles)!=len(specifications):raise ValueError('Obstacle geometry mismatch')
    gap=circle_clearance(position,[],bounds)
    for (center,radius,velocity),spec in zip(obstacles,specifications):
        distance=(point_box_distance(position,center,np.asarray(spec.half_extents),spec.yaw)
                  if spec.shape=='box' else np.linalg.norm(np.asarray(position)-center)-radius)
        gap=min(gap,float(distance)-ROBOT_RADIUS)
    return float(gap)


def shape_request_masks(spec,observed,*,run_speed=.35,predict_horizon=1.6,jump_distance=.5,jump_horizon=3.,goal=None):
    pos=np.asarray(observed['position']);yaw=observed['yaw']
    forward=np.array([np.cos(yaw),np.sin(yaw)])
    directions=[forward]
    if goal is not None:
        left=np.array([-forward[1],forward[0]]);error=np.asarray(goal)-pos
        local=np.array([max(0.,error@forward),error@left])
        world=forward*local[0]+left*local[1]
        if np.linalg.norm(world)>1e-9:directions.append(world/np.linalg.norm(world))
    tuples=[(np.asarray(o['center']),o['radius'],np.asarray(o['velocity'])) for o in observed['obstacles']]
    gap=scene_clearance(pos,tuples,spec.bounds,spec.obstacles)
    run_gap=jump_gap=float('inf')
    for dt in np.linspace(0,predict_horizon,9):
        predicted=[(c+v*dt,r,v) for c,r,v in tuples]
        for direction in directions:
            run_gap=min(run_gap,scene_clearance(pos+direction*run_speed*dt,predicted,spec.bounds,spec.obstacles))
    for dt in np.linspace(0,jump_horizon,17):
        predicted=[(c+v*dt,r,v) for c,r,v in tuples]
        # Disc clearance is 1-Lipschitz in position. Subtract half the sample
        # spacing so the complete straight segment is conservatively covered.
        spacing=jump_distance/16
        sampled=min(scene_clearance(pos+forward*x,predicted,spec.bounds,spec.obstacles)
                    for x in np.linspace(0,jump_distance,17))
        jump_gap=min(jump_gap,sampled-spacing/2)
    return dict(wait=bool(gap>.08),run=bool(run_gap>.15),jump=bool(jump_gap>.35)),dict(
        current_clearance=gap,predicted_run_clearance=float(run_gap),predicted_jump_clearance=float(jump_gap),
        prediction_horizon=predict_horizon,jump_prediction_horizon=jump_horizon,geometry='robot_disc_vs_oriented_boxes',
        predicted_run_directions=[d.tolist() for d in directions])
