import unittest
import numpy as np
from hybrid.arena import Obstacle,Scenario
from experiments.terrain_goal.geometry import point_box_distance,scene_clearance,shape_request_masks
from skills.avoidance import clearance


class ShapeClearanceTests(unittest.TestCase):
    def test_oriented_box_distance_and_robot_radius(self):
        self.assertAlmostEqual(point_box_distance([0.,2.],[0.,0.],np.array([1.,.5]),0.),1.5)
        self.assertAlmostEqual(point_box_distance([0.,2.],[0.,0.],np.array([1.,.5]),np.pi/2),1.)
        self.assertEqual(point_box_distance([0.,0.],[0.,0.],np.array([1.,.5]),.7),0.)
        o=Obstacle('box',(0.,0.),shape='box',half_extents=(1.,.5))
        self.assertAlmostEqual(scene_clearance([0.,2.],[(np.zeros(2),o.planning_radius,np.zeros(2))],(-5,5,-5,5),[o]),1.08)

    def test_cylinder_and_bounds_match_original_guard(self):
        o=Obstacle('cylinder',(1.,0.));items=[(np.array(o.center),o.radius,np.zeros(2))]
        for point in ([0.,0.],[.8,0.],[-4.4,.2]):
            self.assertAlmostEqual(scene_clearance(point,items,(-5,5,-5,5),[o]),clearance(np.asarray(point),items,(-5,5,-5,5)))
        with self.assertRaises(ValueError):scene_clearance([0,0],[],(-5,5,-5,5),[o])

    def test_passage_is_conservative_without_circumscribed_circle_rejection(self):
        obstacles=tuple(Obstacle(f'side{i}',(.9,side*.9),shape='box',half_extents=(.4,.12)) for i,side in enumerate((-1,1)))
        spec=Scenario('passage',obstacles=obstacles,bounds=(-3,6,-3,3))
        observed=dict(position=[0.,0.],yaw=0.,obstacles=[dict(center=list(o.center),radius=o.planning_radius,velocity=[0.,0.]) for o in obstacles])
        masks,g=shape_request_masks(spec,observed)
        self.assertTrue(masks['run']);self.assertGreater(g['predicted_run_clearance'],.15)
        # Every sampled point on the center line leaves >=0.36 m beyond the
        # conservative 0.42 m robot disc; physical guard margins are preserved.
        for x in np.linspace(0,1.8,101):
            tuples=[(np.asarray(o.center),o.planning_radius,np.zeros(2)) for o in obstacles]
            self.assertGreaterEqual(scene_clearance([x,0.],tuples,spec.bounds,obstacles),.36-1e-12)


if __name__=='__main__':unittest.main()
