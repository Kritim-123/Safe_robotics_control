from dataclasses import replace
import unittest
import numpy as np
from experiments.terrain_goal.contract import AdaptationConfig,GoalTracker
from experiments.terrain_goal.encoder import TaskEncoder,add_task
from experiments.continual_snn.predictive import PredictivePacketEncoder


class TerrainGoalContractTests(unittest.TestCase):
    def test_arrival_hysteresis_and_jump_preservation(self):
        t=GoalTracker(AdaptationConfig(goal_arrival=True))
        def ask(distance,task='progress'):return t.request(task,{'position':[distance,0.]},[0.,0.])
        self.assertEqual(ask(.5),'progress');self.assertEqual(ask(.07),'hold')
        self.assertEqual(ask(.13),'hold');self.assertEqual(ask(.18),'progress')
        self.assertEqual(ask(.01,'jump'),'jump');self.assertEqual(ask(.9,'hold'),'hold')
        with self.assertRaises(ValueError):replace(AdaptationConfig(),stop_radius=.2).validate()

    def test_task_population_preserves_all_old_neurons(self):
        old=PredictivePacketEncoder();f={n:0. for n in old.feature_names}
        f.update(goal_forward=.5,predicted_run_clearance=1.,predicted_jump_clearance=1.)
        new=TaskEncoder('task96')
        a=new.encode(add_task(f,'progress'));b=new.encode(add_task(f,'hold'))
        np.testing.assert_array_equal(a['raster'][:,:80],old.encode(f)['raster'])
        self.assertNotEqual(a['spike_counts'][80:],b['spike_counts'][80:])
        self.assertEqual(a['spike_counts'],new.encode(add_task(f,'progress'))['spike_counts'])

    def test_factor_population_is_invariant_to_irrelevant_goal_and_terrain(self):
        old=PredictivePacketEncoder();f={n:0. for n in old.feature_names}
        f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
        e=TaskEncoder('factor32');a=add_task(f,'progress')
        b=dict(a,goal_forward=.01,surface_rise=.3,base_height=.4)
        self.assertEqual(e.encode(a)['spike_counts'],e.encode(b)['spike_counts'])
        with self.assertRaises(ValueError):e.encode(dict(a,task_progress=float('nan')))

    def test_goal_population_preserves_task_counts_and_distinguishes_arrival(self):
        f={n:0. for n in PredictivePacketEncoder().feature_names}
        f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.,
            goal_mode=1.,goal_forward=1.,speed=.15,run_active=1.,braking_active=0.)
        f=add_task(f,'progress');e=TaskEncoder('goal64')
        a=e.encode(f);b=e.encode(dict(f,goal_forward=.04))
        np.testing.assert_array_equal(a['spike_counts'][:32],TaskEncoder('factor32').encode(f)['spike_counts'])
        self.assertNotEqual(a['spike_counts'][32:],b['spike_counts'][32:])
        self.assertEqual(e.encode(dict(f,goal_mode=0.))['spike_counts'],
                         e.encode(dict(f,goal_mode=0.,goal_forward=.001))['spike_counts'])

    def test_monotonic_proximity_population_covers_unseen_nearer_goals(self):
        f={n:0. for n in PredictivePacketEncoder().feature_names}
        f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.,goal_mode=1.)
        f=add_task(f,'progress');e=TaskEncoder('goal_lattice64')
        counts=np.asarray([e.encode(dict(f,goal_forward=d))['spike_counts'][32:] for d in np.linspace(0.,.5,101)])
        self.assertTrue(np.all(np.diff(counts,axis=0)<=0))
        self.assertTrue(np.any(counts[0]>counts[-1]))


if __name__=='__main__':unittest.main()
