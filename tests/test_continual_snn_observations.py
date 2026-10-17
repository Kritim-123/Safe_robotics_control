from dataclasses import replace
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.continual_snn.observations import SensorStream,SensorConfig,feature_packet,request_masks
from experiments.continual_snn.scenes import make_case
from hybrid.arena import Scenario,Obstacle
from hybrid.terrain import TerrainPatch


class CausalObservationTests(unittest.TestCase):
    def test_supplied_observation_not_future_script_drives_features_and_masks(self):
        case=make_case('late_crossing',4000)
        observed=dict(position=[0.,0.],yaw=0.,time=5.,speed=.1,height=.26,
            obstacles=[dict(name='seen',center=[1.,.7],radius=.18,velocity=[0.,.3],height=.8,width=.36)])
        changed=replace(case.scenario,obstacles=(Obstacle('different_future',(4.,2.),velocity=(0.,-2.),
            start_after=1.,stop_after=.5),))
        a=feature_packet(case.scenario,observed,'progress')
        b=feature_packet(changed,observed,'progress')
        self.assertEqual(a,b)
        self.assertEqual(request_masks(case.scenario,observed),request_masks(changed,observed))
        observed2=dict(observed,obstacles=[dict(observed['obstacles'][0],center=[1.,0.])])
        self.assertNotEqual(feature_packet(changed,observed2,'progress'),a)

    def test_original_22_features_reproduced_for_archived_current_state(self):
        path=Path('docs/agents/expanded_association_v2/reencoded_labels.jsonl')
        for raw in path.read_text().splitlines()[::11]:
            row=json.loads(raw);item=row['dense_observation'];values=item['metadata']['scenario'].copy()
            values['obstacles']=tuple(Obstacle(**o) for o in values['obstacles'])
            values['terrain']=tuple(TerrainPatch(**p) for p in values.get('terrain',()))
            spec=Scenario(**values)
            observed=item['metadata']['robot_observation'].copy()
            observed['obstacles']=[dict(o,height=s.height,
                width=2*(s.half_extents[1] if s.shape=='box' else s.radius))
                for o,s in zip(observed['obstacles'],spec.obstacles)]
            task='jump' if item['features']['jump_requested'] else 'progress'
            actual=feature_packet(spec,observed,task)
            self.assertEqual(set(actual),set(item['features']))
            for name in actual:self.assertAlmostEqual(actual[name],item['features'][name],places=6)

    def test_noise_delay_and_reset_are_reproducible(self):
        cfg=SensorConfig(.02,.01,.2,42)
        a,b=SensorStream(cfg),SensorStream(cfg)
        source=[dict(center=[1.,0.],velocity=[0.,.2],radius=.2,height=.8,width=.4)]
        for i in range(10):
            now=i*.1
            sa=a.observe(now,source);sb=b.observe(now,source)
            self.assertEqual(sa,sb)
            self.assertLessEqual(sa[0],now)
            if i>=2:self.assertAlmostEqual(now-sa[0],.2)
        self.assertEqual(source[0]['center'],[1.,0.])
        self.assertIsNone(SensorStream(cfg).last)
        with self.assertRaises(ValueError):a.observe(.5,source)

    def test_jump_reservation_extends_past_running_horizon(self):
        spec=Scenario('approach',bounds=(-5.,6.,-5.,5.))
        observed=dict(position=[0.,0.],yaw=0.,time=2.,speed=0.,height=.26,
            obstacles=[dict(center=[2.,0.],radius=.2,velocity=[-.3,0.],height=.8,width=.4)])
        masks,detail=request_masks(spec,observed)
        self.assertTrue(masks['run'])
        self.assertFalse(masks['jump'])
        self.assertEqual(detail['jump_prediction_horizon'],3.)

    def test_identical_nearest_obstacle_features_can_hide_a_future_threat(self):
        spec=Scenario('aliasing',bounds=(-5.,6.,-5.,5.))
        near=dict(center=[-.5,.85],radius=.18,velocity=[0.,0.],height=.8,width=.36)
        hidden=dict(center=[1.4,0.],radius=.18,velocity=[-.25,0.],height=.8,width=.36)
        a=dict(position=[0.,0.],yaw=0.,time=2.,speed=0.,height=.26,obstacles=[near])
        b=dict(a,obstacles=[near,hidden])
        self.assertEqual(feature_packet(spec,a,'progress'),feature_packet(spec,b,'progress'))
        ma,da=request_masks(spec,a);mb,db=request_masks(spec,b)
        self.assertTrue(ma['run']);self.assertFalse(mb['run'])
        self.assertNotEqual(da['predicted_run_clearance'],db['predicted_run_clearance'])


if __name__=='__main__':unittest.main()
