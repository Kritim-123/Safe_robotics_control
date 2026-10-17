import json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.scenes import make_case
from experiments.continual_snn.runner import execute as old_execute
from experiments.terrain_goal.runner import execute
from experiments.terrain_goal.contract import AdaptationConfig


class TerrainGoalRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=ROOT/'output/terrain_goal/test-tmp';cls.parent.mkdir(parents=True,exist_ok=True)

    def test_disabled_adaptation_is_physically_identical(self):
        with tempfile.TemporaryDirectory(dir=self.parent) as directory:
            root=Path(directory);case=make_case('clear_stop',12100)
            a=old_execute(case,root/'old',mode='rule');b=execute(case,root/'new',mode='rule')
            np.testing.assert_array_equal(np.load(root/'old/trace.npz')['values'],np.load(root/'new/trace.npz')['values'])
            self.assertEqual(a['task_fulfilled'],b['task_fulfilled'])

    def test_arrival_success_requires_goal_distance_and_supported_hold(self):
        from dataclasses import replace
        from experiments.continual_snn.transfer import terrain_cases
        from experiments.continual_snn.scenes import Request
        case=replace(terrain_cases(20261317)[0],required_progress=0.,
            requests=(Request(2.,32.,'progress'),Request(32.,42.,'hold')),finish_after=36.,timeout=42.)
        with tempfile.TemporaryDirectory(dir=self.parent) as directory:
            result=execute(case,Path(directory)/'goal',mode='rule',adaptation=AdaptationConfig(goal_arrival=True))
            self.assertTrue(result['task_fulfilled'],result)
            self.assertLessEqual(result['final_goal_distance'],.12)
            self.assertTrue(result['stable_recovery']);self.assertLess(result['final_linear_speed'],.04)
            self.assertLessEqual(result['max_torque_limit_ratio'],1.)

    def test_neural_stop_receives_progress_and_timed_stop_cannot_fake_arrival(self):
        from experiments.terrain_goal.encoder import TaskEncoder,add_task
        from experiments.terrain_goal.models import Selector
        from experiments.terrain_goal.factored import FactoredGoalSelector
        from experiments.terrain_goal.goal_cases import goal_cases
        encoder=TaskEncoder('factor32');rows=[]
        for group in range(4):
            for task,label in (('hold','wait'),('progress','run'),('jump','jump')):
                f={n:0. for n in encoder.feature_names}
                f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
                rows.append(dict(features=add_task(f,task),label=label,group=str(group)))
        base=Selector(encoder,'ridge').fit(rows)
        model=FactoredGoalSelector(base,92,training_groups=['training'],training_rows=2)
        case=goal_cases(13900,1,'unit',flat_only=True)[0]
        cfg=AdaptationConfig(goal_arrival=True,goal_decision_source='neural',terrain_mode='lift',terrain_lift=.12,
            gait_period=.8,gait_stance=.7,brake_rate=.5,geometry_mode='oriented_boxes')
        with tempfile.TemporaryDirectory(dir=self.parent) as directory:
            root=Path(directory)
            result=execute(case,root/'neural',model=model,adaptation=cfg)
            self.assertTrue(result['task_fulfilled'],result)
            packets=[json.loads(s) for s in (root/'neural/decisions.jsonl').read_text().splitlines()]
            stops=[r for r in packets if r['proposal']['reason']=='learned_goal_count_hold_association']
            self.assertTrue(stops)
            self.assertTrue(all(r['task']=='progress' and r['features']['task_progress']==1. for r in stops))
            timed=execute(case,root/'timed',mode='always_run',adaptation=cfg)
            self.assertFalse(timed['task_fulfilled'])
            self.assertIsNone(timed['first_goal_arrival'])


if __name__=='__main__':unittest.main()
