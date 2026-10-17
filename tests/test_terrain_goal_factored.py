import tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.continual_snn.models import ROOT
from experiments.terrain_goal.encoder import TaskEncoder,add_task
from experiments.terrain_goal.models import Selector
from experiments.terrain_goal.factored import FactoredGoalSelector,fit_threshold


class FactoredGoalTests(unittest.TestCase):
    def base_and_rows(self):
        e=TaskEncoder('factor32');rows=[]
        for g in range(4):
            for task,label in (('hold','wait'),('progress','run'),('jump','jump')):
                f={n:0. for n in e.feature_names};f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
                rows.append(dict(features=add_task(f,task),label=label,group=str(g)))
        return Selector(e,'ridge').fit(rows),rows

    def test_closer_goal_cannot_revoke_hold_and_stream_jumps_are_preserved(self):
        base,rows=self.base_and_rows();model=FactoredGoalSelector(base,92,training_groups=['a'],training_rows=2)
        f=dict(rows[1]['features'],goal_mode=1.)
        actions=[model.predict(dict(f,goal_forward=d))['proposed_skill'] for d in np.linspace(.3,0.,101)]
        self.assertIn('wait',actions);first=actions.index('wait')
        self.assertTrue(all(a=='wait' for a in actions[first:]))
        self.assertEqual(model.predict(dict(rows[2]['features'],goal_mode=0.,goal_forward=0.))['proposed_skill'],'jump')
        self.assertEqual(model.predict(dict(f,goal_mode=0.,goal_forward=.001))['proposed_skill'],'run')

    def test_portable_model_roundtrip(self):
        base,_=self.base_and_rows();model=FactoredGoalSelector(base,92,training_groups=['a'],training_rows=2)
        parent=ROOT/'output/terrain_goal/test-tmp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as d:
            p=Path(d)/'m.npz';model.save(p);loaded=FactoredGoalSelector.load(p)
            self.assertEqual(loaded.checksum,model.checksum)


if __name__=='__main__':unittest.main()
