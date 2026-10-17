import tempfile,unittest
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.terrain_goal.encoder import TaskEncoder,add_task
from experiments.terrain_goal.models import Selector
from experiments.terrain_goal.factored import FactoredGoalSelector
from experiments.terrain_goal.hysteresis import HystereticGoalSelector


class GoalReadoutHysteresisTests(unittest.TestCase):
    def setUp(self):
        e=TaskEncoder('factor32');rows=[]
        for group in range(4):
            for task,label in (('hold','wait'),('progress','run'),('jump','jump')):
                f={n:0. for n in e.feature_names}
                f.update(predicted_run_clearance=1.,predicted_jump_clearance=1.,footprint_clearance=1.)
                rows.append(dict(features=add_task(f,task),label=label,group=str(group)))
        self.features=dict(rows[1]['features'],goal_mode=1.)
        parent=ROOT/'output/terrain_goal/test-tmp';parent.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.directory.cleanup)
        p=Path(self.directory.name)/'base.npz'
        FactoredGoalSelector(Selector(e,'ridge').fit(rows),92,training_groups=['a'],training_rows=2).save(p)
        self.model=HystereticGoalSelector.from_path(p)

    def ask(self,distance,now):return self.model.predict(dict(self.features,goal_forward=distance),now=now)

    def test_hold_survives_goal_noise_and_releases_after_two_fresh_far_packets(self):
        self.assertEqual(self.ask(.08,1.)['proposed_skill'],'wait')
        self.assertFalse(self.model.held)
        self.assertEqual(self.ask(.08,1.1)['proposed_skill'],'wait')
        self.assertEqual(self.ask(.13,1.2)['proposed_skill'],'wait')
        self.assertEqual(self.ask(.2,1.3)['proposed_skill'],'wait')
        self.assertEqual(self.ask(.2,1.3)['release_confirmation_count'],1)
        self.assertEqual(self.ask(.2,1.4)['proposed_skill'],'run')
        self.ask(.08,1.5);self.model.reset()
        self.assertEqual(self.ask(.13,0.)['proposed_skill'],'run')

    def test_task_change_and_portable_checkpoint_do_not_retain_runtime_commitment(self):
        self.ask(.08,1.)
        jump=add_task(self.features,'jump')
        self.assertEqual(self.model.predict(jump,now=1.1)['proposed_skill'],'jump')
        self.assertFalse(self.model.held)
        p=Path(self.directory.name)/'model.npz';self.model.save(p)
        loaded=HystereticGoalSelector.load(p)
        self.assertEqual(loaded.checksum,self.model.checksum);self.assertFalse(loaded.held)
        with self.assertRaises(ValueError):self.ask(.1,.5)


if __name__=='__main__':unittest.main()
