"""Verify diagnostic inputs and portable predictions, not a new motor policy."""
import tempfile,unittest
from pathlib import Path
from experiments.continual_snn.models import ROOT
from experiments.terrain_goal.loading_v2 import load_any
from experiments.terrain_goal.goal_bias import BiasedGoalFeatures


class GoalBiasEvidenceTests(unittest.TestCase):
    def model(self):return load_any(ROOT/'docs/agents/terrain_goal_v2/primary.npz')

    def features(self,model):
        result={n:0. for n in model.encoder.feature_names}
        result.update(goal_forward=.17,goal_mode=1.,task_progress=1.,
            footprint_clearance=1.,predicted_run_clearance=1.,predicted_jump_clearance=1.)
        return result

    def test_archive_replays_recorded_injected_inputs_without_second_bias(self):
        wrapper=BiasedGoalFeatures(self.model(),-.08);original=self.features(wrapper.model)
        injected=wrapper.augment_features(original,task='progress',geometry={},observed={})
        self.assertAlmostEqual(injected['goal_forward'],.09)
        self.assertEqual(original['goal_forward'],.17)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'predictor.npz';wrapper.save(path);loaded=load_any(path)
            self.assertEqual(loaded.checksum,wrapper.checksum)
            for now in (0.,.1):
                live=wrapper.predict(injected,now=now);replayed=loaded.predict(injected,now=now)
                self.assertEqual(live,replayed)
            self.assertEqual(live['proposed_skill'],'wait')
            self.assertEqual(live['goal_count_sum'],102)

    def test_hold_and_jump_goal_vectors_are_not_shifted(self):
        wrapper=BiasedGoalFeatures(self.model(),.08);original=self.features(wrapper.model)
        for task in ('hold','jump'):
            actual=wrapper.augment_features(original,task=task,geometry={},observed={})
            self.assertEqual(actual['goal_forward'],original['goal_forward'])
            self.assertEqual(actual['goal_left'],original['goal_left'])
            self.assertEqual(actual['footprint_clearance'],original['footprint_clearance'])


if __name__=='__main__':unittest.main()
