import unittest
import numpy as np
from hybrid.skill_lif import decode_counts, FEATURES, SKILLS, SkillLIF


class OutputDecoderTests(unittest.TestCase):
    def test_masks_override_spikes(self):
        self.assertEqual(decode_counts([0,16,2],[True,False,True])[0],2)

    def test_zero_spikes_require_wait(self):
        self.assertEqual(decode_counts([0,0,0],[True,True,True])[0],0)
        self.assertIsNone(decode_counts([0,0,0],[False,True,True])[0])

    def test_no_eligible_and_ties(self):
        self.assertIsNone(decode_counts([16,16,16],[False]*3)[0])
        self.assertEqual(decode_counts([3,3,3],[True]*3)[0],0)

    def test_invalid_output(self):
        for counts in ([np.nan,0,0],[-1,0,0],[1,2]):
            with self.assertRaises(ValueError): decode_counts(counts,[True]*3)

    def test_output_lif_emits_binary_and_resets_each_window(self):
        model = SkillLIF.__new__(SkillLIF)
        model.steps, model.beta = 16, .9
        model.w = dict(mean=np.zeros(len(FEATURES),np.float32),std=np.ones(len(FEATURES),np.float32),
                       w1=np.zeros((32,len(FEATURES)),np.float32),b1=np.ones(32,np.float32),
                       w2=np.full((3,32),.01,np.float32),b2=np.array([0.,.2,.4],np.float32))
        counts, trace = model.predict(np.zeros(len(FEATURES)))
        self.assertEqual(trace['output_spikes'].shape,(16,3))
        self.assertTrue(np.isin(trace['output_spikes'],[0,1]).all())
        np.testing.assert_array_equal(counts,trace['output_spikes'].sum(0))
        self.assertTrue(np.all(counts>0))
        np.testing.assert_array_equal(counts,model.predict(np.zeros(len(FEATURES)))[0])
        self.assertEqual(model.choose(np.zeros(len(FEATURES)),[True]*3)[0],2)
        with self.assertRaises(ValueError): model.predict(np.full(len(FEATURES),np.nan))

    def test_no_future_features(self):
        self.assertNotIn('outcome_success',FEATURES)
        self.assertEqual(SKILLS,('wait','run','jump'))


if __name__=='__main__': unittest.main()
