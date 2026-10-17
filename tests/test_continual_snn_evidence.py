import unittest
from experiments.continual_snn.audit import paired
from experiments.continual_snn.transfer import terrain_cases,heading_cases
from experiments.continual_snn.report import failure_signals


class EvidenceTests(unittest.TestCase):
    def test_paired_uncertainty_resamples_shared_seed_blocks(self):
        rows=[]
        for seed,number in ((1,9),(2,1)):
            for i in range(number):
                for arm in ('before','after'):
                    rows.append(dict(arm=arm,summary=dict(case=f'{seed}-{i}',seed=seed,
                        task_fulfilled=seed==2 or arm=='after')))
        result=paired(rows,'before','after')
        self.assertEqual(result['matched_episodes'],10)
        self.assertEqual(result['independent_seed_blocks'],2)
        self.assertEqual(result['mean_success_difference'],.5)
        self.assertEqual(result['seed_cluster_bootstrap_percentile95'],[0.,1.])

    def test_transfer_cases_require_traversal_not_merely_starting(self):
        terrain=terrain_cases();heading=heading_cases([9200])
        self.assertEqual(len(terrain),60);self.assertEqual(len(heading),8)
        self.assertEqual(len({c.scenario.name for c in terrain}),60)
        for case in terrain:
            case.validate();self.assertEqual(case.required_progress,1.55)
            self.assertEqual(case.request_at(2.),'progress')
            self.assertEqual(case.request_at(23.),'hold')

    def test_completed_episode_with_insufficient_progress_remains_a_failure(self):
        result=dict(episode_completed=True,reason='completed',progress=.01,required_progress=.4,
            completed_jumps=0,required_jumps=1,maximum_jumps=1,hold_violation_packets=0,failed_jumps=0)
        self.assertEqual(failure_signals(result),['insufficient progress','missing verified hop'])


if __name__=='__main__':unittest.main()
