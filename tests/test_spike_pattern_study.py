from collections import Counter
import copy
import json
import unittest

from experiments.spike_pattern_skills.analyze import analyze
from experiments.spike_pattern_skills.situations import FAMILIES, situations


def record(context, family, raster):
    counts = [sum(step[i] for step in raster) for i in range(len(raster[0]))]
    return dict(observation=dict(context_id=context, metadata=dict(study_family=family)),
                recording=dict(output_raster=raster, spike_counts=counts,
                               active_neuron_ids=[i for i, n in enumerate(counts) if n]))


class SpikePatternStudyTests(unittest.TestCase):
    def test_protocol_has_100_reproducible_distinct_cases(self):
        cases = situations()
        self.assertEqual(len(cases), 100)
        self.assertEqual(Counter(c.family for c in cases), Counter({f: 10 for f in FAMILIES}))
        self.assertEqual(sum(c.jump_requested for c in cases), 50)
        self.assertEqual(len({c.scene_seed for c in cases}), 100)
        encoded = []
        for case in cases:
            case.scenario.validate()
            geometry = case.scenario.to_dict(); geometry.pop('name')
            encoded.append(json.dumps(geometry, sort_keys=True))
        self.assertEqual(len(set(encoded)), 100)
        self.assertEqual(cases, situations())
        self.assertNotEqual(cases, situations(20261017))

    def test_collisions_counts_and_temporal_patterns_are_counted_separately(self):
        records = [record('a', 'blocked', [[1, 0], [1, 0]]),
                   record('b', 'crossing', [[1, 0], [0, 0]]),
                   record('c', 'crossing', [[0, 0], [1, 0]])]
        result = analyze(records)
        self.assertEqual(result['representations']['active_ids']['unique_patterns'], 1)
        self.assertEqual(result['representations']['active_ids']['shared_pairs'], 3)
        self.assertEqual(result['representations']['active_ids']['cross_family_shared_pairs'], 2)
        self.assertEqual(result['representations']['spike_counts']['unique_patterns'], 2)
        self.assertEqual(result['representations']['raster']['unique_patterns'], 3)
        self.assertEqual(result['active_id_shared_pairs_resolved_by_counts'], 2)
        self.assertEqual(result['active_id_shared_pairs_still_ambiguous'], 1)
        self.assertEqual(result['total_spikes'], 4)
        bad = copy.deepcopy(records)
        bad[0]['recording']['spike_counts'][0] = 1
        with self.assertRaises(ValueError):
            analyze(bad)
        with self.assertRaises(ValueError):
            analyze([])

    def test_silent_patterns_are_retained(self):
        records = [record('a', 'clear', [[0, 0], [0, 0]]),
                   record('b', 'blocked', [[0, 0], [0, 0]])]
        result = analyze(records)
        self.assertEqual(result['silent_observations'], 2)
        self.assertEqual(result['representations']['spike_counts']['shared_pairs'], 1)
        self.assertEqual(result['active_id_shared_pairs_resolved_by_counts'], 0)

    def test_published_records_match_model_csv_and_scalar_reference(self):
        from experiments.spike_pattern_skills.audit import audit
        result = audit()
        self.assertEqual(result['records'], 100)
        self.assertEqual(result['exact_model_replays'], 100)
        self.assertEqual(result['scalar_neuron_steps_checked'], 25600)
        self.assertTrue(result['csv_counts_match'] and result['collision_counts_match'])


if __name__ == '__main__':
    unittest.main()
