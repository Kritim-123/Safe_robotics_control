from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid.arena import Scenario
from hybrid.terrain import TerrainPatch
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.spike_pattern_skills.encoder import RandomLIFEncoder
from experiments.spike_pattern_skills.execute_associations import execute_case, measure_case_outcome
from experiments.spike_pattern_skills.learn_associations import ENCODER_PATH
from experiments.spike_pattern_skills.learn_expanded import conflict_summary
from experiments.spike_pattern_skills.observations import observation_from_simulation
from experiments.spike_pattern_skills.outcome_cases import learning_cases
from experiments.spike_pattern_skills.terrain_encoding import TerrainAwareEncoder, terrain_observation, surface_features
from experiments.spike_pattern_skills.dense_terrain_encoding import DenseTerrainEncoder, dense_observation, dense_surface_features


OBSERVATION = dict(time=2., position=[0., 0.], yaw=0., speed=0., height=.27)


class ExpandedAssociationTests(unittest.TestCase):
    def setUp(self):
        self.legacy = RandomLIFEncoder.load(ENCODER_PATH)
        self.encoder = TerrainAwareEncoder.extend(self.legacy)

    def observed(self, spec, terrain=True):
        function = terrain_observation if terrain else observation_from_simulation
        return function(spec, OBSERVATION, False, observation_id='test', context_id='test')

    def test_predeclared_splits_and_no_scene_id_overlap(self):
        cases = learning_cases()
        self.assertEqual(len(cases), 96)
        self.assertEqual(len({c.scenario.name for c in cases}), 96)
        self.assertEqual([sum(c.split == split for c in cases) for split in ('train','validation','held_out')], [54,14,28])
        self.assertEqual(cases, learning_cases())
        self.assertEqual(sum(c.jump_requested for c in cases), 9)

    def test_extension_preserves_old_flat_response_and_saved_weights(self):
        np.testing.assert_array_equal(self.encoder.weights[:, :14], self.legacy.weights)
        np.testing.assert_array_equal(self.encoder.biases, self.legacy.biases)
        spec = Scenario('flat')
        old = self.legacy.encode(self.observed(spec, False)['features'])
        new = self.encoder.encode(self.observed(spec)['features'])
        self.assertEqual(old['spike_counts'], new['spike_counts'])
        self.assertEqual(old['output_raster'], new['output_raster'])
        self.assertNotEqual(self.encoder.checksum, self.legacy.checksum)
        self.assertEqual(self.encoder.checksum, TerrainAwareEncoder.extend(self.legacy).checksum)
        self.assertFalse(self.encoder.weights.flags.writeable)

    def test_old_aliasing_and_terrain_discrimination(self):
        flat, ramp = learning_cases()[0].scenario, learning_cases()[12].scenario
        old = [self.legacy.encode(self.observed(spec, False)['features'])['spike_counts'] for spec in (flat,ramp)]
        new = [self.encoder.encode(self.observed(spec)['features'])['spike_counts'] for spec in (flat,ramp)]
        self.assertEqual(old[0], old[1])
        self.assertNotEqual(new[0], new[1])
        examples = [dict(context_id=str(i), skill=skill, spike_counts=old[i])
                    for i, skill in enumerate(('dynamic/run','dynamic/wait'))]
        self.assertEqual(conflict_summary(examples)['conflicting_vectors'], 1)
        self.assertIsNone(AssociationMemory(self.legacy.checksum, examples).decode(old[0])['proposed_skill'])
        for i in range(2):
            examples[i]['spike_counts'] = new[i]
        memory = AssociationMemory(self.encoder.checksum, examples)
        self.assertEqual(memory.decode(new[0])['proposed_skill'], 'dynamic/run')
        self.assertEqual(memory.decode(new[1])['proposed_skill'], 'dynamic/wait')

    def test_surface_samples_are_local_and_ignore_scene_name(self):
        flat = Scenario('flat')
        self.assertEqual(list(surface_features(flat, OBSERVATION).values()), [0.]*8)
        patch = TerrainPatch('patch', (.8, 0), (.2, .8), .04, friction=.3)
        spec = replace(flat, terrain=(patch,))
        values = surface_features(spec, OBSERVATION)
        self.assertEqual(values['terrain_present'], 1.)
        self.assertAlmostEqual(values['surface_rise'], .04)
        self.assertAlmostEqual(values['largest_up_step'], .04)
        self.assertAlmostEqual(values['largest_down_step'], .04)
        self.assertAlmostEqual(values['friction_loss'], .5)
        self.assertEqual(values, surface_features(replace(spec, name='different_label'), OBSERVATION))
        distant = replace(spec, terrain=(replace(patch, center=(3., 0)),))
        self.assertEqual(list(surface_features(distant, OBSERVATION).values()), [0.]*8)

    def test_schema_and_artifact_separation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'terrain.npz'
            self.encoder.save(path)
            loaded = TerrainAwareEncoder.load(path)
            self.assertEqual(loaded.checksum, self.encoder.checksum)
            with self.assertRaises(ValueError):
                RandomLIFEncoder.load(path)
            with self.assertRaises(ValueError):
                TerrainAwareEncoder.load(ENCODER_PATH)
        item = self.observed(Scenario('flat'))
        item['features']['future_success'] = 1.
        with self.assertRaises(ValueError):
            self.encoder.encode(item['features'])

    def test_wait_does_not_fulfill_terrain_task(self):
        case = learning_cases()[0]
        summary = dict(decision_observation={'yaw': 0}, displacement=[0.,0.], skill='wait',
            success=True, collision=False, fall=False, max_torque_limit_ratio=.4)
        result = measure_case_outcome(case, [dict(x=0.,y=0.)], summary)
        self.assertFalse(result['task_fulfilled'])
        self.assertIsNone(result['utility'])

    def test_dense_scan_detects_threshold_missed_by_coarse_scan(self):
        spec = learning_cases()[6].scenario
        self.assertEqual(surface_features(spec, OBSERVATION)['terrain_present'], 0.)
        values = dense_surface_features(spec, OBSERVATION)
        self.assertEqual(values['terrain_present'], 1.)
        self.assertAlmostEqual(values['surface_rise'], spec.terrain[0].height)
        dense = DenseTerrainEncoder.extend(self.legacy)
        flat = replace(spec, terrain=())
        counts = []
        for scene in (flat,spec):
            item = dense_observation(scene, OBSERVATION, False, observation_id='x',context_id='x')
            counts.append(dense.encode(item['features'])['spike_counts'])
        self.assertNotEqual(counts[0],counts[1])

    def test_coarse_and_dense_schemas_cannot_be_cross_loaded(self):
        dense = DenseTerrainEncoder.extend(self.legacy)
        with tempfile.TemporaryDirectory() as root:
            coarse_path, dense_path = Path(root)/'coarse.npz', Path(root)/'dense.npz'
            self.encoder.save(coarse_path);dense.save(dense_path)
            self.assertEqual(DenseTerrainEncoder.load(dense_path).checksum,dense.checksum)
            with self.assertRaises(ValueError):
                TerrainAwareEncoder.load(dense_path)
            with self.assertRaises(ValueError):
                DenseTerrainEncoder.load(coarse_path)

    def test_actual_terrain_encoder_execution_and_logging(self):
        from experiments.spike_pattern_skills.record import capture_scenario
        case = learning_cases()[0]
        item = capture_scenario(case.scenario, False, case.scenario.name, case.scene_seed)
        observed = item['metadata']['robot_observation']
        features = terrain_observation(case.scenario, observed, False,
            observation_id='decision', context_id=case.scenario.name)['features']
        counts = self.encoder.encode(features)['spike_counts']
        memory = AssociationMemory(self.encoder.checksum, [dict(context_id='verified-flat',
            skill='dynamic/run', spike_counts=counts)])
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'episode'
            result = execute_case(case, memory, self.encoder, folder, duration=8., timeout=18.)
            self.assertTrue(result['outcome']['task_fulfilled'])
            self.assertEqual(result['summary']['skill'], 'run')
            rows = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows[-1]['observation']['features']), 22)
            self.assertEqual(len(rows[-1]['diagnostic_pattern']['spike_counts']), 16)
            self.assertFalse(result['decision']['supervision']['fallback'])

    def test_saved_dense_bank_has_only_training_data_and_resolves_flat_threshold_alias(self):
        from experiments.spike_pattern_skills.run_learned import MODEL_BUNDLE
        from experiments.spike_pattern_skills.situations import situations
        encoder = DenseTerrainEncoder.load(MODEL_BUNDLE/'dense_encoder.npz')
        memory = AssociationMemory.load(MODEL_BUNDLE/'dense.json',encoder.checksum)
        allowed = {c.scenario.name for c in learning_cases() if c.split=='train'}
        allowed.update('original/'+c.scenario.name for c in situations() if c.variant<4)
        self.assertEqual({e['context_id'] for e in memory.examples},allowed)
        self.assertEqual(len(memory.examples),94)
        self.assertTrue(all(e['split']=='train' for e in memory.examples))
        self.assertEqual(conflict_summary(memory.examples)['conflicting_vectors'],0)
        labels = list(map(json.loads,(MODEL_BUNDLE/'reencoded_labels.jsonl').read_text().splitlines()))
        for context,skill in (('new-terrain-00-flat-00','dynamic/run'),
                              ('new-terrain-01-thresholds-00','dynamic/wait')):
            record = next(r for r in labels if r['context_id']==context)
            self.assertEqual(memory.decode(record['dense_counts'])['proposed_skill'],skill)


if __name__ == '__main__':
    unittest.main()
