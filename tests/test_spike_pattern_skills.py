import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.spike_pattern_skills.encoder import RandomLIFEncoder
from experiments.spike_pattern_skills.observations import FEATURES, feature_vector
from experiments.spike_pattern_skills.record import capture_observation, run, DEFAULT_CONFIG


def example():
    values = [1, 1.2, -.1, 0, .15, .2, .4, .4, .6, 3, 0, 0, .265, 0]
    return dict(observation_id='obs-1', context_id='scene-1', time_s=2., source='synthetic',
                features=dict(zip(FEATURES, values)), metadata={})


class SpikePatternTests(unittest.TestCase):
    def test_seed_and_saved_arrays_are_repeatable_and_frozen(self):
        a, b = RandomLIFEncoder.initialize(42), RandomLIFEncoder.initialize(42)
        c = RandomLIFEncoder.initialize(43)
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(a.biases, b.biases)
        self.assertEqual(a.checksum, b.checksum)
        self.assertFalse(np.array_equal(a.weights, c.weights))
        self.assertNotEqual(a.checksum, c.checksum)
        with self.assertRaises(ValueError):
            a.weights[0, 0] = 0
        a.metadata['window_steps'] = 17
        with self.assertRaisesRegex(ValueError, 'Frozen'):
            a.encode(example()['features'])

    def test_lif_equations_against_hand_calculated_trace(self):
        template = RandomLIFEncoder.initialize(pattern_neurons=1, window_steps=4, beta_per_step=.5)
        network = RandomLIFEncoder(template.metadata, np.zeros((1, len(FEATURES))), [.6], template.scales)
        result = network.encode(dict.fromkeys(FEATURES, 0))
        self.assertEqual(result['output_raster'], [[0], [0], [1], [0]])
        self.assertEqual(result['spike_counts'], [1])
        self.assertEqual(result['active_neuron_ids'], [0])
        np.testing.assert_allclose(np.array(result['membrane_before_reset']).ravel(), [.6, .9, 1.05, .625])
        np.testing.assert_allclose(np.array(result['membrane_after_reset']).ravel(), [.6, .9, .05, .625])
        silent = RandomLIFEncoder(template.metadata, np.zeros((1, len(FEATURES))), [0.], template.scales)
        self.assertTrue(silent.encode(dict.fromkeys(FEATURES, 0))['silent'])
        self.assertEqual(silent.encode(dict.fromkeys(FEATURES, 0))['active_neuron_ids'], [])

    def test_binary_raster_counts_reset_and_named_feature_order(self):
        model = RandomLIFEncoder.initialize()
        obs = example()
        first = model.encode(obs['features'])
        other = dict(obs['features'], jump_requested=1, obstacle_forward=3.)
        model.encode(other)
        self.assertEqual(model.encode(obs['features']), first)
        self.assertEqual(model.encode(dict(reversed(list(obs['features'].items())))), first)
        raster = np.asarray(first['output_raster'])
        self.assertEqual(raster.shape, (16, 16))
        self.assertTrue(np.isin(raster, [0, 1]).all())
        np.testing.assert_array_equal(raster.sum(0), first['spike_counts'])
        self.assertEqual(np.flatnonzero(raster.sum(0)).tolist(), first['active_neuron_ids'])

    def test_save_load_integrity_and_no_overwrite(self):
        model = RandomLIFEncoder.initialize()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            model.save(path)
            loaded = RandomLIFEncoder.load(path)
            self.assertEqual(loaded.record(example()), model.record(example()))
            with self.assertRaises(FileExistsError):
                model.save(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            arrays['weights'][0, 0] += .01
            np.savez(Path(tmp)/'tampered.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                RandomLIFEncoder.load(Path(tmp)/'tampered.npz')

    def test_invalid_inputs_and_parameters_are_rejected(self):
        obs = example()
        for features in (dict(obs['features'], future_success=1),
                         dict(obs['features'], speed=float('nan')),
                         dict(obs['features'], obstacle_present=.5),
                         {'speed': 0}):
            with self.assertRaises(ValueError):
                feature_vector(features)
        for kwargs in (dict(seed=True), dict(seed=-1), dict(pattern_neurons=0),
                       dict(window_steps=0), dict(beta_per_step=1.), dict(threshold=0.)):
            with self.assertRaises(ValueError):
                RandomLIFEncoder.initialize(**kwargs)
        for timestamp in (-1, float('inf'), True):
            with self.assertRaises(ValueError):
                RandomLIFEncoder.initialize().record(dict(obs, time_s=timestamp))

    def test_outcome_metadata_does_not_enter_network_or_assign_skill(self):
        model = RandomLIFEncoder.initialize()
        a, b = example(), example()
        b['metadata'] = dict(future_success=True, label_skill='dynamic/jump')
        self.assertEqual(model.record(a)['recording'], model.record(b)['recording'])
        self.assertIsNone(model.record(b)['skill_label'])
        self.assertEqual(model.record(b)['decoder_status'], 'not_implemented')

    def test_real_simulation_capture_has_no_skill_execution(self):
        observation = capture_observation('crossing', 101)
        self.assertEqual(observation['source'], 'mujoco_state')
        self.assertAlmostEqual(observation['time_s'], 2., places=6)
        episode = observation['metadata']['capture_episode']
        self.assertTrue(episode['capture_only'])
        self.assertFalse(episode['skill_executed'] or episode['collision'] or episode['fall'])
        self.assertLessEqual(episode['peak_torque'], 24)
        feature_vector(observation['features'])
        model = RandomLIFEncoder.initialize()
        self.assertEqual(model.record(observation), model.record(observation))

    def test_recording_roundtrip_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = root/'observations.json'
            observations.write_text(json.dumps([example()]))
            first = run(root/'first', observations=observations)
            second = run(root/'second', observations=observations, model=root/'first/model.npz')
            self.assertTrue(first['repeated_inference_exact'] and first['save_load_exact'])
            self.assertEqual(first['encoder_checksum'], second['encoder_checksum'])
            self.assertEqual((root/'first/records.jsonl').read_bytes(), (root/'second/records.jsonl').read_bytes())
            with self.assertRaises(FileExistsError):
                run(root/'first', observations=observations)
            settings = json.loads(DEFAULT_CONFIG.read_text())
            settings['encoder']['reset_between_observations'] = False
            bad = root/'bad.json'; bad.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, 'encoder contract'):
                run(root/'bad-output', config=bad, observations=observations)


if __name__ == '__main__':
    unittest.main()
