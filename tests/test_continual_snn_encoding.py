from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from experiments.continual_snn.encoding import PacketConfig,PacketEncoder,spike_embedding
from experiments.continual_snn.data import static_records
from experiments.continual_snn.readout import SpikeReadout,grouped_weights


class PacketEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=static_records()

    def test_original_rows_and_independent_count_packets(self):
        original=PacketEncoder();extended=PacketEncoder(PacketConfig(neurons=64))
        np.testing.assert_array_equal(extended.weights[:16],original.base.weights)
        np.testing.assert_array_equal(extended.biases[:16],original.base.biases)
        for row in self.records[::9]:
            features=row['features'];a=original.encode(features);b=original.base.encode(features)
            self.assertEqual(a['spike_counts'],b['spike_counts'])
            self.assertEqual(extended.encode(features)['spike_counts'][:16],a['spike_counts'])
            original.encode(self.records[-1]['features'])
            self.assertEqual(original.encode(features)['spike_counts'],a['spike_counts'])
        self.assertFalse(extended.weights.flags.writeable)

    def test_timing_distance_distinguishes_equal_counts_and_includes_tail(self):
        a=np.asarray([[1],[0],[0],[0]],dtype=np.uint8)
        b=np.asarray([[0],[0],[0],[1]],dtype=np.uint8)
        np.testing.assert_array_equal(spike_embedding(a),spike_embedding(b))
        self.assertGreater(np.linalg.norm(spike_embedding(a,'filtered')-spike_embedding(b,'filtered')),0.)
        # Verify the analytic discrete tail against a very long zero-padding.
        short=spike_embedding(a,'filtered',2.)
        padded=np.vstack([a,np.zeros((200,1),dtype=np.uint8)])
        long=spike_embedding(padded,'filtered',2.)
        self.assertAlmostEqual(float(short@short),float(long@long),places=12)

    def test_adaptation_synaptic_and_configuration_validation(self):
        feature=self.records[0]['features']
        for kind in ('lif','alif','synaptic'):
            encoder=PacketEncoder(PacketConfig(neurons=64,neuron=kind))
            result=encoder.encode(feature)
            self.assertEqual(result['raster'].shape,(16,64))
            self.assertTrue(np.isfinite(result['embedding']).all())
            self.assertEqual(result['reset_state'],'zero_membranes_each_packet')
        with self.assertRaises(ValueError):PacketEncoder(PacketConfig(neurons=3))
        with self.assertRaises(ValueError):PacketEncoder(PacketConfig(adaptation_decay=1.))

    def test_readout_round_trip_group_weights_and_tamper_detection(self):
        rows=[r for r in self.records if r['split']=='train']
        model=SpikeReadout().fit([r['features'] for r in rows],[r['label'] for r in rows],[r['group'] for r in rows])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'readout.npz';model.save(path);loaded=SpikeReadout.load(path)
            self.assertEqual(model.checksum,loaded.checksum)
            self.assertEqual(model.predict(rows[0]['features']),loaded.predict(rows[0]['features']))
            with np.load(path) as b:values={k:b[k].copy() for k in b.files}
            values['coefficients'][0,0]+=1.
            bad=Path(folder)/'changed.npz';np.savez_compressed(bad,**values)
            with self.assertRaisesRegex(ValueError,'checksum'):SpikeReadout.load(bad)
        w=grouped_weights(['a','a','b','c'],np.array([0,0,1,2]))
        self.assertAlmostEqual(float(w[:2].sum()),float(w[2]))
        self.assertAlmostEqual(float(w[2]),float(w[3]))

    def test_silent_packet_abstains_even_if_readout_has_a_high_score(self):
        rows=self.records[:12]
        model=SpikeReadout().fit([r['features'] for r in rows],[r['label'] for r in rows],[r['group'] for r in rows])
        empty=dict(spike_counts=[0]*16,active_neuron_ids=[],embedding=np.zeros(16),reset_state='test')
        with patch.object(model.encoder,'encode',return_value=empty):
            result=model.predict(rows[0]['features'])
        self.assertIsNone(result['proposed_skill'])
        self.assertEqual(result['reason'],'silent_pattern')


if __name__=='__main__':unittest.main()
