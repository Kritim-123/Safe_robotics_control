from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.continual_snn.data import static_records
from experiments.continual_snn.encoding import PacketConfig,PacketEncoder
from experiments.continual_snn.memory import MemoryConfig,TemporalPacketEncoder,TemporalReadout


class TemporalPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=static_records()

    def test_reset_arm_equivalence_and_counts_are_fresh(self):
        cfg=PacketConfig(neurons=64,neuron='alif',adaptation_decay=.99)
        original=PacketEncoder(cfg);temporal=TemporalPacketEncoder(cfg,MemoryConfig(retain=False))
        for i,row in enumerate(self.rows[::13]):
            actual=temporal.encode(row['features'],now=i*.1)
            self.assertEqual(actual['spike_counts'],original.encode(row['features'])['spike_counts'])
            self.assertTrue(all(v<=16 for v in actual['spike_counts']))

    def test_history_changes_state_without_cross_episode_leakage(self):
        cfg=PacketConfig(neurons=64)
        memory=MemoryConfig(reset_on_command_change=False)
        a,b=TemporalPacketEncoder(cfg,memory),TemporalPacketEncoder(cfg,memory)
        first=self.rows[0]['features'];different=self.rows[-1]['features']
        for i in range(4):
            a.encode(first,now=i*.1);b.encode(different,now=i*.1)
        a.encode(first,now=.4);b.encode(first,now=.4)
        self.assertGreater(np.linalg.norm(a.membrane-b.membrane),0.)
        a.reset();b.reset()
        self.assertEqual(a.encode(first,now=.5)['spike_counts'],b.encode(first,now=.5)['spike_counts'])

    def test_command_change_stale_gap_duplicate_and_backwards_time(self):
        encoder=TemporalPacketEncoder(PacketConfig(neurons=64))
        features=self.rows[0]['features']
        a=encoder.encode(features,now=2.)
        self.assertIs(a,encoder.encode(features,now=2.))
        changed=dict(features,jump_requested=1.-features['jump_requested'])
        b=encoder.encode(changed,now=2.1)
        self.assertEqual(b['memory_reset_reason'],'command_changed')
        c=encoder.encode(changed,now=3.)
        self.assertEqual(c['memory_reset_reason'],'stale_gap')
        with self.assertRaises(ValueError):encoder.encode(changed,now=2.)
        self.assertIsNone(encoder.last_time)

    def test_temporal_readout_roundtrip_and_grouped_sequence_boundary(self):
        records=[r for r in self.rows if r['split']=='train']
        encoder=TemporalPacketEncoder(PacketConfig(neurons=64))
        embedding=encoder.embed_records(records)
        self.assertIsNone(encoder.last_time)
        model=TemporalReadout(encoder).fit_embeddings(embedding,[r['label'] for r in records],[r['group'] for r in records])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'temporal.npz';model.save(path);loaded=TemporalReadout.load(path)
            self.assertEqual(model.checksum,loaded.checksum)
            for i,row in enumerate(records[:4]):
                self.assertEqual(model.predict(row['features'],now=i*.1),loaded.predict(row['features'],now=i*.1))


if __name__=='__main__':unittest.main()
