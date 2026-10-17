from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.continual_snn.encoding import PacketEncoder,PacketConfig
from experiments.continual_snn.prototypes import PrototypeMemory
from experiments.continual_snn.data import static_records


class PrototypeMemoryTests(unittest.TestCase):
    def test_explicit_duplicate_conflicts_abstain_and_counts_validate(self):
        features=static_records()[0]['features']
        records=[dict(features=features,label='run',group='a'),dict(features=features,label='wait',group='b')]
        memory=PrototypeMemory(PacketEncoder()).fit(records)
        result=memory.predict(features)
        self.assertIsNone(result['proposed_skill'])
        self.assertEqual(result['reason'],'conflicting_associations')
        self.assertEqual(len(memory.counts),1)
        with self.assertRaises(ValueError):memory.decode_counts([.5]*16)
        with self.assertRaises(ValueError):memory.decode_counts([65536]*16)
        with self.assertRaises(ValueError):PrototypeMemory().fit([])

    def test_roundtrip_and_tampered_prototypes(self):
        rows=[r for r in static_records() if r['split']=='train']
        model=PrototypeMemory(PacketEncoder(PacketConfig(neurons=64))).fit(rows)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bank.npz';model.save(path)
            loaded=PrototypeMemory.load(path)
            self.assertEqual(model.checksum,loaded.checksum)
            self.assertEqual(model.predict(rows[0]['features']),loaded.predict(rows[0]['features']))
            with np.load(path) as bundle:changed={k:bundle[k].copy() for k in bundle.files}
            changed['label_bits'][0]=7
            bad=Path(folder)/'changed.npz';np.savez_compressed(bad,**changed)
            with self.assertRaisesRegex(ValueError,'checksum'):PrototypeMemory.load(bad)


if __name__=='__main__':unittest.main()
