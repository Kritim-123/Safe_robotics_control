from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.continual_snn.predictive import PredictivePacketEncoder
from experiments.continual_snn.encoding import PacketEncoder,PacketConfig
from experiments.continual_snn.data import static_records
from experiments.continual_snn.models import load_model
from experiments.continual_snn.readout import SpikeReadout
from experiments.continual_snn.prototypes import PrototypeMemory


class PredictivePopulationTests(unittest.TestCase):
    def test_preserves_all_64_sensory_neurons_and_isolates_new_information(self):
        encoder=PredictivePacketEncoder();old=PacketEncoder(PacketConfig(neurons=64))
        for row in static_records()[::9]:
            f=dict(row['features'],predicted_run_clearance=.1,predicted_jump_clearance=.3)
            packet=encoder.encode(f)
            self.assertEqual(packet['spike_counts'][:64],old.encode(row['features'])['spike_counts'])
            other=encoder.encode(dict(f,predicted_run_clearance=2.,predicted_jump_clearance=3.))
            self.assertEqual(packet['spike_counts'][:64],other['spike_counts'][:64])
            self.assertNotEqual(packet['spike_counts'][64:],other['spike_counts'][64:])
        w=encoder.weights.copy();w[64,0]=1.
        with self.assertRaises(ValueError):PredictivePacketEncoder(weights=w,biases=encoder.biases)

    def test_roundtrip_full_arrays_readout_and_memory(self):
        records=[dict(r,features=dict(r['features'],predicted_run_clearance=.3,
            predicted_jump_clearance=.5)) for r in static_records()[:12]]
        encoder=PredictivePacketEncoder()
        models=[SpikeReadout(encoder,analog=a).fit([r['features'] for r in records],
            [r['label'] for r in records],[r['group'] for r in records]) for a in (False,True)]
        models.append(PrototypeMemory(encoder).fit(records))
        with tempfile.TemporaryDirectory() as folder:
            for i,model in enumerate(models):
                path=Path(folder)/f'{i}.npz';model.save(path);loaded=load_model(path)
                self.assertEqual(loaded.checksum,model.checksum)
                self.assertEqual(model.predict(records[0]['features']),loaded.predict(records[0]['features']))
                np.testing.assert_array_equal(loaded.encoder.weights,model.encoder.weights)


if __name__=='__main__':unittest.main()
