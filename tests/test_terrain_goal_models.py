import tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.terrain_goal.encoder import TaskEncoder,add_task
from experiments.terrain_goal.models import Selector
from experiments.continual_snn.models import ROOT


class TerrainGoalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent=ROOT/'output/terrain_goal/test-tmp';cls.parent.mkdir(parents=True,exist_ok=True)
    def test_roundtrip_fixed_weights_and_both_decoders(self):
        encoder=TaskEncoder('factor32')
        rows=[]
        for group in range(4):
            for task,label in (('hold','wait'),('progress','run'),('jump','jump')):
                f={n:0. for n in encoder.feature_names}
                f.update(predicted_run_clearance=1.+group*.1,predicted_jump_clearance=1.,footprint_clearance=1.)
                rows.append(dict(group=str(group),features=add_task(f,task),label=label))
        for kind in ('ridge','prototype'):
            model=Selector(encoder,kind).fit(rows)
            with tempfile.TemporaryDirectory(dir=self.parent) as folder:
                path=Path(folder)/'model.npz';model.save(path);loaded=Selector.load(path)
                self.assertEqual(loaded.checksum,model.checksum)
                np.testing.assert_array_equal(loaded.encoder.weights,model.encoder.weights)
                for row in rows:
                    self.assertEqual(loaded.predict(row['features']),model.predict(row['features']))
            with self.assertRaises(ValueError):model.inner.decode_counts([-1]*32) if kind=='prototype' else model.predict(dict(rows[0]['features'],task_hold=float('nan')))


if __name__=='__main__':unittest.main()
