from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.spike_pattern_skills.associations import AssociationMemory
from experiments.continual_snn.models import FrozenBankModel
from experiments.continual_snn.compact_bank import CompactLegacyBank
from experiments.continual_snn.loading import load_selector


class CompactBankTests(unittest.TestCase):
    def fixture(self,conflict=False):
        encoder=FrozenBankModel().encoder
        rows=[dict(context_id=str(i),spike_counts=[1]+[0]*15,skill='dynamic/run') for i in range(3)]
        rows.append(dict(context_id='other',spike_counts=[3]+[0]*15,skill='dynamic/wait'))
        if conflict:rows.append(dict(context_id='conflict',spike_counts=[1]+[0]*15,skill='dynamic/wait'))
        legacy=AssociationMemory(encoder.checksum,rows,neighbors=3,max_distance=4.)
        return legacy,CompactLegacyBank.from_legacy(encoder,legacy)

    def test_multiplicity_preserves_neighbor_cutoff_and_all_tied_labels(self):
        legacy,compact=self.fixture()
        self.assertEqual(len(compact.counts),2)
        for counts in ([1,1]+[0]*14,[2]+[0]*15,[0]*16,[16]*16):
            a,b=legacy.decode(counts),compact.decode_counts(counts)
            self.assertEqual(a['proposed_skill'],b['proposed_skill'])
            self.assertEqual(a['reason'],b['reason'])
            self.assertEqual(len(a['neighbors']),b['matched_associations'])
        self.assertEqual(compact.decode_counts([1,1]+[0]*14)['proposed_skill'],'dynamic/run')
        self.assertIsNone(compact.decode_counts([2]+[0]*15)['proposed_skill'])
        _,conflict=self.fixture(True)
        self.assertIsNone(conflict.decode_counts([1]+[0]*15)['proposed_skill'])

    def test_roundtrip_and_multiplicity_tamper_rejected(self):
        _,model=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bank.npz';model.save(path);loaded=load_selector(path)
            self.assertEqual(model.checksum,loaded.checksum)
            self.assertEqual(model.decode_counts([1,1]+[0]*14),loaded.decode_counts([1,1]+[0]*14))
            with np.load(path) as bundle:changed={k:bundle[k].copy() for k in bundle.files}
            changed['multiplicity'][0]+=1
            bad=Path(folder)/'bad.npz';np.savez_compressed(bad,**changed)
            with self.assertRaisesRegex(ValueError,'checksum'):load_selector(bad)


if __name__=='__main__':unittest.main()
