import json
from pathlib import Path
import tempfile
import unittest

from experiments.spike_pattern_skills.associations import AssociationMemory, outcome, supervise
from experiments.spike_pattern_skills.encoder import RandomLIFEncoder
from experiments.spike_pattern_skills.execute_associations import DEFAULT_ASSOCIATION, execute_case
from experiments.spike_pattern_skills.learn_associations import ENCODER_PATH
from experiments.spike_pattern_skills.situations import situations


def example(name, count, skill='dynamic/run'):
    return dict(context_id=name, spike_counts=[count]+[0]*15, skill=skill)


class PatternAssociationsTest(unittest.TestCase):
    def memory(self, *examples, **kwargs):
        return AssociationMemory('a'*64, list(examples), **kwargs)

    def test_counts_distinguish_same_active_ids(self):
        memory = self.memory(example('a', 1), example('b', 5, 'dynamic/jump'))
        self.assertEqual(memory.decode([1]+[0]*15)['proposed_skill'], 'dynamic/run')
        self.assertEqual(memory.decode([5]+[0]*15)['proposed_skill'], 'dynamic/jump')

    def test_exact_conflict_abstains(self):
        memory = self.memory(example('a', 1), example('b', 1, 'dynamic/jump'))
        self.assertIsNone(memory.decode([1]+[0]*15)['proposed_skill'])

    def test_neighbor_ties_include_conflicting_labels(self):
        memory = self.memory(example('a', 1), example('b', 3, 'dynamic/jump'), neighbors=1)
        self.assertIsNone(memory.decode([2]+[0]*15)['proposed_skill'])

    def test_silent_and_unknown(self):
        memory = self.memory(example('a', 1))
        for counts in ([0]*16, [16]*16):
            self.assertIsNone(memory.decode(counts)['proposed_skill'])
        with self.assertRaises(ValueError):
            memory.decode([.5]*16)

    def test_eligibility_and_fallback(self):
        proposal = dict(proposed_skill='dynamic/run', reason='exact association')
        self.assertEqual(supervise(proposal, {'run': False, 'wait': True})['executed_skill'], 'dynamic/wait')
        self.assertIsNone(supervise(proposal, {'run': False, 'wait': False})['executed_skill'])
        self.assertFalse(supervise(proposal, {'run': True})['fallback'])

    def test_artifact_integrity(self):
        memory = self.memory(example('a', 1))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'association.json'
            memory.save(path)
            self.assertEqual(AssociationMemory.load(path, 'a'*64).checksum, memory.checksum)
            with self.assertRaises(ValueError):
                AssociationMemory.load(path, 'b'*64)
            with self.assertRaises(FileExistsError):
                memory.save(path)
            content = json.loads(path.read_text()); content['examples'][0]['skill'] = 'dynamic/jump'
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                AssociationMemory.load(path, 'a'*64)

    def test_progress_uses_heading_wait_is_not_task_success(self):
        import math
        summary = dict(decision_observation={'yaw': math.pi}, displacement=[-.5, 0],
                       success=True, collision=False, fall=False, max_torque_limit_ratio=1., skill='run')
        self.assertTrue(outcome(summary, False)['task_fulfilled'])
        summary['skill'] = 'wait'
        self.assertFalse(outcome(summary, False)['task_fulfilled'])

    def test_learned_execution_logs_and_interrupts(self):
        encoder = RandomLIFEncoder.load(ENCODER_PATH)
        memory = AssociationMemory.load(DEFAULT_ASSOCIATION, encoder.checksum)
        with tempfile.TemporaryDirectory() as root:
            for index, case in enumerate(situations()[:2]):
                folder = Path(root)/str(index)
                result = execute_case(case, memory, encoder, folder)
                self.assertTrue(result['outcome']['task_fulfilled'])
                self.assertEqual(result['summary']['skill'], ('run', 'jump')[index])
                events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
                self.assertEqual([e['event'] for e in events], ['start', 'decision', 'finish'])
                rows = [json.loads(line) for line in (folder/'telemetry.jsonl').read_text().splitlines()]
                self.assertGreater(len(rows), 50)
                self.assertEqual(len(rows[-1]['diagnostic_pattern']['spike_counts']), 16)
                self.assertLessEqual(result['summary']['max_torque_limit_ratio'], 1.)
                with self.assertRaises(FileExistsError):
                    execute_case(case, memory, encoder, folder)
            interrupted = execute_case(situations()[0], memory, encoder, Path(root)/'interrupted', interrupt_after=.2)
            self.assertEqual(interrupted['summary']['reason'], 'interrupted')
            self.assertFalse(interrupted['outcome']['task_fulfilled'])


if __name__ == '__main__':
    unittest.main()
