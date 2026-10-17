"""Outcome-labeled count-pattern memory, separate from the frozen SNN encoder."""
import hashlib
import json
from pathlib import Path

import numpy as np

SKILLS = ('dynamic/wait', 'dynamic/run', 'dynamic/jump')


def _checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def count_vector(counts):
    values = np.asarray(counts)
    if (values.shape != (16,) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 16) or np.any(values != np.floor(values))):
        raise ValueError('Expected 16 integer spike counts in [0,16]')
    return values.astype(np.int64)


class AssociationMemory:
    def __init__(self, encoder_checksum, examples, *, neighbors=3, max_distance=4.):
        if not isinstance(encoder_checksum, str) or len(encoder_checksum) != 64:
            raise ValueError('Require the frozen encoder checksum')
        if type(neighbors) is not int or neighbors < 1 or not np.isfinite(max_distance) or max_distance < 0:
            raise ValueError('Invalid association neighborhood')
        self.encoder_checksum = encoder_checksum
        self.examples = json.loads(json.dumps(examples, allow_nan=False))
        seen = set()
        for example in self.examples:
            count_vector(example['spike_counts'])
            if example['skill'] not in SKILLS or not example['context_id'] or example['context_id'] in seen:
                raise ValueError('Invalid skill label or duplicate training context')
            seen.add(example['context_id'])
        if not self.examples:
            raise ValueError('Require at least one verified training association')
        self.neighbors, self.max_distance = neighbors, float(max_distance)
        self.counts = np.array([e['spike_counts'] for e in self.examples], dtype=np.int64)

    def payload(self):
        return dict(schema_version=1, encoder_checksum=self.encoder_checksum, representation='16_spike_counts',
                    skills=list(SKILLS), neighbors=self.neighbors, max_count_distance=self.max_distance,
                    examples=self.examples, encoder_frozen=True)

    @property
    def checksum(self):
        return _checksum(self.payload())

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(self.payload(), checksum=self.checksum), stream, indent=2, allow_nan=False)
            stream.write('\n')

    @classmethod
    def load(cls, path, encoder_checksum):
        content = json.loads(Path(path).read_text(encoding='utf-8'))
        checksum = content.pop('checksum')
        if _checksum(content) != checksum:
            raise ValueError('Association artifact checksum mismatch')
        if (content['schema_version'] != 1 or content['representation'] != '16_spike_counts'
                or content['skills'] != list(SKILLS) or content['encoder_frozen'] is not True
                or content['encoder_checksum'] != encoder_checksum):
            raise ValueError('Association/encoder schema mismatch')
        result = cls(encoder_checksum, content['examples'], neighbors=content['neighbors'],
                     max_distance=content['max_count_distance'])
        if result.checksum != checksum:
            raise ValueError('Association artifact normalization mismatch')
        return result

    def decode(self, counts):
        query = count_vector(counts)
        if not query.any():
            return dict(proposed_skill=None, reason='silent pattern', neighbors=[])
        distances = np.linalg.norm(self.counts-query, axis=1)
        exact = np.flatnonzero(distances == 0)
        if len(exact):
            indices = exact
            reason = 'exact association'
        else:
            sorted_indices = np.argsort(distances, kind='stable')
            eligible = sorted_indices[distances[sorted_indices] <= self.max_distance]
            if not len(eligible):
                return dict(proposed_skill=None, reason='unknown pattern', neighbors=[])
            cutoff = distances[eligible[min(self.neighbors, len(eligible))-1]]
            indices = eligible[distances[eligible] <= cutoff+1e-12]
            reason = 'neighbor agreement'
        labels = {self.examples[int(i)]['skill'] for i in indices}
        references = [dict(context_id=self.examples[int(i)]['context_id'],
                           skill=self.examples[int(i)]['skill'], distance=float(distances[i])) for i in indices]
        if len(labels) != 1:
            return dict(proposed_skill=None, reason='conflicting associations', neighbors=references)
        return dict(proposed_skill=next(iter(labels)), reason=reason, neighbors=references)


def supervise(proposal, eligibility):
    """Unknown/conflicting/ineligible proposals use eligible wait, or abstain."""
    proposed = proposal['proposed_skill']
    if proposed is not None and proposed not in SKILLS:
        raise ValueError('Unsupported proposed skill')
    if proposed is not None and eligibility.get(proposed.removeprefix('dynamic/'), False):
        return dict(executed_skill=proposed, fallback=False, reason=proposal['reason'])
    reason = proposal['reason'] if proposed is None else 'proposed skill ineligible'
    if eligibility.get('wait', False):
        return dict(executed_skill='dynamic/wait', fallback=True, reason=reason)
    return dict(executed_skill=None, fallback=True, reason='no eligible fallback: '+reason)


def outcome(summary, jump_requested):
    """Measured task utility; use selection heading, not world-X progress."""
    observation = summary['decision_observation']
    yaw = observation['yaw'] if observation else 0.
    progress = float(np.asarray(summary['displacement']) @ [np.cos(yaw), np.sin(yaw)])
    valid = (summary['success'] and not summary['collision'] and not summary['fall']
             and summary['max_torque_limit_ratio'] <= 1+1e-9)
    utility = None
    fulfilled = False
    if valid:
        if jump_requested and summary['skill'] == 'jump' and summary['all_feet_airborne'] and summary['stable_landing']:
            utility, fulfilled = 10.+summary['longest_airtime'], True
        elif not jump_requested and summary['skill'] == 'run' and progress > .1:
            utility, fulfilled = 1.+progress, True
        elif summary['skill'] == 'wait':
            utility = 0.
    return dict(task='in_place_jump' if jump_requested else 'forward_progress',
                task_fulfilled=fulfilled, utility=utility, body_forward_progress=progress,
                primitive_completed=bool(summary['success']))
