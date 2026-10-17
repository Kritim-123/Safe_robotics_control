"""Grouped historical outcome and newly collected causal-teacher observations."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .models import ROOT,BUNDLE
from .encoding import PacketEncoder


def static_records():
    model=PacketEncoder();encoder=model.base
    bank=json.loads((BUNDLE/'dense.json').read_text())
    lookup={e['context_id']:e for e in bank['examples']}
    records=[]
    source=ROOT/'output/spike_pattern_skills/expanded-associations-v1/reused_training.jsonl'
    for line in source.read_text().splitlines():
        old=json.loads(line);group='original/'+old['example']['context_id']
        features=dict(old['observation']['features'])
        from experiments.spike_pattern_skills.terrain_encoding import TERRAIN_FEATURES
        features.update({name:0. for name in TERRAIN_FEATURES})
        assert encoder.encode(features)['spike_counts']==lookup[group]['spike_counts']
        records.append(dict(group=group,split='train',features=features,label=lookup[group]['skill'],
            label_source='historical_measured_skill_outcome',source='static'))
    for line in (BUNDLE/'reencoded_labels.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['skill'] is None:continue
        group=row['context_id'];features=row['dense_observation']['features']
        assert encoder.encode(features)['spike_counts']==row['dense_counts']
        if row['split']=='train':assert lookup[group]['skill']==row['skill']
        records.append(dict(group=group,split=row['split'],features=features,label=row['skill'],
            label_source='historical_measured_skill_outcome',source='static'))
    assert sum(r['split']=='train' for r in records)==94
    return records


def stream_records(folder,*,split):
    records=[]
    for path in sorted(Path(folder).glob('*/*/decisions.jsonl')):
        scenario=json.loads((path.parent/'scenario.json').read_text())
        group=scenario['scenario']['name']
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row['teacher_request'] is None:continue
            records.append(dict(group=group,split=split,features=row['features'],label=row['teacher_request'],
                label_source='causal_geometry_task_teacher_not_outcome',source='stream',time=row['time']))
    return records


def save_records(path,records):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        for row in records:stream.write(json.dumps(row,allow_nan=False)+'\n')
    summary=dict(rows=len(records),groups=len({r['group'] for r in records}),
        labels=dict(Counter(r['label'] for r in records)),sources=dict(Counter(r['label_source'] for r in records)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
