"""Check published training group IDs against declared final episode IDs."""
import argparse,gzip,json,hashlib
from pathlib import Path
from experiments.continual_snn.study import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('output/terrain_goal');release=Path('docs/agents/terrain_goal_v1');groups=set();sources={}
    for name in ('base_training_records.jsonl.gz','goal_count_training_records.jsonl.gz'):
        path=release/name
        with gzip.open(path,'rt') as f:rows=[json.loads(s) for s in f]
        current={r['group'] for r in rows};groups.update(current)
        sources[name]=dict(rows=len(rows),groups=len(current),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    refined=Path('docs/agents/terrain_goal_v3/training_records.jsonl.gz')
    if refined.exists():
        with gzip.open(refined,'rt') as f:records=[json.loads(s) for s in f]
        current={r['group'] for r in records};groups.update(current)
        sources[str(refined)]=dict(rows=len(records),groups=len(current),sha256=hashlib.sha256(refined.read_bytes()).hexdigest())
    train_names={g.split('/')[-1] for g in groups}
    train_seeds={int(g.split('/')[1]) for g in groups if '/' in g and g.split('/')[1].isdigit()}
    tests={}
    for pattern in ('final-main-chunk-*','hysteresis-final-*-v2','final-ood-v1',
                    'request-refinement-confirmation-v3-*','request-refinement-noise-v3'):
        for batch in sorted(root.glob(pattern)):
            path=batch/'cases.json'
            if not path.exists():continue
            cases=json.loads(path.read_text());names={c['scenario']['name'] for c in cases};seeds={c['seed'] for c in cases}
            tests[batch.name]=dict(planned_episodes=len(cases),training_name_overlap=sorted(names&train_names),
                explicit_training_seed_overlap=sorted(seeds&train_seeds),
                cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    passed=all(not r['training_name_overlap'] and not r['explicit_training_seed_overlap'] for r in tests.values())
    save(a.output,dict(passed=passed,training=sources,final_manifests=tests,
        limitations=['Checks declared group names and explicit training/seed/name IDs, not proof of statistical independence',
            'Terrain templates are shared; mild/clean conditions intentionally repeat scenes',
            'Original historical 60-case benchmark informed development and is excluded from a fresh-test claim']))
    if not passed:raise AssertionError('Declared training/test episode IDs overlap')
    print('No declared training/final episode ID overlap in',len(tests),'manifests')


if __name__=='__main__':main()
