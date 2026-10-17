"""Archive explicit project sources/assets and verify immutable release hashes."""
import argparse,hashlib,json,zipfile,gzip,platform
from pathlib import Path
from importlib.metadata import distributions
from experiments.continual_snn.models import ROOT
from experiments.continual_snn.study import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sources=set();verification={}
    versions=['terrain_goal_v1','terrain_goal_v2']
    if (ROOT/'docs/agents/terrain_goal_v3/frozen_selection.json').exists():versions.append('terrain_goal_v3')
    for version in versions:
        manifest=json.loads((ROOT/'docs/agents'/version/'frozen_selection.json').read_text())
        recorded=dict(manifest['source_sha256']);recorded.update(manifest.get('transitive_source_and_asset_sha256',{}))
        verification[version]={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in recorded.items()}
        primary=ROOT/'docs/agents'/version/'primary.npz'
        verification[version][str(primary.relative_to(ROOT))]=hashlib.sha256(primary.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
        data_manifest=primary.parent/'training_data_manifest.json'
        if data_manifest.exists():
            for name,info in json.loads(data_manifest.read_text())['files'].items():
                path=primary.parent/name
                verification[version][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
        if 'training_sha256' in manifest:
            path=primary.parent/'training_records.jsonl.gz'
            verification[version][str(path.relative_to(ROOT))+' decompressed']=hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()==manifest['training_sha256']
        sources.update(ROOT/name for name in recorded)
    session=json.loads((ROOT/'output/terrain_goal/session.json').read_text())
    verification['preserved_baselines']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        for name,digest in session['preserved_sha256'].items()}
    if any(not valid for group in verification.values() for valid in group.values()):
        raise AssertionError('Frozen source or preserved baseline changed')
    for folder in ['experiments/terrain_goal']+['docs/agents/'+v for v in versions]:
        sources.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources.update(p for p in (ROOT/'docs/agents/continual_snn_v2').iterdir() if p.is_file())
    sources.update((ROOT/'tests').glob('test_terrain_goal*.py'))
    sources.update((ROOT/'docs/figures').glob('terrain-goal-*'))
    sources.update(ROOT/name for name in ('requirements.txt','requirements-experiments.txt',
        'docs/agents/terrain_goal_protocol.md','docs/agents/terrain_goal_hysteresis_protocol.md',
        'docs/agents/terrain_goal_request_refinement_protocol.md',
        'docs/agents/terrain_goal_validation.md','unitree_go2/LICENSE'))
    sources={p for p in sources if p.is_file()}
    file_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    with zipfile.ZipFile(a.output/'source-and-assets.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(sources):archive.write(path,str(path.relative_to(ROOT)))
        archive.writestr('archive-sha256.json',json.dumps(file_hashes,indent=2))
    # Names/versions only: do not capture environment variables, credentials or package URLs.
    versions=sorted([dict(name=d.metadata['Name'],version=d.version) for d in distributions()],key=lambda d:d['name'].lower())
    save(a.output/'verification.json',dict(hash_verification=verification,files=file_hashes,
        archive_sha256=hashlib.sha256((a.output/'source-and-assets.zip').read_bytes()).hexdigest(),
        included_files=len(sources),installed_package_versions=versions,
        python_version=platform.python_version(),platform=platform.platform(),
        scope='Explicit frozen sources and assets, new milestone code/tests/models/training, report and protocols; no environment or Git directory'))
    print('Verified',sum(map(len,verification.values())),'frozen hashes; archived',len(sources),'files',flush=True)


if __name__=='__main__':main()
