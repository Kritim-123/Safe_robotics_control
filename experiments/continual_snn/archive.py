"""Archive runtime sources/assets without environment files or generated traces."""
import hashlib
import argparse
from pathlib import Path
import sys
import time
import zipfile
from .models import ROOT
from .study import save


def main(tag=None):
    if tag and (not tag.replace('-','').isalnum()):raise ValueError('Use a simple alphanumeric archive tag')
    folder=ROOT/'output/continual_snn/session-v1'
    paths=set()
    for name in ('controllers','hybrid','experiments/continual_snn','experiments/spike_pattern_skills',
                 'unitree_go2','configs','tests','docs/agents/expanded_association_v2',
                 'docs/agents/continual_snn_v1','docs/agents/continual_snn_v2'):
        for path in (ROOT/name).rglob('*'):
            if any(part in ('.git','.aws','.codex','.agents') or part.startswith('.venv') for part in path.parts):continue
            if path.name.startswith('.env') or path.suffix.lower() in ('.key','.pem','.p12','.pfx'):continue
            if name.startswith('docs/agents/continual_snn_v') and path.is_file():
                allowed={'primary.npz','association_memory.npz','compact_motion_bank16.npz','frozen_evaluation.json',
                    'frozen_selection.json','model_card.json','reproduction_data.json','training_protocol.json',
                    'fit_manifest.json','episode_qualification.json','teacher_episode_outcomes.json','matched_control_protocol.json'}
                if path.name not in allowed and path.suffix!='.gz':continue
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.mp4','.csv'):
                if not path.resolve().is_relative_to(ROOT):raise ValueError('Runtime source resolves outside project')
                paths.add(path)
    paths.update((ROOT/'skills').glob('*.py'))
    # Include transitive project imports used by the actual runtime, while
    # excluding bundled/virtual environments and private configuration.
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            relative=path.relative_to(ROOT)
            if not any(part.startswith('.venv') or part in ('.codex','.agents','.git','tmp','output') for part in relative.parts):paths.add(path)
    paths.update(ROOT/name for name in ('requirements.txt','requirements-experiments.txt') if (ROOT/name).exists())
    suffix='-'+tag if tag else ''
    archive=folder/('runtime-sources-and-assets'+suffix+'.zip')
    checksums={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for path in sorted(paths):
            rel=path.relative_to(ROOT).as_posix();data=path.read_bytes()
            checksums[rel]=hashlib.sha256(data).hexdigest();bundle.writestr(rel,data)
    report=dict(recorded_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        archive=str(archive.relative_to(ROOT)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        files=len(checksums),file_sha256=checksums,
        scope='Current selector runtime, legacy encoder/decoder source, controllers, skills, MuJoCo robot assets, configs and model checkpoints',
        note='Captured after primary freeze; frozen runtime hashes are separately verified; generated traces, Python environment and private configuration are excluded')
    save(folder/('runtime-sources'+suffix+'.json'),report)
    save(ROOT/'docs/agents/continual_snn_v1/runtime_sources.json',report)
    if (ROOT/'docs/agents/continual_snn_v2').exists():save(ROOT/'docs/agents/continual_snn_v2/runtime_sources.json',report)
    print('Archived',len(checksums),'files;',round(archive.stat().st_size/1024/1024,2),'MiB',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag')
    main(parser.parse_args().tag)
