"""Verify pinned rl_sar assets and export numeric NumPy inference packages.

Run after downloading the pinned upstream files into a source directory:
python -m skills.import_pretrained --source tmp/rl_sar_upstream
Torch is used only for conversion and parity checks, never by runtime inference.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

REVISION = '376d42c9b128f963ab08579762d5a216a976ce39'
PACKAGE_ROOT = Path(__file__).resolve().parent/'policies'/'rl_sar'
SOURCE_HASHES = {
    'LICENSE': '6c18491053290fe1bada9775674b2b14493c4f0fdeab41e013dd5d4c82461d57',
    'policy/go2/base.yaml': '78e9b9c2be72544ab334362535ab876630808f87170ec3e38c56f9817cd53d02',
    'policy/go2/robot_lab/config.yaml': 'b8fbd540597b9f326fd4acb83d97d72306fb9bbc8ef84e1d8e350871b4de4e37',
    'policy/go2/robot_lab/policy.pt': '9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b',
    'policy/go2/himloco/config.yaml': '67179d241385bc142ffe02d0b1c04899f46f2ae9f9eef6e57935a6f57c5430c5',
    'policy/go2/himloco/himloco.pt': 'd9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_config(text):
    """Parse only the scalar/list subset in these hash-verified source configs."""
    fields = re.findall(r'^  (\w+):\s*(.*?)(?=^  \w+:|\Z)',
                        re.sub(r'#[^\n]*', '', text), flags=re.M | re.S)
    return {key: ast.literal_eval(value.strip()) for key, value in fields}


def export(source, output=PACKAGE_ROOT):
    import torch
    from skills.pretrained_velocity import numeric_forward
    torch.set_num_threads(1)
    source, output = Path(source).resolve(), Path(output).resolve()
    for file, expected in SOURCE_HASHES.items():
        if sha256(source/file) != expected:
            raise ValueError(f'Pinned source checksum mismatch: {file}')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'LICENSE', output/'LICENSE')
    shutil.copyfile(source/'policy/go2/base.yaml', output/'source_base.yaml')
    base = pinned_config((source/'policy/go2/base.yaml').read_text())
    results = {}
    for name, filename in [('robot_lab', 'policy.pt'), ('himloco', 'himloco.pt')]:
        folder = output/name
        folder.mkdir(parents=True, exist_ok=True)
        upstream = source/'policy/go2'/name
        shutil.copyfile(upstream/filename, folder/'source_policy.pt')
        shutil.copyfile(upstream/'config.yaml', folder/'source_config.yaml')
        config = pinned_config((upstream/'config.yaml').read_text())
        model = torch.jit.load(str(folder/'source_policy.pt'), map_location='cpu').eval()
        arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        # These exports are feedforward ELU actors, HIM additionally includes
        # its estimator and L2-normalized 16D latent. Reject a changed graph.
        expected_dims = {'actor': [45, 512, 256, 128, 12]} if name == 'robot_lab' else {
            'actor': [64, 512, 256, 128, 12], 'estimator': [270, 128, 64, 19]}
        for prefix, dims in expected_dims.items():
            net = getattr(model, prefix)
            for i in range(len(dims)-1):
                assert getattr(net, str(2*i)).original_name == 'Linear'
                assert arrays[f'{prefix}.{2*i}.weight'].shape == (dims[i+1], dims[i])
                if i < len(dims)-2:
                    assert getattr(net, str(2*i+1)).original_name == 'ELU'
        if name == 'robot_lab':
            assert model.normalizer.original_name == 'Identity'
        np.savez_compressed(folder/'model.npz', **arrays)
        rng = np.random.default_rng(604)
        inputs = rng.normal(0, .5, (128, 45 if name == 'robot_lab' else 270)).astype(np.float32)
        with torch.inference_mode():
            reference = model(torch.from_numpy(inputs)).numpy()
        converted = np.stack([numeric_forward(arrays, name, x) for x in inputs])
        error = float(np.max(np.abs(reference-converted)))
        np.testing.assert_allclose(converted, reference, rtol=2e-5, atol=1e-5)
        config['joint_names'] = [base['joint_names'][i] for i in config['joint_mapping']]
        manifest = dict(schema_version=1, policy=name,
                        upstream='https://github.com/fan-ziqi/rl_sar', revision=REVISION,
                        source_directory=f'policy/go2/{name}', license='Apache-2.0',
                        original_files={p: h for p, h in SOURCE_HASHES.items()
                                        if f'/{name}/' in p or p in ('LICENSE', 'policy/go2/base.yaml')},
                        artifacts={f: sha256(folder/f) for f in ('source_policy.pt', 'source_config.yaml', 'model.npz')},
                        control_period_s=.02, source_physics_period_s=.005,
                        project_physics_period_s=.002, config=config,
                        source_torch_version=torch.__version__, export_seed=604,
                        export_samples=128, export_max_abs_error=error,
                        conversion='Unchanged numeric tensors; feedforward ELU and HIM estimator in NumPy',
                        validation='See validation.json; not eligible for the existing SNN output schemas')
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        results[name] = error
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PACKAGE_ROOT)
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))
