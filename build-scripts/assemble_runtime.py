"""Put the same current-source native package into two real Git checkouts."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

root = Path('/work')
native = root / 'package/native'
commit = 'eb34e115d041ac685fa38eadf8eae6c51d61dc23'
builds = ['core-build.json', 'flashmla-build.json', 'deepgemm-build.json', 'flashattn-build.json']
artifact_hashes = {}
for name in builds:
    record = json.loads((root / 'receipts' / name).read_text())
    assert record['source_commit'] == commit and record.get('finished_at'), name
    assert all(step.get('exit_code') == 0 for step in record['commands']), name
    for relative, digest in record['artifacts'].items():
        assert relative not in artifact_hashes or artifact_hashes[relative] == digest, relative
        artifact_hashes[relative] = digest
assert (native / 'vllm/_C_stable_libtorch.abi3.so').is_file()
assert (native / 'vllm/_flashmla_C.abi3.so').is_file()
assert (native / 'vllm/third_party/deep_gemm/_C_extension.abi3.so').is_file()
assert list((native / 'vllm/vllm_flash_attn').glob('_vllm_fa3_C*.so'))
files = []
source_files = {}
for arm in ('baseline', 'tuned'):
    repository = root / 'assembled' / arm
    assert subprocess.check_output(['git', '-c', 'safe.directory=' + str(repository), 'rev-parse', 'HEAD'], cwd=repository, text=True).strip() == commit
for path in sorted(native.rglob('*')):
    assert not path.is_symlink(), 'runtime must not depend on a build-only symlink: ' + str(path)
    if not path.is_file():
        continue
    relative = path.relative_to(native).as_posix()
    # These targets were installed once and were not modified by later phases.
    # Reuse their recorded artifact identities; the GPU admission rechecks copies.
    digest = artifact_hashes[relative]
    files.append({'path': '/native-runtime/package/native/' + relative, 'sha256': digest})
    source_files[relative] = digest
    for arm in ('baseline', 'tuned'):
        repository = root / 'assembled' / arm
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
target = 'vllm/model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json'
baseline_config = root / 'assembled/baseline' / target
assert baseline_config.is_file()
baseline_config.unlink()
receipt = {
    'schema_version': 1, 'source_commit': commit,
    'runtime_bind': '/data03/tiger/vllm_pr58986_native_20261028_eb34e115:/native-runtime:ro',
    'versions': {
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'triton': metadata.version('triton'), 'python': sys.version,
        'driver': (root / 'receipts/host-driver-version.txt').read_text().strip(),
        'build_receipt': ['/native-runtime/receipts/' + name for name in builds],
        'image_id': json.loads(Path('/common/manifest.json').read_text())['image_id'],
        'dependency_manifest_sha256': hashlib.sha256(Path('/common/manifest.json').read_bytes()).hexdigest(),
    },
    'pins': {
        'vllm': commit,
        'flashmla': 'e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b',
        'flashmla_cutlass': '4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b',
        'flashkda': '17a037d98da546deb4591e967cf961a43c034d8b',
        'flashkda_cutlass': '5c149f52a436782210263fb2f19b354443a61c6a',
        'deepgemm': '0820030756945b2daec208e8e8255268360f337d',
        'deepgemm_cutlass': 'f3fde58372d33e9a5650ba7b80fc48b3b49d40c8',
        'deepjit': '3732a3b9a0a2e5126396908a362ba5abdf043b2f',
        'flashattention': 'b273a02621d3c94bc747df67204f591b67f38fae',
        'flashattention_cutlass': '62750a2b75c802660e4894434dc55e839f322277',
        'core_cutlass_revision': 'v4.7.1',
    },
    'files': files, 'source_files': source_files,
    'arms': {arm: {'repo': '/native-runtime/assembled/' + arm,
                   'package_root': '/native-runtime/assembled/' + arm + '/vllm'}
             for arm in ('baseline', 'tuned')},
    'limitation': 'Build and CPU import/schema receipts do not establish CUDA execution, numerical correctness, or model readiness.',
}
assert not torch.cuda.is_initialized()
(root / 'receipts/native-receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps({'artifact_count': len(files), 'native_receipt': '/native-runtime/receipts/native-receipt.json',
                  'arms': receipt['arms'], 'cuda_initialized': torch.cuda.is_initialized()}, indent=2))
