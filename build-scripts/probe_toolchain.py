"""Record the fixed image toolchain and verify every build write path."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

root = Path('/work')
record = {
    'python': sys.executable,
    'python_version': sys.version,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'torch_path': torch.__file__,
    'cuda_initialized': torch.cuda.is_initialized(),
    'tools': {},
    'write_paths': {},
    'mountinfo': Path('/proc/self/mountinfo').read_text(),
}
for name, arguments in {
    'nvcc': ['--version'], 'cmake': ['--version'], 'ninja': ['--version'],
    'g++': ['--version'], 'g++-13': ['--version'], 'g++-14': ['--version'],
    'ccache': ['--version'], 'uv': ['--version'],
}.items():
    executable = shutil.which(name)
    record['tools'][name] = {'path': executable}
    if executable:
        completed = subprocess.run([executable, *arguments], capture_output=True, text=True)
        record['tools'][name].update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
for name in ('sources', 'build', 'package', 'logs', 'receipts', 'tmp', 'cache'):
    directory = root / name
    directory.mkdir(exist_ok=True)
    probe = directory / '.write-probe'
    probe.write_text('native-build-write-probe\n')
    record['write_paths'][name] = {'path': str(directory), 'device': probe.stat().st_dev, 'statvfs': list(os.statvfs(directory))}
    assert probe.read_text() == 'native-build-write-probe\n'
    probe.unlink()
assert len({p['device'] for p in record['write_paths'].values()}) == 1
assert not torch.cuda.is_initialized()
(root / 'receipts/toolchain.json').write_text(json.dumps(record, indent=2))
print(json.dumps({k: v for k, v in record.items() if k not in ('mountinfo', 'write_paths')}, indent=2))
