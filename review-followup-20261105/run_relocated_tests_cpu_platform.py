"""Run the relocated GLM regression cases in the real current-source module."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path('/work/results-cpu-platform')
out.mkdir(exist_ok=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void'
child = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import pytest
import torch
import vllm
from vllm.platforms import current_platform
assert current_platform.is_cpu()
from benchmarks.kernels import benchmark_moe
from transformers import Glm5NextConfig, Glm5NextTextConfig
assert str(Path(vllm.__file__).resolve()).startswith('/subject/vllm/')
assert str(Path(benchmark_moe.__file__).resolve()).startswith('/subject/benchmarks/')
assert not torch.cuda.is_initialized()
print('IMPORT_EVIDENCE ' + json.dumps({
    'python': sys.version if 'sys' in globals() else __import__('sys').version,
    'executable': __import__('sys').executable,
    'vllm': vllm.__file__, 'benchmark': benchmark_moe.__file__,
    'constructors': [Glm5NextConfig.__module__, Glm5NextTextConfig.__module__],
    'versions': {n: metadata.version(n) for n in ['torch','triton','transformers','ray','pytest']},
    'cuda_initialized_before': torch.cuda.is_initialized(),
    'platform': type(current_platform).__module__ + '.' + type(current_platform).__qualname__,
    'VLLM_TARGET_DEVICE': __import__('os').environ['VLLM_TARGET_DEVICE'],
}), flush=True)
rc = pytest.main([
    '/subject/tests/kernels/moe/test_moe.py', '-k', 'glm5next',
    '--confcutdir=/subject/tests/kernels/moe', '-p', 'no:cacheprovider',
    '-vv', '--tb=short', '--junitxml=/work/results-cpu-platform/pytest.xml',
])
print('CUDA_INITIALIZED_AFTER ' + str(torch.cuda.is_initialized()), flush=True)
assert not torch.cuda.is_initialized()
raise SystemExit(rc)
'''
env = dict(os.environ, PYTHONPATH='/subject:/common/site-packages:/work', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', VLLM_TARGET_DEVICE='cpu')
command = [sys.executable, '-B', '-c', child]
(out/'command.json').write_text(json.dumps({'command':command,'cwd':'/work','environment':{'PYTHONPATH':env['PYTHONPATH'],'PYTEST_DISABLE_PLUGIN_AUTOLOAD':env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'],'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'NVIDIA_VISIBLE_DEVICES':env['NVIDIA_VISIBLE_DEVICES'],'VLLM_TARGET_DEVICE':env['VLLM_TARGET_DEVICE']}}, indent=2)+'\n')
result = subprocess.run(command, cwd='/work', env=env, capture_output=True, text=True)
(out/'stdout.txt').write_text(result.stdout)
(out/'stderr.txt').write_text(result.stderr)
(out/'exit.json').write_text(json.dumps({'returncode':result.returncode},indent=2)+'\n')
print(result.stdout[-10000:], flush=True)
print(result.stderr[-10000:], file=sys.stderr, flush=True)
raise SystemExit(result.returncode)
