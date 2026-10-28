"""Add the fixed-current DeepGEMM host extension and vendored JIT inputs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/work')
source = root / 'sources/pr-repo'
driver = root / 'driver-core'
recipe = driver / 'CMakeLists.txt'
text = recipe.read_text()
include = '\ninclude("/work/sources/pr-repo/cmake/external_projects/deepgemm.cmake")\n'
if include not in text:
    recipe.write_text(text + include)
(root / 'receipts/deepgemm-driver-CMakeLists.txt').write_text(recipe.read_text())
environment = dict(os.environ)
environment.update(TORCH_CUDA_ARCH_LIST='9.0', CUDA_HOME='/usr/local/cuda',
                   FLASH_MLA_SRC_DIR='/work/sources/FlashMLA',
                   VLLM_CUTLASS_SRC_DIR='/dependencies/cutlass-v4.7.1',
                   FLASH_KDA_SRC_DIR='/dependencies/flashkda-17a037d98da546deb4591e967cf961a43c034d8b',
                   DEEPGEMM_SRC_DIR='/work/sources/DeepGEMM-0820030756945b2daec208e8e8255268360f337d',
                   DG_JIT_CACHE_DIR='/work/cache/deepgemm', DG_JIT_PRINT_COMPILER_COMMAND='1',
                   MAX_JOBS='16', TMPDIR='/work/tmp', TMP='/work/tmp', TEMP='/work/tmp',
                   CUDA_CACHE_PATH='/work/cache/cuda', CCACHE_DIR='/work/cache/ccache')
commands = [
    ['cmake', '-S', str(driver), '-B', '/work/build/core', '-G', 'Ninja',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
     '-DVLLM_PYTHON_EXECUTABLE=' + sys.executable,
     '-DCMAKE_INSTALL_PREFIX=/work/package/native', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
     '-DCMAKE_CXX_COMPILER=/opt/byted/toolchain/bin/g++', '-DNVCC_THREADS=1'],
    ['cmake', '--build', '/work/build/core', '--parallel', '16', '--target', '_deep_gemm_C'],
    ['cmake', '--install', '/work/build/core', '--component', '_deep_gemm_C'],
]
receipt = {'source_commit': 'eb34e115d041ac685fa38eadf8eae6c51d61dc23',
           'deepgemm_commit': '0820030756945b2daec208e8e8255268360f337d',
           'deepjit_commit': '3732a3b9a0a2e5126396908a362ba5abdf043b2f',
           'cutlass_commit': 'f3fde58372d33e9a5650ba7b80fc48b3b49d40c8',
           'scope': 'Current DeepGEMM host extension and the matching Python package/JIT include files',
           'commands': [], 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source/'cmake/external_projects/deepgemm.cmake', source/'tools/build_deepgemm_C.py')},
           'environment': {k: environment[k] for k in ('TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'DEEPGEMM_SRC_DIR', 'DG_JIT_CACHE_DIR', 'TMPDIR')}}
destination = root / 'receipts/deepgemm-build.json'
for index, command in enumerate(commands):
    log_path = root / ('logs/deepgemm-step-' + str(index) + '.log')
    entry = {'command': command, 'log': str(log_path)}
    receipt['commands'].append(entry)
    destination.write_text(json.dumps(receipt, indent=2))
    print('Running: ' + repr(command), flush=True)
    with log_path.open('w') as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    entry['exit_code'] = completed.returncode
    destination.write_text(json.dumps(receipt, indent=2))
    if completed.returncode:
        print(log_path.read_text()[-18000:], flush=True)
        raise SystemExit(completed.returncode)
receipt['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt['artifacts'] = {str(p.relative_to(root / 'package/native')): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root / 'package/native/vllm/third_party/deep_gemm').rglob('*') if p.is_file()}
destination.write_text(json.dumps(receipt, indent=2))
print(json.dumps({k:v for k,v in receipt.items() if k != 'artifacts'}, indent=2))
