"""Build only fixed-current FlashMLA targets using the unmodified target recipe."""
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
driver.mkdir(exist_ok=True)
for name in ('cmake', 'csrc', 'tools'):
    link = driver / name
    if not link.exists():
        link.symlink_to(source / name, target_is_directory=True)
original = (source / 'CMakeLists.txt').read_text()
prefix = original[:original.index('# For CUDA and HIP builds also build the triton_kernels external package.')]
(root / 'receipts/core-driver-CMakeLists.txt').write_text((driver / 'CMakeLists.txt').read_text())
(driver / 'CMakeLists.txt').write_text(prefix + '\ninclude("/work/sources/pr-repo/cmake/external_projects/flashkda.cmake")\ninclude("/work/sources/pr-repo/cmake/external_projects/flashmla.cmake")\n')
(root / 'receipts/flashmla-driver-CMakeLists.txt').write_text((driver / 'CMakeLists.txt').read_text())
environment = dict(os.environ)
environment.update(TORCH_CUDA_ARCH_LIST='9.0', CUDA_HOME='/usr/local/cuda',
                   FLASH_MLA_SRC_DIR='/work/sources/FlashMLA',
                   VLLM_CUTLASS_SRC_DIR='/dependencies/cutlass-v4.7.1',
                   FLASH_KDA_SRC_DIR='/dependencies/flashkda-17a037d98da546deb4591e967cf961a43c034d8b',
                   TMPDIR='/work/tmp', TMP='/work/tmp', TEMP='/work/tmp',
                   CUDA_CACHE_PATH='/work/cache/cuda', CCACHE_DIR='/work/cache/ccache')
commands = [
    ['cmake', '-S', str(driver), '-B', '/work/build/core', '-G', 'Ninja',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
     '-DVLLM_PYTHON_EXECUTABLE=' + sys.executable,
     '-DCMAKE_INSTALL_PREFIX=/work/package/native', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
     '-DCMAKE_CXX_COMPILER=/opt/byted/toolchain/bin/g++', '-DNVCC_THREADS=1'],
    ['cmake', '--build', '/work/build/core', '--parallel', '16', '--target', '_flashmla_C', '_flashmla_extension_C'],
    ['cmake', '--install', '/work/build/core', '--component', '_flashmla_C'],
    ['cmake', '--install', '/work/build/core', '--component', '_flashmla_extension_C'],
]
receipt = {'source_commit': 'eb34e115d041ac685fa38eadf8eae6c51d61dc23',
           'flashmla_commit': 'e2afa5fd8c220fbd05fe03fc24c9d5e64c24641b',
           'scope': 'Incremental FlashMLA targets added after the exact current main native CMake sections; reuses the completed current core build cache',
           'commands': [], 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_files': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (source/'CMakeLists.txt', source/'cmake/utils.cmake', source/'cmake/external_projects/flashmla.cmake')},
           'environment': {k: environment[k] for k in ('TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'FLASH_MLA_SRC_DIR', 'TMPDIR', 'CUDA_CACHE_PATH', 'CCACHE_DIR')}}
destination = root / 'receipts/flashmla-build.json'
for index, command in enumerate(commands):
    log_path = root / ('logs/flashmla-step-' + str(index) + '.log')
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
                        for p in (root / 'package/native').rglob('*') if p.is_file()}
destination.write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
