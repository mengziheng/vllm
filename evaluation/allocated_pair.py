#!/usr/bin/env python3
"""Create a fixed pair manifest only after the pool has allocated four visible H20s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import compare as c
import run_pair as runner


def write_fixed(path, value):
    path = Path(path)
    if path.exists():
        c.require(c.read(path) == value, 'fixed input changed; use a new pair output: ' + str(path))
    else:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)


def cache_inventory(source):
    result = []
    for path in sorted(Path(source).rglob('*')):
        stat = path.lstat()
        if path.is_symlink():
            result.append([str(path.relative_to(source)), 'symlink', os.readlink(path)])
        elif path.is_file():
            result.append([str(path.relative_to(source)), 'file', stat.st_size, stat.st_mtime_ns])
    return result


def copy_cache(source, destination):
    source, destination = Path(source), Path(destination)
    inventory = cache_inventory(source)
    c.require(inventory, 'prepopulated official cache is empty')
    identity = {'source': str(source.resolve()),
                'source_inventory_sha256': hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest(),
                'source_file_bytes': sum(row[2] for row in inventory if row[1] == 'file'),
                'symlinks_preserved': True}
    receipt = destination.with_name(destination.name + '.copy.json')
    if destination.exists():
        c.require(receipt.is_file() and c.read(receipt) == identity, 'existing cache copy has different/uncompleted inputs')
        return identity
    partial = destination.with_name(destination.name + '.partial')
    owner = destination.with_name(destination.name + '.copy-owner.json')
    if partial.exists():
        c.require(owner.is_file() and c.read(owner) == identity, 'unowned partial cache directory')
        shutil.rmtree(partial)
    write_fixed(owner, identity)
    shutil.copytree(source, partial, symlinks=True)
    write_fixed(receipt, identity)
    partial.replace(destination)
    return identity


def read_only_mount(path):
    path = Path(path).resolve()
    candidates = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[4].replace('\\040', ' '))
        if path == mount or path.is_relative_to(mount):
            candidates.append((len(mount.parts), fields[5].split(','), str(mount)))
    c.require(candidates, 'cannot identify actual input mount: ' + str(path))
    _, options, mount = max(candidates)
    c.require('ro' in options, 'input must actually be read-only: ' + str(path))
    return {'path': str(path), 'mount': mount, 'options': options}


def runtime_environment(args, root, client=False):
    # Only image variables needed for executables/CUDA loading may enter the job.
    allowed = ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'CUDACXX')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.setdefault('PATH', '/usr/local/bin:/usr/bin:/bin')
    env.update(HOME=str(root / 'home'), LANG='C.UTF-8', LC_ALL='C.UTF-8',
               PYTHONCOERCECLOCALE='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               VLLM_PLUGINS='', VLLM_BATCH_INVARIANT='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               HF_HOME=str(root / 'hf-cache'), HF_HUB_CACHE=str(root / 'hf-cache/hub'),
               HF_DATASETS_CACHE=str(root / 'hf-cache/datasets'),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               CUDA_VISIBLE_DEVICES='' if client else os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    dependencies = [args.client, *args.dependency_path]
    if client:
        dependencies.insert(0, args.client_source_repo or args.tuned_repo)
    env['PYTHONPATH'] = ':'.join(dependencies)
    for key, subdir in {'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'xdg', 'XDG_CONFIG_HOME': 'config',
                        'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
                        'VLLM_CACHE_ROOT': 'vllm', 'CUDA_CACHE_PATH': 'cuda',
                        'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch-extensions',
                        'CCACHE_DIR': 'ccache', 'DG_JIT_CACHE_DIR': 'deep-gemm', 'DJ_JIT_CACHE_DIR': 'deep-jit',
                        'FLASHINFER_WORKSPACE_BASE': 'flashinfer', 'CUTE_DSL_CACHE_DIR': 'cute',
                        'TVM_CACHE_DIR': 'tvm', 'HF_MODULES_CACHE': 'hf-modules'}.items():
        env[key] = str(root / 'cache' / subdir)
    env['TMP'] = env['TEMP'] = env['TMPDIR']
    for key in ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'CUDA_CACHE_PATH', 'TORCH_HOME',
                'TORCH_EXTENSIONS_DIR', 'CCACHE_DIR', 'DG_JIT_CACHE_DIR', 'DJ_JIT_CACHE_DIR',
                'FLASHINFER_WORKSPACE_BASE', 'CUTE_DSL_CACHE_DIR', 'TVM_CACHE_DIR', 'HF_MODULES_CACHE'):
        path = Path(env[key])
        path.mkdir(parents=True, exist_ok=True)
        c.require(path.stat().st_dev == Path(args.data_root).stat().st_dev, 'cache directory is outside actual data filesystem: ' + key)
    c.require_nonsecret_environment(env)
    return env


def discover_allocation(env):
    visible = env.get('CUDA_VISIBLE_DEVICES', '').split(',')
    c.require(len(visible) == 4 and len(set(visible)) == 4, 'pool must allocate exactly four visible GPUs')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
                                  env=env, text=True, timeout=30)
    lines = [[part.strip() for part in line.split(',')] for line in raw.splitlines() if line.strip()]
    selected = [lines[int(device)] if device.isdigit() else next(row for row in lines if row[1] == device) for device in visible]
    c.require(len({row[1] for row in selected}) == 4 and all('H20' in row[0].split() for row in selected), 'actual allocated devices are not four H20s')
    c.require(len({row[2] for row in selected}) == 1, 'selected GPU driver versions differ')
    return {'gpu_name': selected[0][0], 'gpu_uuids': [row[1] for row in selected],
            'driver_version': selected[0][2], 'raw_nvidia_smi': raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime-root', 'baseline-repo', 'tuned-repo', 'native-receipt', 'model', 'model-receipt', 'client', 'cache', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--client-source-repo')
    parser.add_argument('--dependency-path', action='append', default=[])
    parser.add_argument('--data-root', default='/work')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--budget-gib', type=int, default=32)
    parser.add_argument('--headroom-gib', type=int, default=20)
    parser.add_argument('--pair-id', default='glm53-h20-pr58986-eb34e115-native-full')
    parser.add_argument('--print-plan', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps({'status': 'plan-only', 'gpu_validation': 'deferred until actual allocation',
                          'arguments': vars(args), 'shared_server_args': c.read(Path(__file__).with_name('manifest.example.json'))['shared']['server_args'],
                          'sequence': ['actual allocation/storage/model receipt', 'copy small offline cache',
                          'fixed client source proof', 'baseline health and full workloads', 'stop baseline', 'tuned health and full workloads']}, indent=2))
        return 0
    runner.install_signal_handlers()
    c.require(args.dependency_path, 'explicit dependency PYTHONPATH entries required')
    root = Path(args.output).resolve()
    data = Path(args.data_root).resolve()
    c.require(root.is_relative_to(data), 'output must be under actual task data root')
    root.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get('OPERATOR_SWEEP_RUN_ID')
    c.require(run_id, 'must run through current pool launcher')
    launcher = data / 'unified-environment' / (run_id + '.json')
    c.require(launcher.is_file(), 'current launcher receipt missing')
    pointer = root / 'current-launcher-receipt.json'
    tmp_pointer = root / 'current-launcher-receipt.tmp'
    if tmp_pointer.is_symlink():
        tmp_pointer.unlink()
    tmp_pointer.symlink_to(launcher)
    tmp_pointer.replace(pointer)
    mounts = [read_only_mount(path) for path in (args.runtime_root, args.model, args.model_receipt, args.client, args.cache, Path(__file__).parent)]
    allocation_env = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES') if key in os.environ}
    allocation = discover_allocation(allocation_env)
    write_fixed(root / ('allocation-' + run_id + '.json'), {'allocation': allocation, 'read_only_inputs': mounts,
                                                          'launcher_receipt': str(launcher)})
    cache_copy = copy_cache(args.cache, root / 'hf-cache')
    model_identity = c.model_snapshot(args.model, args.model_receipt)
    m = c.read(Path(__file__).with_name('manifest.example.json'))
    m['pair_id'] = args.pair_id
    m['shared']['native_receipt_sha256'] = c.sha256(args.native_receipt)
    m['shared']['model_revision'] = 'structural-receipt-sha256:' + c.sha256(args.model_receipt)
    m['shared']['hardware'] = {key: allocation[key] for key in ('gpu_name', 'gpu_uuids', 'driver_version')}
    m['model_identity'] = model_identity
    m['model_structural_receipt'] = args.model_receipt
    m['offline_cache_copy'] = cache_copy
    server_env = runtime_environment(args, root)
    for arm, repo in (('baseline', args.baseline_repo), ('tuned', args.tuned_repo)):
        m['arms'][arm].update(repo=repo, package_root=str(Path(repo) / 'vllm'), python=args.python,
                              env=dict(server_env), proof=str(root / arm / 'proof.json'))
    m.update(client_python=args.python, client_source_repo=args.client_source_repo or args.tuned_repo,
             client_package_root=str(Path(args.client_source_repo or args.tuned_repo) / 'vllm'),
             client_env=runtime_environment(args, root, client=True), dataset_cache=str(root / 'hf-cache'),
             hf_hub_cache=str(root / 'hf-cache/hub'), hf_datasets_cache=str(root / 'hf-cache/datasets'))
    m['storage'] = {'data_root': str(data), 'launcher_receipt': str(pointer),
                    'budget_bytes': args.budget_gib * 1024**3, 'min_free_bytes': args.headroom_gib * 1024**3, 'retention_days': 14}
    c.write(root / ('storage-' + run_id + '.json'), c.storage_preflight(m, root))
    probe_argv = [args.python, '-c', 'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("lm_eval","vllm","transformers","datasets")}))']
    m['client_versions'] = runner.run_job(probe_argv, root / 'setup-versions', m['client_env'],
                                         lambda directory: c.read(directory / 'stdout.txt'), 120, 1024**2, resume=True)
    c.validate_manifest(m)
    manifest = root / 'manifest.json'
    write_fixed(manifest, m)
    argv = [args.python, str(Path(__file__).with_name('run_pair.py')), '--manifest', str(manifest), '--model', args.model,
            '--native-receipt', args.native_receipt, '--output', str(root / 'pair'), '--endpoint', args.endpoint,
            '--execute', '--resume', '--smoke']
    if args.smoke_only:
        argv.append('--smoke-only')
    attempt = root / f'runner-{len(list(root.glob("runner-*"))) + 1:03d}'
    try:
        runner.saved_process(argv, attempt, {**m['client_env'], 'OPERATOR_SWEEP_RUN_ID': run_id}, 4 * 86400, 2 * 1024**3)
    finally:
        c.check_model_snapshot(model_identity, args.model)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
