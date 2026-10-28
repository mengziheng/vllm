#!/usr/bin/env python3
"""Run only new CPU orchestration tests and the actual lm_eval client contract."""
import argparse
import json
import os
from pathlib import Path
import sys

import run_pair as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'client', 'framework', 'common-site', 'cache', 'model', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reuse-tests')
    parser.add_argument('--reuse-evaluation')
    args = parser.parse_args()
    runner.install_signal_handlers()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(out / 'home'), 'LANG': 'C.UTF-8',
           'PYTHONPATH': ':'.join([args.client, args.framework, args.common_site, args.evaluation]),
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'HF_HOME': args.cache, 'HF_HUB_CACHE': str(Path(args.cache) / 'hub'),
           'HF_DATASETS_CACHE': str(Path(args.cache) / 'datasets'),
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
           'VLLM_PLUGINS': ''}
    for key, subdir in {'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'cache', 'HF_MODULES_CACHE': 'hf-modules',
                        'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
                        'VLLM_CACHE_ROOT': 'vllm-cache', 'CUDA_CACHE_PATH': 'cuda-cache',
                        'TORCH_EXTENSIONS_DIR': 'torch-extensions', 'CCACHE_DIR': 'ccache'}.items():
        env[key] = str(out / subdir)
        Path(env[key]).mkdir()
    Path(env['HOME']).mkdir()
    Path(env['HOME'], '.cache').mkdir()
    storage = {'output_device': out.stat().st_dev, 'cache_device': Path(args.cache).stat().st_dev,
               'output_free_bytes': os.statvfs(out).f_bavail * os.statvfs(out).f_frsize,
               'output_free_inodes': os.statvfs(out).f_favail}
    runner.c.require(storage['output_device'] == storage['cache_device'], 'offline read-only alias differs from actual data filesystem')
    mounts = [line for line in Path('/proc/self/mountinfo').read_text().splitlines()
              if any(' ' + path + ' ' in line for path in ('/work', args.evaluation, args.client, args.cache, args.model, '/tmp', '/sweep-home'))]
    report = {'status': 'running', 'python': sys.version, 'python_executable': sys.executable,
              'storage': storage, 'actual_mountinfo': mounts, 'gpu_work_submitted': False,
              'old_tests_repeated': False, 'full_dataset_reload': False}
    runner.c.write(out / 'cpu-receipt.json', report)
    try:
        if args.reuse_tests:
            runner.c.require(args.reuse_evaluation, 'previous frozen test source required for reuse')
            runner.c.require(runner.c.read(Path(args.reuse_tests) / 'exit.json')['returncode'] == 0, 'previous tests did not pass')
            for name in ('compare.py', 'run_pair.py', 'bench_record.py', 'tests/test_pair.py'):
                runner.c.require(runner.c.sha256(Path(args.evaluation) / name) == runner.c.sha256(Path(args.reuse_evaluation) / name),
                                 'tested code changed: ' + name)
            report['tests_reused_from'] = args.reuse_tests
        else:
            runner.saved_process([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(args.evaluation) / 'tests'), '-v'],
                                 out / 'new-tests', env, 120, 20 * 1024**2)
        runner.saved_process([sys.executable, str(Path(args.evaluation) / 'contract_probe.py'),
                              '--model', args.model, '--output', str(out / 'contract-probe.json')],
                             out / 'client-contract', env, 180, 20 * 1024**2)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        runner.c.write(out / 'cpu-receipt.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
