#!/usr/bin/env python3
"""Paired, sequential H20 validation. No installation, build or server launch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from urllib.parse import urlparse

SHAPES = ((8192, 1), (2048, 512), (512, 1024), (512, 128))
_NATIVE_HASHES = {}
ACCURACY_PROTOCOL = {
    'gsm8k': {'api_mode': 'local-completions', 'fewshot': 5, 'count': 1319, 'max_gen_toks': 256,
              'until': ['Question:', '</s>', '<|im_end|>']},
    'aime25': {'api_mode': 'local-chat-completions', 'fewshot': 0, 'count': 30, 'max_gen_toks': 32768,
               'until': ['Question:', '</s>', '<|im_end|>', '<|eot_id|>']}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def native_sha256(path):
    """Reuse hashes only for unchanged native receipt files in this runner process."""
    actual = Path(path).resolve()
    stat = actual.stat()
    identity = (str(actual), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity not in _NATIVE_HASHES:
        _NATIVE_HASHES[identity] = sha256(actual)
    return _NATIVE_HASHES[identity]


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_nonsecret_environment(env):
    sensitive = re.compile(r'(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIALS?|API_KEY|ACCESS_KEY|PRIVATE_KEY|AUTHORIZATION|COOKIE)(?:_|$)', re.I)
    require(not any(sensitive.search(key) for key in env), 'credential/secret environment keys are forbidden; values are not recorded')


def validate_runtime_environment(declared, actual):
    require_nonsecret_environment(declared)
    require_nonsecret_environment(actual)
    require(set(actual) == set(declared), 'live environment key mismatch: missing=' +
            repr(sorted(set(declared) - set(actual))) + ', undeclared=' + repr(sorted(set(actual) - set(declared))))
    require(actual == declared, 'live server environment values differ from fixed manifest')
    return dict(actual)


def server_environment(pid):
    raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
    return dict(x.decode().split('=', 1) for x in raw.split(b'\0') if b'=' in x)


def run_command(argv, outdir, env, cwd=None):
    """Save command and both streams even on spawn/nonzero failure. Never reuse a directory."""
    require_nonsecret_environment(env)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    write(outdir / 'command.json', {'argv': argv, 'cwd': str(cwd) if cwd else None, 'env': env})
    with (outdir / 'stdout.txt').open('wb') as stdout, (outdir / 'stderr.txt').open('wb') as stderr:
        try:
            proc = subprocess.run(argv, cwd=cwd, env=dict(env), stdout=stdout, stderr=stderr, check=False)
            code = proc.returncode
        except OSError as e:
            stderr.write((str(e) + '\n').encode())
            code = 127
    write(outdir / 'exit.json', {'returncode': code})
    return code


def make_workloads(python, model, endpoint, run_dir, repeats=3, seed=20261028, api_mode='local-chat-completions'):
    """The same fixed plan is used for both arms; only endpoint/output location differ."""
    jobs = []
    for task, fewshot, count in [('gsm8k', 5, 1319), ('aime25', 0, 30)]:
        name = 'accuracy-' + task
        output = Path(run_dir) / name / 'results'
        protocol = ACCURACY_PROTOCOL[task]
        api_mode = protocol['api_mode']
        suffix = '/v1/chat/completions' if api_mode == 'local-chat-completions' else '/v1/completions'
        # JSON model_args preserves commas/spaces in a local model path.
        args = {'model': model, 'tokenizer': model, 'base_url': endpoint.rstrip('/') + suffix,
                'num_concurrent': 8, 'max_retries': 0, 'tokenized_requests': False,
                'tokenizer_backend': 'huggingface' if task == 'gsm8k' else None,
                'max_length': 40000, 'timeout': 7200, 'seed': seed}
        argv = [python, '-m', 'lm_eval', '--model', api_mode, '--model_args', json.dumps(args),
                '--tasks', task, '--num_fewshot', str(fewshot), '--batch_size', '1',
                '--seed', ','.join([str(seed)] * 4), '--log_samples', '--output_path', str(output),
                '--gen_kwargs', json.dumps({'do_sample': False, 'temperature': 0.0,
                    'max_gen_toks': protocol['max_gen_toks'], 'until': protocol['until']})]
        if api_mode == 'local-chat-completions':
            argv.append('--apply_chat_template')
        jobs.append({'name': name, 'kind': 'accuracy', 'argv': argv, 'expected': {'task': task, 'count': count}})
    for repeat in range(repeats):
        for input_len, output_len in SHAPES:
            name = f'bench-r{repeat + 1}-{input_len}-{output_len}'
            argv = [python, str(Path(__file__).with_name('bench_record.py')),
                    '--input-receipt', str(Path(run_dir) / name / 'results' / 'input-receipt.json'),
                    '--', 'bench', 'serve', '--backend', 'vllm',
                    '--base-url', endpoint.rstrip('/'), '--endpoint', '/v1/completions',
                    '--model', model, '--tokenizer', model, '--dataset-name', 'random',
                    '--random-input-len', str(input_len), '--random-output-len', str(output_len),
                    '--random-range-ratio', '0', '--num-prompts', '128', '--num-warmups', '16',
                    '--seed', str(seed + repeat), '--temperature', '0',
                    '--extra-body', json.dumps({'seed': seed + repeat,
                        'cache_salt': f'pr58986-bench-{input_len}-{output_len}-r{repeat + 1}-seed{seed + repeat}'}),
                    '--request-rate', 'inf', '--max-concurrency', '128',
                    '--ignore-eos', '--save-result', '--save-detailed',
                    '--result-dir', str(Path(run_dir) / name / 'results'), '--result-filename', 'benchmark.json']
            jobs.append({'name': name, 'kind': 'benchmark', 'argv': argv,
                         'expected': {'input_len': input_len, 'output_len': output_len, 'seed': seed + repeat}})
    return jobs


def validate_manifest(m):
    require(m['schema_version'] == 1 and m['pair_id'], 'schema_version/pair_id required')
    shared = m['shared']
    require(re.fullmatch(r'[0-9a-f]{40}', shared['commit']) is not None, 'fixed 40-digit commit required')
    for name in ('common_diff_sha256', 'native_receipt_sha256'):
        require(re.fullmatch(r'[0-9a-f]{64}', shared[name]) is not None, name + ' must be SHA256')
    require(shared['model_revision'], 'fixed model revision/weight receipt required')
    require(re.search(r'\bH20\b', shared['hardware']['gpu_name']) is not None and len(set(shared['hardware']['gpu_uuids'])) == 4,
            'four identified H20 GPUs required')
    require(m['target_configs'], 'target_configs cannot be empty')
    for target in m['target_configs']:
        require(target.startswith('vllm/') and '..' not in Path(target).parts and target.endswith('.json'), 'unsafe config path')
    reserved = ('--model', '--tensor-parallel-size', '-tp', '--moe-backend', '--host', '--port', '--served-model-name')
    for arg in shared['server_args']:
        require(not any(arg == x or arg.startswith(x + '=') for x in reserved), 'reserved server option: ' + arg)
    for arm in ('baseline', 'tuned'):
        a = m['arms'][arm]
        require_nonsecret_environment(a['env'])
        require(a['env'].get('VLLM_BATCH_INVARIANT') == '0', 'VLLM_BATCH_INVARIANT must be explicitly 0')
        require(not a['env'].get('VLLM_TUNED_CONFIG_FOLDER'), 'override folder is forbidden; baseline needs removed bundled JSON')
    require_nonsecret_environment(m.get('client_env', {}))
    require(m['arms']['baseline']['env'] == m['arms']['tuned']['env'], 'server env must match')
    require(m['arms']['baseline']['repo'] != m['arms']['tuned']['repo'], 'independent baseline/tuned repos required')


def validate_evidence(m, arm, endpoint, log, model):
    validate_manifest(m)
    a = m['arms'][arm]
    proof = read(a['proof'])
    for field, expected in [('pair_id', m['pair_id']), ('arm', arm), ('endpoint', endpoint), ('model', model),
                            ('shared', m['shared']), ('repo', a['repo']), ('python', a['python']),
                            ('package_root', a['package_root']), ('env', a['env'])]:
        require(proof.get(field) == expected, 'proof mismatch: ' + field)
    validate_runtime_environment(a['env'], proof['actual_server_env'])
    require(proof.get('source_verified') is True, 'source verification receipt required')
    require(proof.get('server_pid') and proof.get('server_start_ticks'), 'live server identity required')
    require(sha256(log) == proof['server_log_sha256'], 'server log changed; collect a fresh proof')
    require(sha256(proof['native_receipt']) == m['shared']['native_receipt_sha256'], 'native receipt mismatch')
    text = Path(log).read_text(errors='replace')
    backend_lines = re.findall(r'Using [^\n]*MoE backend[^\n]*', text, re.I)
    selected_backends = [re.match(r"Using ['\"]?([A-Za-z0-9_]+)['\"]?\s", line, re.I) for line in backend_lines]
    require(selected_backends and all(match and match.group(1).upper() == 'TRITON' for match in selected_backends),
            'actual Triton MoE selection log required; conflicting backends rejected')
    config_hashes = {}
    for target in m['target_configs']:
        name = Path(target).name
        source = Path(a['repo']) / target
        installed = Path(a['package_root']) / target.removeprefix('vllm/')
        loaded = [line for line in text.splitlines() if 'Using configuration from ' in line and name in line]
        if arm == 'baseline':
            require(not source.exists() and not installed.exists() and not loaded,
                    'baseline still contains or loads tuned config: ' + name)
            defaults = [line for line in text.splitlines() if 'Using default MoE config.' in line and name in line]
            require(defaults, 'baseline default MoE config evidence missing: ' + name)
        else:
            require(source.is_file() and installed.is_file(), 'tuned config missing: ' + name)
            require(sha256(source) == sha256(installed), 'source/installed config mismatch: ' + name)
            require(any(str(installed.resolve()) in line for line in loaded), 'tuned exact loaded config path missing: ' + name)
            config_hashes[target] = sha256(installed)
    if arm == 'tuned':
        require(proof.get('config_sha256') == config_hashes == m['tuned_config_sha256'], 'tuned config SHA mismatch')
    return proof


def audit_benchmark(path, input_len, output_len, num_prompts=128):
    obj = read(path)
    require(obj['completed'] == num_prompts and obj['failed'] == 0, 'benchmark has failed/missing requests')
    for key in ('input_lens', 'output_lens', 'errors'):
        require(len(obj.get(key, [])) == num_prompts, 'missing detailed request audit: ' + key)
    require(not any(obj['errors']), 'benchmark request errors')
    require(all(x == input_len for x in obj['input_lens']), 'benchmark input length mismatch')
    require(all(x == output_len for x in obj['output_lens']), 'benchmark output length mismatch')
    return {'requests': num_prompts, 'input_len': input_len, 'output_len': output_len, 'failed': 0}


def audit_accuracy(directory, task, count):
    directory = Path(directory)
    results = [p for p in directory.rglob('*.json') if 'results' in p.name]
    require(len(results) == 1, 'exactly one lm_eval result file required')
    obj = read(results[0])
    require(task in obj['results'], 'accuracy result missing task')
    require(obj['n-samples'][task] == {'original': count, 'effective': count}, 'accuracy full sample count mismatch')
    samples = list(directory.rglob('*samples*' + task + '*.jsonl'))
    require(len(samples) == 1, 'per-sample accuracy output required')
    rows = [json.loads(line) for line in samples[0].read_text().splitlines() if line]
    expected_filters = {'gsm8k': ('strict-match', 'flexible-extract'), 'aime25': ('none',)}[task]
    grouped = {name: [] for name in expected_filters}
    require({row.get('filter') for row in rows} == set(expected_filters), 'official accuracy filter set mismatch')
    for row in rows:
        grouped[row['filter']].append(row)
    expected_ids = set(range(count))
    for name, group in grouped.items():
        require(len(group) == count and len({row['doc_id'] for row in group}) == count and
                {row['doc_id'] for row in group} == expected_ids, 'accuracy filter doc coverage incomplete: ' + name)
    # 0.4.13 emits one logged row per filter/doc, with identical raw input/output.
    shared_fields = ('doc', 'doc_hash', 'prompt_hash', 'target_hash', 'arguments', 'target', 'resps')
    reference = {row['doc_id']: row for row in grouped[expected_filters[0]]}
    for name in expected_filters[1:]:
        for row in grouped[name]:
            require(all(key in row and key in reference[row['doc_id']] and
                        row[key] == reference[row['doc_id']][key] for key in shared_fields),
                    'accuracy cross-filter input/target/raw response mismatch')
    require(all(row.get('resps') and any(str(v).strip() for group in row['resps'] for v in group) for row in rows), 'empty accuracy response')
    return {'task': task, 'samples': count, 'full_dataset': True, 'logged_sample_rows': len(rows),
            'filters': {name: len(grouped[name]) for name in expected_filters}}


def launch_command(m, arm, model, endpoint):
    a = m['arms'][arm]
    url = urlparse(endpoint)
    require(url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1') and url.port,
            'run this tool on server host with an explicit localhost port')
    require(url.path in ('', '/'), 'endpoint must be server root, without /v1')
    return [a['python'], '-m', 'vllm.entrypoints.cli.main', 'serve', model,
            '--tensor-parallel-size', '4', '--moe-backend', 'triton', '--host', '127.0.0.1',
            '--port', str(url.port), *m['shared']['server_args']]


def live_start_ticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def verify_source(m, arm, env, native_receipt=None):
    repo = Path(m['arms'][arm]['repo'])
    git = ['git', '-c', 'safe.directory=' + str(repo.resolve())]
    def capture(argv):
        return subprocess.check_output([*git, *argv[1:]], cwd=repo, env=env, text=True).strip()
    require(capture(['git', 'rev-parse', 'HEAD']) == m['shared']['commit'], 'live source commit mismatch')
    generated = {}
    if native_receipt:
        require(sha256(native_receipt) == m['shared']['native_receipt_sha256'], 'native receipt mismatch')
        generated = read(native_receipt).get('source_files', {})
        for relative, digest in generated.items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'unsafe generated source path')
            require(native_sha256(repo / relative) == digest, 'generated source receipt changed: ' + relative)
    exclusions = [':(exclude)' + x for x in [*m['target_configs'], *generated]]
    diff = subprocess.check_output([*git, 'diff', '--binary', 'HEAD', '--', '.', *exclusions], cwd=repo, env=env)
    require(hashlib.sha256(diff).hexdigest() == m['shared']['common_diff_sha256'], 'common source diff mismatch')
    extras = capture(['git', 'ls-files', '--others', '--exclude-standard']).splitlines()
    require(all(name in generated for name in extras), 'unreceipted untracked repo files prevent fixed source proof')
    return {'commit': m['shared']['commit'], 'generated_source_files': generated}


def inspect_live(m, arm, model, endpoint, pid):
    """Read Linux process/source receipts without installing, importing model weights or starting GPUs."""
    a = m['arms'][arm]
    proc = Path('/proc') / str(pid)
    cmd = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    expected = launch_command(m, arm, model, endpoint)
    require(Path(cmd[0]).resolve() == Path(expected[0]).resolve() and cmd[1:] == expected[1:], 'server argv differs from generated launch command')
    require((proc / 'cwd').resolve() == Path(a['repo']).resolve(), 'server cwd differs from repo')
    env = validate_runtime_environment(a['env'], server_environment(pid))
    require(not env.get('VLLM_TUNED_CONFIG_FOLDER'), 'live override folder is forbidden')
    start = live_start_ticks(pid)
    def capture(argv, cwd=None):
        return subprocess.check_output(argv, cwd=cwd, env=env, text=True).strip()
    verify_source(m, arm, env, m.get('_native_receipt'))
    package = capture([a['python'], '-c', 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).resolve().parent)'], a['repo'])
    require(Path(package).resolve() == Path(a['package_root']).resolve() == (Path(a['repo']) / 'vllm').resolve(), 'actual Python vllm package must be the fixed checkout package')
    gpu = capture(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'])
    lines = [line.split(',') for line in gpu.splitlines()]
    visible = env.get('CUDA_VISIBLE_DEVICES', '')
    require(visible and len(visible.split(',')) == 4, 'explicit four CUDA_VISIBLE_DEVICES required')
    selected = []
    for device in visible.split(','):
        if device.isdigit():
            selected.append(lines[int(device)])
        else:
            selected.append(next(line for line in lines if line[1].strip() == device))
    require(all(re.search(r'\bH20\b', line[0]) for line in selected), 'live GPUs are not H20')
    require([line[1].strip() for line in selected] == m['shared']['hardware']['gpu_uuids'], 'live GPU identities mismatch')
    return start


def check_native(proof):
    receipt = read(proof['native_receipt'])
    require(receipt.get('files'), 'native receipt must list actual extension/runtime files and hashes')
    require(all(receipt.get('versions', {}).get(k) for k in ('torch','cuda','triton','driver','python','build_receipt')), 'native receipt must record runtime/build versions')
    for item in receipt['files']:
        require(native_sha256(item['path']) == item['sha256'], 'native binary receipt changed: ' + item['path'])


def validate_storage_paths(m, output_root):
    data = Path(m['storage']['data_root']).resolve()
    # A read-only bind of another directory on the same data filesystem may have
    # an unrelated container path. Confirm its actual device rather than its spelling.
    cache = Path(m['dataset_cache']).resolve()
    require(cache.is_relative_to(data) or (cache.is_dir() and cache.stat().st_dev == data.stat().st_dev),
            'dataset cache must be on the actual data filesystem')
    require(Path(output_root).resolve().is_relative_to(data), 'output root must be under data mount')
    for arm in ('baseline', 'tuned'):
        for key in ('TMPDIR', 'HF_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME'):
            value = m['arms'][arm]['env'].get(key)
            require(value and Path(value).resolve().is_relative_to(data), 'server cache/tmp path required under data mount: ' + key)


def storage_preflight(m, output_root):
    validate_storage_paths(m, output_root)
    s = m['storage']
    data = Path(s['data_root']).resolve()
    require(data.is_dir(), 'data mount must exist')
    launcher_receipt = s.get('launcher_receipt')
    if launcher_receipt:
        receipt_path = Path(launcher_receipt).resolve()
        require(receipt_path.is_relative_to(data), 'launcher storage receipt must be on actual task data mount')
        record = read(receipt_path)
        require(os.environ.get('OPERATOR_SWEEP_RUN_ID') and record['name'] == os.environ['OPERATOR_SWEEP_RUN_ID'], 'launcher storage run identity mismatch')
        require(record['status'] == 'running' and record['storage']['storage_ready'] is True, 'current launcher storage is not ready/running')
        require(record['storage']['data_storage']['device'] == data.stat().st_dev, 'launcher data filesystem differs from actual mount')
        docker_device = record['storage']['docker_storage']['device']
        require(docker_device != data.stat().st_dev, 'data filesystem must be distinct from Docker filesystem')
    else:
        docker = Path(s['docker_root']).resolve()
        require(docker.is_dir(), 'actual Docker filesystem path required')
        docker_device = docker.stat().st_dev
        require(data.stat().st_dev != docker_device, 'data filesystem must be distinct from Docker filesystem')
    root = Path(output_root).resolve()
    require(root.is_relative_to(data), 'output root must be under data mount')
    free = shutil.disk_usage(data).free
    require(free >= s['min_free_bytes'] + s['budget_bytes'], 'insufficient storage budget')
    require(os.statvfs(data).f_favail > 1000 and s['retention_days'] > 0, 'inode/retention requirement missing')
    return {'free_bytes': free, 'free_inodes': os.statvfs(data).f_favail, 'data_device': data.stat().st_dev,
            'docker_device': docker_device, 'launcher_receipt': launcher_receipt}


def file_stat(path):
    stat = Path(path).stat()
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def model_snapshot(model, receipt_path):
    model = Path(model).resolve()
    receipt = read(receipt_path)
    require(not receipt.get('errors'), 'existing model structural receipt contains errors')
    files = {}
    for name, item in receipt['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe model receipt filename')
        files[name] = file_stat(model / name)
        require(files[name]['size'] == item['size'], 'model shard size differs from structural receipt: ' + name)
    for name, digest in receipt['metadata_sha256'].items():
        require(sha256(model / name) == digest, 'small model metadata SHA mismatch: ' + name)
        files[name] = file_stat(model / name)
    return {'model': str(model), 'structural_receipt_sha256': sha256(receipt_path), 'files': files,
            'verification_scope': receipt.get('verification_scope'), 'full_weight_body_hash_repeated': False}


def check_model_snapshot(snapshot, model):
    require(Path(model).resolve() == Path(snapshot['model']), 'model snapshot path changed')
    for name, expected in snapshot['files'].items():
        require(file_stat(Path(model) / name) == expected, 'model file changed since initial stat: ' + name)


def execute(args):
    m = read(args.manifest)
    validate_manifest(m)
    require(args.arm or not args.execute, '--execute requires --arm baseline|tuned for sequential servers')
    run = Path(args.output_root) / ('run-' + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'manifest.json', m)
    status = {'status': 'preparing', 'executed': False, 'pair_id': m['pair_id'], 'arm': args.arm,
              'manifest_sha256': sha256(args.manifest), 'model': args.model, 'repeats': args.repeats,
              'seed': args.seed, 'api_mode': args.api_mode}
    write(run / 'status.json', status)
    try:
        arms = [args.arm] if args.arm else ['baseline', 'tuned']
        plans = {}
        for arm in arms:
            endpoint = getattr(args, arm + '_endpoint')
            plans[arm] = {'launch': {'argv': launch_command(m, arm, args.model, endpoint),
                                    'cwd': m['arms'][arm]['repo'], 'env': m['arms'][arm]['env']},
                          'workloads': make_workloads(args.python, args.model, endpoint, run, args.repeats, args.seed, args.api_mode)}
        write(run / 'plan.json', plans)
        if not args.execute:
            status['status'] = 'dry-run'
            return run
        arm = args.arm
        require(getattr(args, arm + '_log'), 'selected arm requires --' + arm + '-log')
        endpoint, log = getattr(args, arm + '_endpoint'), Path(getattr(args, arm + '_log'))
        status['storage'] = storage_preflight(m, args.output_root)
        require(log.resolve().is_relative_to(Path(m['storage']['data_root']).resolve()), 'server log must be on data mount')
        proof = validate_evidence(m, arm, endpoint, log, args.model)
        shutil.copy2(log, run / 'server.log')
        write(run / 'proof.json', proof)
        require(inspect_live(m, arm, args.model, endpoint, proof['server_pid']) == proof['server_start_ticks'], 'stale server process proof')
        check_native(proof)
        require(Path(args.model).is_dir(), 'local model directory required; no automatic weight download')
        env = {**m.get('client_env', {}), 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HOME': str(run / 'cache' / 'hf'),
               'TMPDIR': str(run / 'tmp'), 'XDG_CACHE_HOME': str(run / 'cache'),
               'VLLM_BATCH_INVARIANT': '0'}
        # Reuse the task-specific, prepopulated offline dataset/tokenizer cache on the data mount.
        require(Path(m['dataset_cache']).is_dir(), 'prepopulated offline dataset cache required')
        env['HF_HOME'] = m['dataset_cache']
        for path in (run / 'cache', run / 'tmp'):
            path.mkdir()
        require(args.python == m['client_python'], 'client Python differs from fixed manifest')
        client_dir = run / 'client-version'
        code = run_command([args.python, '-c', 'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("lm_eval","vllm","transformers","datasets")}))'], client_dir, env)
        require(code == 0, 'client version probe failed')
        versions = read(client_dir / 'stdout.txt')
        require(versions == m['client_versions'] and versions['lm_eval'] == '0.4.13', 'client versions differ from pinned manifest (lm_eval 0.4.13)')
        status['executed'] = True
        status['client_python'] = args.python
        status['client_versions'] = versions
        audits = []
        for job in plans[arm]['workloads']:
            require(live_start_ticks(proof['server_pid']) == proof['server_start_ticks'], 'server changed between workloads')
            code = run_command(job['argv'], run / job['name'], env)
            require(code == 0, f"command failed ({code}): {job['name']}")
            expected = job['expected']
            if job['kind'] == 'benchmark':
                result = audit_benchmark(run / job['name'] / 'results' / 'benchmark.json', expected['input_len'], expected['output_len'])
            else:
                result = audit_accuracy(run / job['name'] / 'results', expected['task'], expected['count'])
            audits.append({'name': job['name'], 'audit': result})
            write(run / 'audit.json', audits)
        status['status'] = 'passed'
        return run
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        if args.execute and args.arm and getattr(args, args.arm + '_log'):
            final_log = Path(getattr(args, args.arm + '_log'))
            if final_log.is_file():
                shutil.copy2(final_log, run / 'server-final.log')
        write(run / 'status.json', status)
        print(str(run))


def audit_pair(baseline, tuned):
    statuses = [read(Path(p) / 'status.json') for p in (baseline, tuned)]
    for expected_arm, status in zip(('baseline', 'tuned'), statuses):
        require(status['executed'] and status['status'] == 'passed' and status['arm'] == expected_arm,
                'pair requires completed real baseline and tuned executions')
    for key in ('pair_id', 'manifest_sha256', 'model', 'repeats', 'seed', 'api_mode', 'client_python', 'client_versions'):
        require(statuses[0][key] == statuses[1][key], 'pair mismatch: ' + key)
    proof_envs = []
    for expected_arm, directory in zip(('baseline', 'tuned'), (baseline, tuned)):
        manifest = read(Path(directory) / 'manifest.json')
        proof = read(Path(directory) / 'proof.json')
        proof_envs.append(validate_runtime_environment(manifest['arms'][expected_arm]['env'], proof['actual_server_env']))
    require(proof_envs[0] == proof_envs[1], 'actual baseline/tuned server environments differ')
    require(read(Path(baseline) / 'audit.json') == read(Path(tuned) / 'audit.json'), 'request/sample audits differ')
    # Re-audit raw saved outputs; status flags alone do not establish complete requests.
    for run in (Path(baseline), Path(tuned)):
        plans = read(run / 'plan.json')
        arm = read(run / 'status.json')['arm']
        for job in plans[arm]['workloads']:
            require(read(run / job['name'] / 'exit.json')['returncode'] == 0, 'saved command failed')
            e = job['expected']
            if job['kind'] == 'benchmark':
                audit_benchmark(run / job['name'] / 'results' / 'benchmark.json', e['input_len'], e['output_len'])
            else:
                audit_accuracy(run / job['name'] / 'results', e['task'], e['count'])
    return {'pair_id': statuses[0]['pair_id'], 'status': 'paired-audits-passed', 'baseline': str(baseline), 'tuned': str(tuned)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    for name in ('manifest', 'model', 'baseline-endpoint', 'tuned-endpoint', 'output-root', 'python'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--baseline-log')
    p.add_argument('--tuned-log')
    p.add_argument('--arm', choices=['baseline', 'tuned'])
    p.add_argument('--execute', action='store_true')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--api-mode', choices=['local-completions','local-chat-completions'], default='local-chat-completions')
    p = sub.add_parser('audit')
    p.add_argument('baseline')
    p.add_argument('tuned')
    p = sub.add_parser('collect-proof')
    for name in ('manifest', 'model', 'endpoint', 'log', 'native-receipt', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--arm', required=True, choices=['baseline','tuned'])
    p.add_argument('--pid', required=True, type=int)
    args = parser.parse_args()
    try:
        if args.command == 'run':
            require(args.repeats > 0, 'repeats must be positive')
            execute(args)
        elif args.command == 'audit':
            print(json.dumps(audit_pair(args.baseline, args.tuned), ensure_ascii=False, indent=2))
        else:
            m = read(args.manifest); validate_manifest(m)
            a = m['arms'][args.arm]
            proof = {'pair_id':m['pair_id'], 'arm':args.arm, 'endpoint':args.endpoint, 'model':args.model,
                     'shared':m['shared'], **{k:a[k] for k in ('repo','python','package_root','env')},
                     'native_receipt':args.native_receipt, 'server_log_sha256':sha256(args.log),
                     'server_pid':args.pid, 'server_start_ticks':inspect_live(m,args.arm,args.model,args.endpoint,args.pid),
                     'source_verified':True, 'collected_at':datetime.now(timezone.utc).isoformat()}
            proof['actual_server_env'] = validate_runtime_environment(a['env'], server_environment(args.pid))
            check_native(proof)
            proof['config_sha256'] = {target:sha256(Path(a['package_root']) / target.removeprefix('vllm/'))
                                      for target in m['target_configs']} if args.arm == 'tuned' else {}
            with Path(args.output).open('x') as f:
                json.dump(proof,f,ensure_ascii=False,indent=2)
            print(args.output)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError, StopIteration, IndexError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
