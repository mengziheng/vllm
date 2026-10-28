#!/usr/bin/env python3
"""Run baseline then tuned on one allocated four-GPU group; retain every attempt."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

import compare as c


def install_signal_handlers():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)


def atomic_write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    c.write(tmp, value)
    tmp.replace(path)


def capture_startup_log(live_log, snapshot):
    captured = Path(live_log).read_bytes()
    with Path(snapshot).open('xb') as stream:
        stream.write(captured)


def stop_group(process):
    """Only a process group created by this script is eligible for termination."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A failed leader can leave workers in its group after it has exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def saved_process(argv, directory, env, timeout, max_bytes, cwd=None, watch=None):
    c.require_nonsecret_environment(env)
    directory.mkdir(parents=True, exist_ok=False)
    c.write(directory / 'command.json', {'argv': argv, 'cwd': str(cwd) if cwd else None, 'env': env})
    process = None
    reason = None
    started = time.time()
    try:
        with (directory / 'stdout.txt').open('xb') as out, (directory / 'stderr.txt').open('xb') as err:
            process = subprocess.Popen(argv, cwd=cwd, env=dict(env), stdout=out, stderr=err, start_new_session=True)
            if sys.platform == 'linux':
                # Linux can briefly expose an empty environ while exec replaces mm.
                # Retry that transient state only; a nonempty mismatch still fails closed.
                environment_reads = 0
                for environment_reads in range(100):
                    actual = c.server_environment(process.pid)
                    if actual:
                        break
                    c.require(process.poll() is None, 'client exited before actual environment capture')
                    time.sleep(.01)
                c.require_nonsecret_environment(actual)
                c.write(directory / 'actual-environment.json', actual)
                c.write(directory / 'environment-capture.json', {'reads': environment_reads + 1, 'pid': process.pid})
                c.validate_runtime_environment(env, actual)
            while process.poll() is None:
                if watch:
                    watch()
                if time.time() - started > timeout:
                    reason = 'timeout'
                    break
                if out.tell() + err.tell() > max_bytes:
                    reason = 'output budget exceeded'
                    break
                time.sleep(.1)
            if reason:
                stop_group(process)
            code = process.wait()
    except BaseException as error:
        reason = str(error)
        if process:
            stop_group(process)
        code = process.returncode if process and process.returncode is not None else 127
        raise
    finally:
        c.write(directory / 'exit.json', {'returncode': code, 'reason': reason, 'started_unix': started, 'ended_unix': time.time()})
    c.require(not reason and code == 0, f'command failed: returncode={code}, reason={reason}; retained {directory}')


def verify_reaudit_context(attempt, context):
    previous_path = attempt / 'server-context.json'
    previous = c.read(previous_path) if previous_path.is_file() else None
    c.require(bool(previous) == bool(context), 're-audit proof context missing or newly introduced')
    if not context:
        return {'original': None, 'current': None}
    c.require('proof' in previous and 'proof' in context, 're-audit requires captured service/source proofs')
    proofs = []
    for item in (previous, context):
        c.require(c.sha256(item['proof']) == item['proof_sha256'], 're-audit proof file changed')
        proof = c.read(item['proof'])
        c.require(proof.get('source_verified') is True, 're-audit source proof was not verified')
        c.validate_runtime_environment(proof['env'], proof['actual_server_env'])
        c.require(c.sha256(proof['native_receipt']) == proof['shared']['native_receipt_sha256'], 're-audit native receipt changed')
        c.require(c.sha256(proof['server_log_snapshot']) == proof['server_log_sha256'], 're-audit original startup snapshot changed')
        proofs.append(proof)
    for field in ('pair_id', 'arm', 'endpoint', 'model', 'shared', 'repo', 'python', 'package_root', 'env',
                  'actual_server_env', 'config_sha256', 'native_receipt'):
        c.require(proofs[0].get(field) == proofs[1].get(field), 're-audit original/current proof differs: ' + field)
    return {'original': previous, 'current': context}


def run_job(argv, directory, env, audit, timeout, max_bytes, resume=False, context=None, watch=None):
    directory = Path(directory)
    identity = {'argv': argv, 'env': env}
    if directory.exists():
        c.require(c.read(directory / 'identity.json') == identity, 'saved command/environment changed')
        complete = directory / 'complete.json'
        if complete.exists():
            receipt = c.read(complete)
            attempt = directory / receipt['attempt']
            c.require(c.read(attempt / 'exit.json')['returncode'] == 0, 'saved command did not succeed')
            result = audit(attempt)
            c.require(result == receipt['audit'], 'saved raw result audit changed')
            return result
        c.require(resume, 'unfinished job exists; use --resume to retain it and create another attempt')
        # A completed CLI can have valid raw output rejected only by an older audit.
        # Re-audit the earliest exit-zero attempt; never rerun it to select another score.
        for candidate in sorted(path for path in directory.glob('attempt-*') if path.is_dir()):
            if not (candidate / 'exit.json').is_file():
                continue
            exit_receipt = c.read(candidate / 'exit.json')
            if exit_receipt.get('returncode') != 0 or exit_receipt.get('reason') is not None:
                continue
            effective = [arg.replace(str(directory / 'results'), str(candidate / 'results')) for arg in argv]
            c.require(c.read(candidate / 'command.json') == {'argv': effective, 'cwd': None, 'env': env},
                      're-audit original command/environment differs')
            actual_env = candidate / 'actual-environment.json'
            if sys.platform == 'linux':
                c.require(actual_env.is_file(), 're-audit actual client environment missing')
            if actual_env.is_file():
                c.validate_runtime_environment(env, c.read(actual_env))
            proof_contexts = verify_reaudit_context(candidate, context)
            result = audit(candidate)
            atomic_write(directory / 'complete.json', {
                'attempt': candidate.name, 'audit': result,
                're_audit': {'reason': 'previous command exited 0; unchanged raw output re-audited without rerunning the command',
                             'source_proof_contexts': proof_contexts, 'original_exit': exit_receipt}})
            return result
    else:
        directory.mkdir(parents=True)
        c.write(directory / 'identity.json', identity)
    attempt = directory / f'attempt-{len(list(directory.glob("attempt-*"))) + 1:03d}'
    # Keep the fixed logical command as identity, and preserve the actual unique output path too.
    effective = [arg.replace(str(directory / 'results'), str(attempt / 'results')) for arg in argv]
    try:
        saved_process(effective, attempt, env, timeout, max_bytes, watch=watch)
    finally:
        if context and attempt.is_dir():
            c.write(attempt / 'server-context.json', context)
    result = audit(attempt)
    atomic_write(directory / 'complete.json', {'attempt': attempt.name, 'audit': result})
    return result


def client_environment(m, directory):
    directory = Path(directory)
    env = {**m.get('client_env', {}), 'HF_HOME': m['dataset_cache'],
           'HF_HUB_CACHE': m['hf_hub_cache'], 'HF_DATASETS_CACHE': m['hf_datasets_cache'],
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TMPDIR': str(directory / 'tmp'), 'XDG_CACHE_HOME': str(directory / 'cache'),
           'VLLM_BATCH_INVARIANT': '0'}
    for name in ('tmp', 'cache'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    c.require_nonsecret_environment(env)
    return env


@contextmanager
def running_server(argv, directory, cwd, env, endpoint, health_timeout, max_log_bytes):
    directory.mkdir(parents=True, exist_ok=False)
    c.write(directory / 'command.json', {'argv': argv, 'cwd': str(cwd), 'env': env})
    # A responding pre-existing service must never be mistaken for this script's service.
    try:
        with urlopen(endpoint.rstrip('/') + '/health', timeout=2):
            raise ValueError('endpoint already has a healthy service')
    except OSError:
        pass
    process = None
    started = time.time()
    try:
        with (directory / 'server.log').open('xb') as log:
            process = subprocess.Popen(argv, cwd=cwd, env=dict(env), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            c.write(directory / 'process.json', {'pid': process.pid, 'start_ticks': c.live_start_ticks(process.pid)})
            while True:
                c.require(process.poll() is None, f'server exited before health: {process.returncode}')
                c.require(time.time() - started < health_timeout, 'server health timeout')
                c.require((directory / 'server.log').stat().st_size <= max_log_bytes, 'server log budget exceeded')
                try:
                    with urlopen(endpoint.rstrip('/') + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(2)
            yield process, directory / 'server.log'
    finally:
        if process:
            stop_group(process)
            c.write(directory / 'exit.json', {'returncode': process.returncode, 'started_unix': started, 'ended_unix': time.time()})


def collect_proof(m, arm, model, endpoint, process, log, native):
    a = m['arms'][arm]
    proof = {'pair_id': m['pair_id'], 'arm': arm, 'endpoint': endpoint, 'model': model,
             'shared': m['shared'], **{k: a[k] for k in ('repo', 'python', 'package_root', 'env')},
             'native_receipt': native, 'server_log_sha256': c.sha256(log),
             'server_pid': process.pid, 'server_start_ticks': c.inspect_live(m, arm, model, endpoint, process.pid),
             'actual_server_env': c.validate_runtime_environment(a['env'], c.server_environment(process.pid)),
             'source_verified': True,
             'config_sha256': {t: c.sha256(Path(a['package_root']) / t.removeprefix('vllm/')) for t in m['target_configs']} if arm == 'tuned' else {}}
    c.check_native(proof)
    return proof


def audit_job(job, directory):
    expected = job['expected']
    if job['kind'] == 'benchmark':
        result = c.audit_benchmark(directory / 'results/benchmark.json', expected['input_len'], expected['output_len'])
        receipt = c.read(directory / 'results/input-receipt.json')
        requests = receipt['requests']
        c.require(len(requests) == 128 and [r['index'] for r in requests] == list(range(128)), 'actual input receipt incomplete')
        c.require(all(r['input_len'] == expected['input_len'] and r['output_len'] == expected['output_len'] for r in requests), 'actual input receipt length mismatch')
        result['actual_input_sha256'] = [r['prompt_sha256'] for r in requests]
        for phase, expected_count in (('warmup', 16), ('measured', 128)):
            rows = [row for row in receipt['request_outcomes'] if row['phase'] == phase]
            c.require(len(rows) == expected_count and all(row['success'] and not row['error'] for row in rows),
                      phase + ' has failed or missing request outcomes')
        result['warmups'] = 16
        result['warmup_failed'] = 0
        return result
    return c.audit_accuracy(directory / 'results', expected['task'], expected['count'])


def audit_client_provenance(obj, m):
    c.require(obj['versions'] == m['client_versions'] and obj['versions']['lm_eval'] == '0.4.13', 'client distribution version mismatch')
    package = Path(m['client_package_root']).resolve()
    c.require(package == (Path(m['client_source_repo']) / 'vllm').resolve(), 'client source/package paths differ')
    c.require(Path(obj['vllm_file']).resolve() == package / '__init__.py' and
              Path(obj['bench_serve_file']).resolve() == package / 'benchmarks/serve.py',
              'actual client import path differs from fixed source')
    c.require(obj['source']['commit'] == m['shared']['commit'], 'actual client source commit differs')
    return obj


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'model', 'output', 'native-receipt'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8000')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-only', action='store_true', help='check the tuned service startup and one output token, then stop')
    parser.add_argument('--health-timeout', type=float, default=3600)
    parser.add_argument('--job-timeout', type=float, default=86400)
    parser.add_argument('--max-log-bytes', type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.smoke_only:
        args.smoke = True
    m = c.read(args.manifest)
    c.validate_manifest(m)
    root = Path(args.output)
    identity = {'manifest': m, 'manifest_sha256': c.sha256(args.manifest), 'model': args.model,
                'endpoint': args.endpoint, 'native_receipt_sha256': c.sha256(args.native_receipt),
                'execution_mode': 'smoke-only-tuned' if args.smoke_only else 'full-pair'}
    c.require(identity['native_receipt_sha256'] == m['shared']['native_receipt_sha256'], 'native receipt differs from manifest')
    if root.exists():
        c.require(args.resume, 'output exists; use --resume')
        c.require(c.read(root / 'identity.json') == identity, 'pair inputs changed; choose a new output')
    else:
        root.mkdir(parents=True)
        c.write(root / 'identity.json', identity)
    plan = {arm: {'launch': {'argv': c.launch_command(m, arm, args.model, args.endpoint),
                            'cwd': m['arms'][arm]['repo'], 'env': m['arms'][arm]['env']},
                  'workloads': c.make_workloads(m['client_python'], args.model, args.endpoint, root / arm)}
            for arm in ('baseline', 'tuned')}
    if not (root / 'plan.json').exists():
        c.write(root / 'plan.json', plan)
    c.require(c.read(root / 'plan.json') == plan, 'saved plan changed')
    if not args.execute:
        print(root / 'plan.json')
        return 0
    m['_native_receipt'] = args.native_receipt
    c.write(root / 'storage-preflight.json', c.storage_preflight(m, root))
    c.require(Path(args.model).is_dir(), 'local model directory required')
    for key in ('dataset_cache', 'hf_hub_cache', 'hf_datasets_cache'):
        c.require(Path(m[key]).is_dir(), 'missing offline cache: ' + key)
    # The scheduler owns GPU allocation; this lock prevents concurrent writers to this output.
    import fcntl
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        audits = {}
        env = client_environment(m, root / 'client')
        for arm in (('tuned',) if args.smoke_only else ('baseline', 'tuned')):
            if m.get('model_identity'):
                c.check_model_snapshot(m['model_identity'], args.model)
            arm_dir = root / arm
            arm_dir.mkdir(exist_ok=True)
            jobs = plan[arm]['workloads']
            if not args.smoke_only and all((arm_dir / job['name'] / 'complete.json').exists() for job in jobs):
                audits[arm] = [{'name': job['name'], 'audit': run_job(job['argv'], arm_dir / job['name'], env,
                    lambda directory, job=job: audit_job(job, directory), args.job_timeout, args.max_log_bytes, resume=True)} for job in jobs]
                continue
            version_argv = [m['client_python'], str(Path(__file__).with_name('client_probe.py')),
                            '--manifest', args.manifest, '--native-receipt', args.native_receipt,
                            '--output', str(root / 'client-version/results/client-provenance.json')]
            def version_audit(directory):
                return audit_client_provenance(c.read(directory / 'results/client-provenance.json'), m)
            run_job(version_argv, root / 'client-version', env, version_audit, 120, args.max_log_bytes, resume=args.resume)
            server_dir = arm_dir / f'server-{len(list(arm_dir.glob("server-*"))) + 1:03d}'
            launch = plan[arm]['launch']
            with running_server(launch['argv'], server_dir, launch['cwd'], launch['env'], args.endpoint,
                                args.health_timeout, args.max_log_bytes) as (process, log):
                if args.smoke:
                    smoke_argv = [m['client_python'], str(Path(__file__).with_name('smoke_request.py')),
                                  '--endpoint', args.endpoint, '--model', args.model]
                    def smoke_audit(directory):
                        obj = c.read(directory / 'stdout.txt')
                        c.require(obj['response']['usage']['completion_tokens'] == 1, 'startup output token missing')
                        return {'real_output_tokens': 1, 'scores_computed': False}
                    run_job(smoke_argv, server_dir / 'startup-request', env, smoke_audit, 660, args.max_log_bytes,
                            context={'server_process': str(server_dir / 'process.json')})
                startup_snapshot = server_dir / 'startup-log.snapshot.txt'
                capture_startup_log(log, startup_snapshot)
                proof = collect_proof(m, arm, args.model, args.endpoint, process, startup_snapshot, args.native_receipt)
                proof['server_log_snapshot'] = str(startup_snapshot)
                proof['live_server_log'] = str(log)
                proof_path = server_dir / 'proof.json'
                c.write(proof_path, proof)
                # validate_evidence uses the same unchanged proof path from the manifest.
                check_manifest = json.loads(json.dumps(m))
                check_manifest['arms'][arm]['proof'] = str(proof_path)
                c.validate_evidence(check_manifest, arm, args.endpoint, startup_snapshot, args.model)
                c.write(server_dir / 'launch-confirmations.json', {
                    'moe_backend': 'triton', 'tensor_parallel_size': 4, 'VLLM_BATCH_INVARIANT': '0',
                    'gpu_name': m['shared']['hardware']['gpu_name'],
                    'gpu_uuids': m['shared']['hardware']['gpu_uuids'],
                    'evidence': ['actual /proc argv/environment', 'actual nvidia-smi selected GPUs', 'actual backend/config server log']})
                if args.smoke_only:
                    c.write(root / 'smoke-audit.json', {'status': 'passed', 'scope': 'tuned-startup-one-real-output-token-no-task-score',
                                                       'server_proof': str(proof_path), 'startup_request': str(server_dir / 'startup-request')})
                    continue
                audits[arm] = []
                def watch():
                    c.require(process.poll() is None, 'server exited while client was running')
                    c.require(log.stat().st_size <= args.max_log_bytes, 'server log budget exceeded')
                for job in jobs:
                    c.require(process.poll() is None and c.live_start_ticks(process.pid) == proof['server_start_ticks'], 'server identity changed')
                    c.require(log.stat().st_size <= args.max_log_bytes, 'server log budget exceeded')
                    result = run_job(job['argv'], arm_dir / job['name'], env,
                                     lambda directory, job=job: audit_job(job, directory), args.job_timeout,
                                     args.max_log_bytes, resume=args.resume,
                                     context={'proof': str(proof_path), 'proof_sha256': c.sha256(proof_path)}, watch=watch)
                    audits[arm].append({'name': job['name'], 'audit': result})
                    atomic_write(arm_dir / 'audit.json', audits[arm])
            atomic_write(arm_dir / 'status.json', {'status': 'passed', 'jobs': len(jobs)})
            if m.get('model_identity'):
                c.check_model_snapshot(m['model_identity'], args.model)
        if args.smoke_only:
            print(root / 'smoke-audit.json')
            return 0
        c.require(audits['baseline'] == audits['tuned'], 'baseline/tuned sample or request audits differ')
        atomic_write(root / 'pair-audit.json', {'status': 'paired-audits-passed', 'pair_id': m['pair_id'], 'audits': audits})
    print(root / 'pair-audit.json')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
