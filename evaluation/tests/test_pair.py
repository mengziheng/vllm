import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = {'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8'}

    def tearDown(self):
        self.tmp.cleanup()

    def module(self):
        self.assertTrue((ROOT / 'run_pair.py').is_file(), 'sequential runner is missing')
        import run_pair
        return run_pair

    def test_completed_job_is_reused_without_running_or_overwriting(self):
        p = self.module()
        out = self.root / 'job'
        argv = [sys.executable, '-c', 'print("done")']
        audit = lambda d: {'stdout': (d / 'stdout.txt').read_text()}
        first = p.run_job(argv, out, self.env, audit, 5, 100000)
        original = (out / 'attempt-001' / 'stdout.txt').read_bytes()
        second = p.run_job(argv, out, self.env, audit, 5, 100000)
        self.assertEqual(first, second)
        self.assertEqual(original, (out / 'attempt-001' / 'stdout.txt').read_bytes())
        self.assertEqual(len(list(out.glob('attempt-*'))), 1)

    def test_failed_job_preserves_attempt_and_requires_explicit_resume(self):
        p = self.module()
        flag = self.root / 'flag'
        argv = [sys.executable, '-c', 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); old=p.exists(); p.touch(); print("raw"); sys.exit(0 if old else 9)', str(flag)]
        out = self.root / 'job'
        audit = lambda d: {'ok': True}
        with self.assertRaisesRegex(ValueError, 'returncode=9'):
            p.run_job(argv, out, self.env, audit, 5, 100000)
        with self.assertRaisesRegex(ValueError, 'resume'):
            p.run_job(argv, out, self.env, audit, 5, 100000)
        p.run_job(argv, out, self.env, audit, 5, 100000, resume=True)
        self.assertEqual(json.loads((out / 'attempt-001' / 'exit.json').read_text())['returncode'], 9)
        self.assertEqual(len(list(out.glob('attempt-*'))), 2)

    def test_argument_is_literal_and_timeout_retains_streams(self):
        p = self.module()
        arg = 'space ; $(touch NEVER)'
        out = self.root / 'literal'
        p.run_job([sys.executable, '-c', 'import sys; print(sys.argv[1])', arg], out, self.env,
                  lambda d: {'ok': True}, 5, 100000)
        self.assertEqual((out / 'attempt-001/stdout.txt').read_text().strip(), arg)
        with self.assertRaisesRegex(ValueError, 'timeout'):
            p.run_job([sys.executable, '-u', '-c', 'import time; print("before"); time.sleep(20)'],
                      self.root / 'timeout', self.env, lambda d: {}, .2, 100000)
        self.assertIn('before', (self.root / 'timeout/attempt-001/stdout.txt').read_text())

    def test_completed_job_rejects_changed_command(self):
        p = self.module()
        out = self.root / 'job'
        p.run_job([sys.executable, '-c', 'print(1)'], out, self.env, lambda d: {}, 5, 100000)
        with self.assertRaisesRegex(ValueError, 'command'):
            p.run_job([sys.executable, '-c', 'print(2)'], out, self.env, lambda d: {}, 5, 100000, resume=True)

    def test_client_cache_paths_are_explicit_and_offline(self):
        p = self.module()
        m = {'client_env': self.env, 'dataset_cache': str(self.root / 'hf'),
             'hf_hub_cache': '/cache/hub', 'hf_datasets_cache': '/cache/datasets'}
        env = p.client_environment(m, self.root)
        self.assertEqual(env['HF_HUB_CACHE'], '/cache/hub')
        self.assertEqual(env['HF_DATASETS_CACHE'], '/cache/datasets')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')

    def test_accuracy_uses_task_modes_official_stops_and_eight_concurrent(self):
        p = self.module()
        jobs = p.c.make_workloads('python', '/model', 'http://127.0.0.1:8000', self.root)
        gsm, aime = jobs[:2]
        for job in (gsm, aime):
            args = job['argv']
            self.assertEqual(json.loads(args[args.index('--model_args') + 1])['num_concurrent'], 8)
        self.assertEqual(gsm['argv'][gsm['argv'].index('--model') + 1], 'local-completions')
        self.assertNotIn('--apply_chat_template', gsm['argv'])
        self.assertIn('--apply_chat_template', aime['argv'])
        self.assertEqual(json.loads(gsm['argv'][gsm['argv'].index('--gen_kwargs') + 1])['max_gen_toks'], 256)
        self.assertEqual(json.loads(aime['argv'][aime['argv'].index('--gen_kwargs') + 1])['max_gen_toks'], 32768)

    def test_real_git_source_allows_only_receipted_generated_files(self):
        p = self.module()
        self.assertTrue(hasattr(p.c, 'verify_source'), 'native-overlay-aware source verifier is missing')
        repo = self.root / 'repo'
        repo.mkdir()
        def git(*argv):
            return subprocess.check_output(['git', *argv], cwd=repo, env=self.env, text=True).strip()
        git('init', '-q')
        (repo / 'model.py').write_text('fixed source\n')
        git('add', '.')
        git('-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixed')
        generated = repo / 'generated.py'
        generated.write_text('built wrapper\n')
        receipt = self.root / 'native.json'
        p.c.write(receipt, {'source_files': {'generated.py': p.c.sha256(generated)}})
        m = {'shared': {'commit': git('rev-parse', 'HEAD'),
                        'common_diff_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        'native_receipt_sha256': p.c.sha256(receipt)},
             'target_configs': ['target.json'], 'arms': {'baseline': {'repo': str(repo)}}}
        p.c.verify_source(m, 'baseline', self.env, str(receipt))
        (repo / 'model.py').write_text('unexpected edit\n')
        with self.assertRaisesRegex(ValueError, 'source diff'):
            p.c.verify_source(m, 'baseline', self.env, str(receipt))

    @unittest.skipUnless(sys.platform == 'linux', 'production service proof uses Linux /proc')
    def test_server_context_stops_own_service_and_preserves_unrelated_process(self):
        p = self.module()
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        script = self.root / 'server.py'
        script.write_text('from http.server import BaseHTTPRequestHandler,HTTPServer\n'
                          'class Handler(BaseHTTPRequestHandler):\n'
                          ' def do_GET(self):\n'
                          '  self.send_response(200);self.end_headers();self.wfile.write(b"ok")\n'
                          f'HTTPServer(("127.0.0.1",{port}),Handler).serve_forever()\n')
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            with p.running_server([sys.executable, str(script)], self.root / 'server', self.root,
                                  self.env, f'http://127.0.0.1:{port}', 10, 100000) as (server, log):
                self.assertIsNone(server.poll())
                self.assertTrue(log.is_file())
            self.assertIsNotNone(server.poll())
            self.assertIsNone(unrelated.poll())
            self.assertTrue((self.root / 'server/exit.json').is_file())
        finally:
            unrelated.terminate()
            unrelated.wait()

    def test_benchmark_receipt_changes_when_actual_prompt_changes(self):
        self.assertTrue((ROOT / 'bench_record.py').is_file(), 'actual prompt receipt wrapper missing')
        import bench_record
        from types import SimpleNamespace
        one = SimpleNamespace(prompt='first prompt', prompt_len=2, expected_output_len=1)
        other = SimpleNamespace(prompt='changed prompt', prompt_len=2, expected_output_len=1)
        receipt = bench_record.request_receipt([one])
        self.assertEqual(receipt[0]['input_len'], 2)
        self.assertNotEqual(receipt, bench_record.request_receipt([other]))

    def test_benchmark_recording_passes_original_requests_without_rng_change(self):
        import bench_record
        self.assertTrue(hasattr(bench_record, 'record_benchmark'), 'transparent async recorder missing')
        import asyncio
        import random
        from types import SimpleNamespace
        requests = [SimpleNamespace(prompt='unchanged', prompt_len=1, expected_output_len=2)]
        state = random.getstate()
        async def consumer(*args, **kwargs):
            self.assertIs(kwargs['input_requests'], requests)
            self.assertEqual(kwargs['input_requests'][0].prompt, 'unchanged')
            return 'official result'
        result = asyncio.run(bench_record.record_benchmark(consumer, self.root / 'receipt.json', input_requests=requests))
        self.assertEqual(result, 'official result')
        self.assertEqual(state, random.getstate())

    def test_benchmark_cache_salt_matches_arms_and_differs_by_shape_round(self):
        p = self.module()
        jobs = p.c.make_workloads('python', '/model', 'http://127.0.0.1:8000', self.root)
        bodies = [json.loads(j['argv'][j['argv'].index('--extra-body') + 1]) for j in jobs[2:]]
        self.assertEqual(len({body.get('cache_salt') for body in bodies}), 12)

    def test_runner_sigterm_saves_exit_and_stops_its_child(self):
        p = self.module()
        self.assertTrue(hasattr(p, 'install_signal_handlers'), 'runner SIGTERM cleanup is missing')
        out = self.root / 'signal-job'
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import run_pair;'
                'from pathlib import Path;run_pair.install_signal_handlers();'
                'run_pair.saved_process([sys.executable,"-u","-c",'
                '"import os,time;print(os.getpid(),flush=True);time.sleep(30)"],'
                'Path(sys.argv[2]),{"PATH":sys.argv[3],"LANG":"C.UTF-8"},60,100000)')
        runner = subprocess.Popen([sys.executable, '-c', code, str(ROOT), str(out), self.env['PATH']])
        import time
        deadline = time.time() + 5
        while time.time() < deadline and not (out / 'stdout.txt').exists():
            time.sleep(.02)
        while time.time() < deadline and not (out / 'stdout.txt').read_text().strip():
            time.sleep(.02)
        runner.terminate()
        self.assertEqual(runner.wait(timeout=10), 143)
        self.assertNotEqual(json.loads((out / 'exit.json').read_text())['returncode'], 0)

    def test_storage_accepts_existing_bind_alias_on_same_data_filesystem(self):
        p = self.module()
        data = self.root / 'data'
        data.mkdir()
        alias = self.root / 'readonly-cache-alias'
        alias.mkdir()
        env = {key: str(data / key) for key in ('TMPDIR', 'HF_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME')}
        m = {'storage': {'data_root': str(data)}, 'dataset_cache': str(alias),
             'arms': {'baseline': {'env': env}, 'tuned': {'env': env}}}
        p.c.validate_storage_paths(m, data / 'results')

    def test_same_distribution_version_cannot_hide_wrong_client_import(self):
        p = self.module()
        self.assertTrue(hasattr(p, 'audit_client_provenance'), 'actual client import verifier missing')
        m = {'client_versions': {'lm_eval': '0.4.13', 'vllm': 'same-version'},
             'client_source_repo': '/current', 'client_package_root': '/current/vllm',
             'shared': {'commit': 'eb34e115d041ac685fa38eadf8eae6c51d61dc23'}}
        obj = {'versions': dict(m['client_versions']), 'vllm_file': '/old/vllm/__init__.py',
               'bench_serve_file': '/old/vllm/benchmarks/serve.py',
               'source': {'commit': m['shared']['commit']}}
        with self.assertRaisesRegex(ValueError, 'client.*import'):
            p.audit_client_provenance(obj, m)

    def test_failed_warmup_is_rejected_even_when_all_measured_requests_succeed(self):
        import bench_record
        import asyncio
        from types import SimpleNamespace
        request = SimpleNamespace(prompt='fixed', prompt_len=1, expected_output_len=1)
        started = 0
        async def request_func(*args, **kwargs):
            nonlocal started
            started += 1
            return SimpleNamespace(success=started not in (1, 3), error='request error' if started in (1, 3) else '')
        async def wait_for_endpoint(func, *args, **kwargs):
            first = await func(*args, **kwargs)
            return first if first.success else await func(*args, **kwargs)
        module = SimpleNamespace(ASYNC_REQUEST_FUNCS={'vllm': request_func}, wait_for_endpoint=wait_for_endpoint)
        async def official_fixture(**kwargs):
            await module.wait_for_endpoint(module.ASYNC_REQUEST_FUNCS['vllm'], request_func_input=request)
            for _ in range(16 + 128):
                await module.ASYNC_REQUEST_FUNCS['vllm'](request_func_input=request)
            return {'completed': 128, 'failed': 0}
        path = self.root / 'warmup-receipt.json'
        with self.assertRaisesRegex(ValueError, 'warmup'):
            asyncio.run(bench_record.record_benchmark(official_fixture, path, serve_module=module,
                        input_requests=[request] * 128, endpoint_type='vllm', num_warmups=16))
        obj = json.loads(path.read_text())
        self.assertEqual(sum(row['phase'] == 'readiness' for row in obj['request_outcomes']), 2)
        self.assertEqual(sum(row['phase'] == 'warmup' for row in obj['request_outcomes']), 16)
        self.assertEqual(sum(row['phase'] == 'measured' and row['success'] for row in obj['request_outcomes']), 128)

    def test_startup_snapshot_keeps_captured_bytes_when_live_log_appends(self):
        p = self.module()
        self.assertTrue(hasattr(p, 'capture_startup_log'), 'immutable startup capture is missing')
        live = self.root / 'server.log'
        live.write_bytes(b'actual backend/config startup\n')
        snapshot = self.root / 'startup.snapshot'
        p.capture_startup_log(live, snapshot)
        digest = p.c.sha256(snapshot)
        with live.open('ab') as stream:
            stream.write(b'periodic stats appended later\n')
        self.assertEqual(snapshot.read_bytes(), b'actual backend/config startup\n')
        self.assertEqual(p.c.sha256(snapshot), digest)
        self.assertIn(b'periodic stats', live.read_bytes())


if __name__ == '__main__':
    unittest.main()
