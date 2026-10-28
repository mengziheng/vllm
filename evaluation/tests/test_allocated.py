import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare as c


class AllocatedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_launcher_storage_uses_real_current_data_device_and_rejects_other_run(self):
        cache = self.root / 'cache'
        cache.mkdir()
        receipt = self.root / 'launcher.json'
        c.write(receipt, {'name': 'current-run', 'status': 'running', 'storage': {
            'storage_ready': True, 'data_storage': {'device': self.root.stat().st_dev},
            'docker_storage': {'device': self.root.stat().st_dev + 1, 'root': '/host/docker'}}})
        env = {key: str(cache) for key in ('TMPDIR', 'HF_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'XDG_CACHE_HOME')}
        m = {'dataset_cache': str(cache), 'arms': {'baseline': {'env': env}, 'tuned': {'env': env}},
             'storage': {'data_root': str(self.root), 'launcher_receipt': str(receipt), 'budget_bytes': 0,
                         'min_free_bytes': 0, 'retention_days': 14}}
        with patch.dict(os.environ, {'OPERATOR_SWEEP_RUN_ID': 'current-run'}):
            result = c.storage_preflight(m, self.root / 'output')
            self.assertEqual(result['data_device'], self.root.stat().st_dev)
        with patch.dict(os.environ, {'OPERATOR_SWEEP_RUN_ID': 'other-run'}):
            with self.assertRaisesRegex(ValueError, 'run'):
                c.storage_preflight(m, self.root / 'output')

    def test_cache_copy_preserves_hub_symlink_and_reuses_completed_copy(self):
        self.assertTrue((ROOT / 'allocated_pair.py').exists(), 'allocated entry is missing')
        import allocated_pair as a
        source = self.root / 'input-cache'
        (source / 'hub/blobs').mkdir(parents=True)
        (source / 'hub/blobs/one').write_text('cache bytes')
        (source / 'hub/link').symlink_to('blobs/one')
        dest = self.root / 'copy'
        first = a.copy_cache(source, dest)
        self.assertTrue((dest / 'hub/link').is_symlink())
        self.assertEqual((dest / 'hub/link').read_text(), 'cache bytes')
        self.assertEqual(first, a.copy_cache(source, dest))

    def test_model_snapshot_stats_shards_and_detects_change_without_body_hash(self):
        self.assertTrue(hasattr(c, 'model_snapshot'), 'model receipt/stat adapter missing')
        model = self.root / 'model'
        model.mkdir()
        shard = model / 'one.safetensors'
        shard.write_bytes(b'weights fixture')
        (model / 'config.json').write_text('{}')
        receipt = self.root / 'model-receipt.json'
        c.write(receipt, {'files': {'one.safetensors': {'size': 15}},
                         'metadata_sha256': {'config.json': c.sha256(model / 'config.json')},
                         'errors': [], 'verification_scope': 'fixture structural receipt'})
        original_sha = c.sha256
        def small_only(path):
            self.assertFalse(str(path).endswith('.safetensors'), 'weight body must not be hashed')
            return original_sha(path)
        with patch.object(c, 'sha256', side_effect=small_only):
            snapshot = c.model_snapshot(model, receipt)
        c.check_model_snapshot(snapshot, model)
        shard.write_bytes(b'changed length')
        with self.assertRaisesRegex(ValueError, 'model file changed'):
            c.check_model_snapshot(snapshot, model)

    def test_startup_client_sends_one_token_request_and_records_raw_response(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'text': 'one token'}], 'usage': {'completion_tokens': 1}}).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            process = subprocess.run([sys.executable, str(ROOT / 'smoke_request.py'), '--endpoint',
                                      f'http://127.0.0.1:{server.server_port}', '--model', '/fixture-model'],
                                     text=True, capture_output=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(received[0]['max_tokens'], 1)
            self.assertEqual(received[0]['seed'], 20261028)
            self.assertEqual(json.loads(process.stdout)['response']['usage']['completion_tokens'], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
