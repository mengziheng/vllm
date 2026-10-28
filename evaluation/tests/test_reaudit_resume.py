import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare as c
import run_pair as p


class ReauditResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = {'PATH': os.environ['PATH'], 'LANG': 'C.UTF-8'}
        self.counter = self.root / 'request-count'
        self.argv = [sys.executable, '-c',
                     'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_text(str(int(p.read_text())+1) if p.exists() else "1"); print("unchanged raw output")',
                     str(self.counter)]
        native = self.root / 'native-receipt.json'
        c.write(native, {'fixture_only': True})
        self.shared = {'commit': 'eb34e115d041ac685fa38eadf8eae6c51d61dc23',
                       'native_receipt_sha256': c.sha256(native), 'model_revision': 'fixture-structural-receipt',
                       'hardware': {'fixture_only': True, 'gpu_uuids': ['fixture-a', 'fixture-b', 'fixture-c', 'fixture-d']}}
        self.native = native
        self.old_context = self.make_context('original', 100)
        self.new_context = self.make_context('current', 200)

    def tearDown(self):
        self.tmp.cleanup()

    def make_context(self, name, pid):
        directory = self.root / name
        directory.mkdir()
        log = directory / 'startup.snapshot'
        log.write_text('fixture captured startup ' + name)
        proof = {'pair_id': 'fixture-pair', 'arm': 'baseline', 'endpoint': 'http://127.0.0.1:8000',
                 'model': '/fixture-model', 'shared': copy.deepcopy(self.shared), 'repo': '/fixture-source',
                 'python': sys.executable, 'package_root': '/fixture-source/vllm', 'env': self.env,
                 'actual_server_env': self.env, 'source_verified': True, 'config_sha256': {},
                 'native_receipt': str(self.native), 'server_log_snapshot': str(log),
                 'server_log_sha256': c.sha256(log), 'server_pid': pid, 'server_start_ticks': str(pid)}
        path = directory / 'proof.json'
        c.write(path, proof)
        return {'proof': str(path), 'proof_sha256': c.sha256(path)}

    def previous_exit_zero_audit_failure(self):
        def old_audit(directory):
            raise ValueError('old audit schema rejected official rows')
        with self.assertRaisesRegex(ValueError, 'old audit schema'):
            p.run_job(self.argv, self.root / 'job', self.env, old_audit, 5, 100000, context=self.old_context)

    def test_resume_reaudits_original_exit_zero_attempt_without_new_request(self):
        self.previous_exit_zero_audit_failure()
        original = (self.root / 'job/attempt-001/stdout.txt').read_bytes()
        result = p.run_job(self.argv, self.root / 'job', self.env, lambda directory: {'official_raw_valid': True},
                           5, 100000, resume=True, context=self.new_context)
        self.assertEqual(result, {'official_raw_valid': True})
        self.assertEqual(self.counter.read_text(), '1')
        self.assertEqual(original, (self.root / 'job/attempt-001/stdout.txt').read_bytes())
        receipt = c.read(self.root / 'job/complete.json')
        self.assertEqual(receipt['attempt'], 'attempt-001')
        self.assertIn('re_audit', receipt)
        self.assertIn('without rerunning', receipt['re_audit']['reason'])

    def test_changed_current_proof_is_rejected_without_request(self):
        self.previous_exit_zero_audit_failure()
        path = Path(self.new_context['proof'])
        proof = c.read(path)
        proof['shared']['commit'] = 'f' * 40
        c.write(path, proof)
        self.new_context['proof_sha256'] = c.sha256(path)
        with self.assertRaisesRegex(ValueError, 'proof'):
            p.run_job(self.argv, self.root / 'job', self.env, lambda directory: {},
                      5, 100000, resume=True, context=self.new_context)
        self.assertEqual(self.counter.read_text(), '1')

    def test_changed_original_startup_snapshot_is_rejected_without_request(self):
        self.previous_exit_zero_audit_failure()
        (self.root / 'original/startup.snapshot').write_text('corrupted original startup')
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            p.run_job(self.argv, self.root / 'job', self.env, lambda directory: {},
                      5, 100000, resume=True, context=self.new_context)
        self.assertEqual(self.counter.read_text(), '1')

    def test_still_invalid_raw_is_not_retried_to_select_another_result(self):
        self.previous_exit_zero_audit_failure()
        def still_invalid(directory):
            raise ValueError('raw audit still invalid')
        with self.assertRaisesRegex(ValueError, 'raw audit still invalid'):
            p.run_job(self.argv, self.root / 'job', self.env, still_invalid,
                      5, 100000, resume=True, context=self.new_context)
        self.assertEqual(self.counter.read_text(), '1')
        self.assertFalse((self.root / 'job/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
