import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('admission', Path(__file__).with_name('pool_legacy_admission.py'))
admission = importlib.util.module_from_spec(spec)
if spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(admission)

class LegacyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.base = self.state / 'staging' / 'moe58986' / '20261024-153729-7168ad'
        self.base.mkdir(parents=True)
        self.legacy = dict(task='moe58986', release='20261024-153729-7168ad', source_node='h20-04', source_root='/data03/tiger/vllm_pr58986_20261024', hosts={'h20-01':'/data03/task', 'h20-02':'/data03/task', 'h20-04':'/data03/task'}, errors={}, staged_bytes=123, node_root_overrides={})
        self.current = dict(pool_release_id=self.legacy['release'], framework='v032', source_hosts={'h20-04': self.legacy['source_root']}, hosts={k:v for k,v in self.legacy['hosts'].items() if k != 'h20-04'})
        self.write()
    def write(self):
        (self.base / 'PREPARATION.json').write_text(json.dumps(self.legacy))
    def resolve(self):
        def original(*args):
            raise KeyError('config_template')
        return admission.resolve_legacy(original, self.state, 'moe58986', self.legacy['release'], self.current)
    def test_exact_current_legacy_release_preserves_current_ready_hosts(self):
        self.assertEqual(self.resolve(), self.current)
    def test_superseded_release_is_rejected(self):
        self.current['pool_release_id'] = 'new-release'
        with self.assertRaises(ValueError): self.resolve()
    def test_changed_host_is_rejected(self):
        self.current['hosts']['h20-01'] = '/other'
        with self.assertRaises(ValueError): self.resolve()
    def test_changed_source_is_rejected(self):
        self.current['source_hosts']['h20-04'] = '/other'
        with self.assertRaises(ValueError): self.resolve()
    def test_modern_receipt_errors_are_not_bypassed(self):
        self.legacy['config_template'] = {}
        self.write()
        with self.assertRaises(KeyError): self.resolve()
    def test_original_success_is_unchanged(self):
        expected={'modern':'result'}
        self.assertIs(admission.resolve_legacy(lambda *a:expected, self.state, 'other-task', 'release', {}), expected)
    def test_changed_receipt_digest_is_rejected(self):
        def original(*args): raise KeyError('config_template')
        with self.assertRaises(ValueError):
            admission.resolve_legacy(original, self.state, 'moe58986', self.legacy['release'], self.current, expected_receipt_sha256='0'*64)
    def test_supplement_is_not_silently_ignored(self):
        (self.base/'PREPARATION_SUPPLEMENT.json').write_text('{}')
        with self.assertRaises(ValueError): self.resolve()

if __name__ == '__main__': unittest.main()
