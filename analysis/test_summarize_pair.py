"""Synthetic fixtures only: no model result or performance claim is stored here."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
import compare


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic-summary-only-')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def module(self):
        helper = ROOT / 'analysis/summarize_pair.py'
        self.assertTrue(helper.exists(), 'summary helper is not implemented')
        spec = importlib.util.spec_from_file_location('summary_under_test', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def summary(self):
        return self.module().summarize(self.root)

    def fixture(self):
        """Full required sample counts, but deliberately artificial responses/numbers."""
        m = {'pair_id': 'SYNTHETIC_ONLY', 'shared': {'commit': 'e' * 40}, 'client_python': 'python'}
        identity = {'manifest': m, 'model': '/synthetic/model', 'endpoint': 'http://127.0.0.1:8000'}
        write(self.root / 'identity.json', identity)
        plan, audits = {}, {}
        for arm in ('baseline', 'tuned'):
            jobs = compare.make_workloads('python', identity['model'], identity['endpoint'], self.root / arm)
            plan[arm] = {'workloads': jobs}
            proof = {'pair_id': m['pair_id'], 'arm': arm, 'model': identity['model'],
                     'shared': m['shared'], 'source_verified': True}
            proof_path = self.root / arm / 'server-001/proof.json'
            write(proof_path, proof)
            rows = []
            for job in jobs:
                directory = self.root / arm / job['name']
                attempt = directory / 'attempt-001'
                results = attempt / 'results'
                write(attempt / 'exit.json', {'returncode': 0, 'reason': None})
                write(attempt / 'server-context.json', {'proof': str(proof_path), 'proof_sha256': compare.sha256(proof_path)})
                if job['kind'] == 'accuracy':
                    task, count = job['expected']['task'], job['expected']['count']
                    score = .25 if arm == 'baseline' else .5
                    filters = ('strict-match', 'flexible-extract') if task == 'gsm8k' else ('none',)
                    metrics = {'alias': task}
                    for name in filters:
                        metrics['exact_match,' + name] = score + (.125 if name == 'flexible-extract' else 0)
                        metrics['exact_match_stderr,' + name] = .01
                    write(results / 'results_fixture.json', {'results': {task: metrics},
                        'n-samples': {task: {'original': count, 'effective': count}}})
                    samples = results / ('samples_' + task + '.jsonl')
                    samples.write_text('\n'.join(json.dumps({'doc_id': n, 'filter': name,
                        'doc': {'question': str(n)}, 'arguments': {'gen_args_0': {
                            'arg_0': f'synthetic prompt {n}', 'arg_1': {'temperature': 0}}},
                        'target': 'fixture reference', 'resps': [[arm + ' SYNTHETIC ONLY']],
                        **({'doc_hash': f'{n:064x}', 'prompt_hash': f'{n + 10000:064x}',
                            'target_hash': 'f' * 64} if task == 'gsm8k' else {}),
                        'filtered_resps': [name + ' synthetic extraction']})
                        for name in filters for n in range(count)))
                    audit = {'task': task, 'samples': count, 'full_dataset': True,
                             'logged_sample_rows': count * len(filters), 'filters': {name: count for name in filters}}
                else:
                    il, ol = job['expected']['input_len'], job['expected']['output_len']
                    repeat = int(job['name'].split('-')[1][1:])
                    # Non-sorted repeats make last-value/mean/arm swaps observable.
                    rate = {'baseline': (100, 300, 200), 'tuned': (400, 300, 500)}[arm][repeat - 1]
                    write(results / 'benchmark.json', {'completed': 128, 'failed': 0, 'num_prompts': 128,
                        'input_lens': [il] * 128, 'output_lens': [ol] * 128, 'errors': [''] * 128,
                        'duration': 128 / rate, 'request_throughput': rate,
                        'output_throughput': rate * ol, 'total_token_throughput': rate * (il + ol)})
                    requests = [{'index': n, 'input_len': il, 'output_len': ol, 'prompt_sha256': f'{repeat:064x}'} for n in range(128)]
                    outcomes = [{'phase': phase, 'success': True, 'error': ''} for phase, count in [('warmup', 16), ('measured', 128)] for _ in range(count)]
                    write(results / 'input-receipt.json', {'requests': requests, 'request_outcomes': outcomes})
                    audit = {'requests': 128, 'input_len': il, 'output_len': ol, 'failed': 0,
                             'actual_input_sha256': [f'{repeat:064x}'] * 128, 'warmups': 16, 'warmup_failed': 0}
                write(directory / 'complete.json', {'attempt': 'attempt-001', 'audit': audit})
                rows.append({'name': job['name'], 'audit': audit})
            audits[arm] = rows
            write(self.root / arm / 'status.json', {'status': 'passed', 'jobs': 14})
            write(self.root / arm / 'audit.json', rows)
        write(self.root / 'plan.json', plan)
        write(self.root / 'pair-audit.json', {'status': 'paired-audits-passed', 'pair_id': m['pair_id'], 'audits': audits})

    def sample_log(self, rows):
        path = self.root / 'SYNTHETIC_ONLY_samples.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        return path

    def filter_rows(self):
        # Actual GSM8K logging shape, with deliberately artificial content.
        return [{'doc_id': n, 'filter': name, 'doc': {'question': str(n)},
                 'arguments': {'gen_args_0': {'arg_0': f'synthetic prompt {n}'}}, 'target': str(n),
                 'resps': [[f'synthetic response {n}']]}
                for name in ('strict-match', 'flexible-extract') for n in range(1319)]

    def test_two_official_filters_count_2638_rows_as_1319_docs(self):
        inputs, sources, counts = self.module().sample_inputs(self.sample_log(self.filter_rows()))
        self.assertEqual(len(inputs), 1319)
        self.assertEqual(counts, {'physical_log_rows': 2638, 'unique_docs': 1319,
                                 'filter_counts': {'strict-match': 1319, 'flexible-extract': 1319}})
        self.assertEqual(sources['prompt_hash'], ['sha256:arguments'])

    def test_single_filter_parser_fixture_is_not_a_full_gsm8k_claim(self):
        # Isolated parser case; no completed-pair receipt or official GSM8K claim.
        rows = [row for row in self.filter_rows() if row['filter'] == 'strict-match'][:3]
        inputs, _, counts = self.module().sample_inputs(self.sample_log(rows))
        self.assertEqual(len(inputs), 3)
        self.assertEqual(counts, {'physical_log_rows': 3, 'unique_docs': 3,
                                 'filter_counts': {'strict-match': 3}})

    def test_rejects_duplicate_doc_within_one_filter(self):
        rows = self.filter_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, 'filter'):
            self.module().sample_inputs(self.sample_log(rows))

    def test_rejects_cross_filter_input_reference_or_raw_response_change(self):
        original = self.filter_rows()
        for field, value in [('doc', {'question': 'changed'}), ('arguments', [['changed prompt']]),
                             ('target', 'changed reference'), ('resps', [['changed response']]),
                             ('doc_hash', 'a different official hash')]:
            rows = [dict(row) for row in original]
            rows[1319][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.module().sample_inputs(self.sample_log(rows))

    def test_hand_derived_medians_arm_direction_and_official_accuracy(self):
        self.fixture()
        result = self.summary()
        self.assertEqual(set(result['benchmarks']), {'8192-1', '2048-512', '512-1024', '512-128'})
        shape = result['benchmarks']['512-128']
        self.assertEqual(shape['baseline']['request_throughput'], {'median': 200, 'min': 100, 'max': 300})
        self.assertEqual(shape['tuned']['request_throughput'], {'median': 400, 'min': 300, 'max': 500})
        self.assertEqual(shape['baseline']['output_throughput']['median'], 25600)
        self.assertEqual(shape['tuned']['total_token_throughput']['median'], 256000)
        self.assertEqual(shape['median_change_percent']['request_throughput'], 100)
        self.assertEqual(shape['median_tuned_over_baseline_percent']['request_throughput'], 200)
        self.assertEqual([r['request_throughput'] for r in shape['baseline']['runs']], [100, 300, 200])
        self.assertEqual(result['accuracy']['gsm8k']['baseline']['metrics']['exact_match,strict-match'], .25)
        self.assertEqual(result['accuracy']['gsm8k']['baseline']['metrics']['exact_match,flexible-extract'], .375)
        self.assertEqual(result['accuracy']['gsm8k']['tuned']['metrics']['exact_match,strict-match'], .5)
        self.assertEqual(result['accuracy']['gsm8k']['baseline']['stderr']['exact_match_stderr,strict-match'], .01)
        self.assertEqual(result['accuracy']['gsm8k']['baseline']['n_samples']['effective'], 1319)
        self.assertEqual(result['accuracy']['aime25']['tuned']['n_samples']['original'], 30)
        self.assertEqual(result['accuracy']['gsm8k']['input_audit']['samples'], 1319)
        self.assertEqual(result['accuracy']['gsm8k']['baseline']['sample_log'], {
            'physical_log_rows': 2638, 'unique_docs': 1319,
            'filter_counts': {'strict-match': 1319, 'flexible-extract': 1319}})
        self.assertEqual(result['accuracy']['aime25']['tuned']['sample_log'], {
            'physical_log_rows': 30, 'unique_docs': 30, 'filter_counts': {'none': 30}})
        self.assertEqual(result['accuracy']['gsm8k']['input_audit']['sources']['prompt_hash'], ['official:prompt_hash'])
        self.assertEqual(result['accuracy']['aime25']['input_audit']['sources']['prompt_hash'], ['sha256:arguments'])

    def test_absent_pair_has_no_partial_summary(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.summary()

    def test_rejects_missing_or_duplicate_repeat(self):
        self.fixture()
        plan = json.loads((self.root / 'plan.json').read_text())
        plan['tuned']['workloads'][-1] = plan['tuned']['workloads'][-5]
        write(self.root / 'plan.json', plan)
        with self.assertRaises(ValueError):
            self.summary()

    def test_rejects_missing_completion_receipt(self):
        self.fixture()
        (self.root / 'baseline/bench-r3-512-128/complete.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.summary()

    def test_reaudits_failed_requests_shapes_and_warmups(self):
        self.fixture()
        path = self.root / 'tuned/bench-r2-512-128/attempt-001/results/benchmark.json'
        original = json.loads(path.read_text())
        for key, value in [('failed', 1), ('input_lens', [513] * 128), ('output_lens', [129] * 128), ('errors', ['failed'] * 128)]:
            write(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summary()
        write(path, original)
        receipt = path.with_name('input-receipt.json')
        data = json.loads(receipt.read_text())
        data['request_outcomes'][0]['success'] = False
        write(receipt, data)
        with self.assertRaises(ValueError):
            self.summary()

    def test_rejects_swapped_arm_proof(self):
        self.fixture()
        proof = self.root / 'baseline/server-001/proof.json'
        obj = json.loads(proof.read_text())
        obj['arm'] = 'tuned'
        write(proof, obj)
        # Updating the receipt hash must not make wrong-arm provenance valid.
        context = self.root / 'baseline/accuracy-gsm8k/attempt-001/server-context.json'
        write(context, {'proof': str(proof), 'proof_sha256': compare.sha256(proof)})
        with self.assertRaises(ValueError):
            self.summary()

    def test_rejects_unpaired_or_stale_audit(self):
        self.fixture()
        path = self.root / 'pair-audit.json'
        obj = json.loads(path.read_text())
        obj['audits']['tuned'][0]['audit']['samples'] = 30
        write(path, obj)
        with self.assertRaises(ValueError):
            self.summary()

    def test_rejects_changed_actual_prompt_or_reference_without_rejecting_responses(self):
        self.fixture()
        path = self.root / 'tuned/accuracy-aime25/attempt-001/results/samples_aime25.jsonl'
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for field, value in [('arguments', [['different prompt']]), ('target', 'different reference')]:
            rows = [dict(row) for row in original]
            rows[0][field] = value
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summary()

    def test_prefers_official_hashes_and_records_the_source(self):
        self.fixture()
        for arm in ('baseline', 'tuned'):
            path = self.root / arm / 'accuracy-aime25/attempt-001/results/samples_aime25.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row.update(doc_hash='d' * 64, prompt_hash='a' * 64, target_hash='b' * 64)
                for field in ('doc', 'arguments', 'target'):
                    row.pop(field)
            path.write_text('\n'.join(json.dumps(row) for row in rows))
        audit = self.summary()['accuracy']['aime25']['input_audit']
        self.assertEqual(audit['sources']['prompt_hash'], ['official:prompt_hash'])
        self.assertEqual(audit['samples'], 30)

    def test_rejects_official_result_without_standard_error(self):
        self.fixture()
        path = self.root / 'baseline/accuracy-gsm8k/attempt-001/results/results_fixture.json'
        obj = json.loads(path.read_text())
        obj['results']['gsm8k'].pop('exact_match_stderr,strict-match')
        write(path, obj)
        with self.assertRaises(ValueError):
            self.summary()

    def test_cli_failure_preserves_existing_output(self):
        helper = ROOT / 'analysis/summarize_pair.py'
        path = self.root / 'summary.json'
        path.write_text('existing output must remain untouched')
        result = subprocess.run([sys.executable, str(helper), str(self.root), '--output', str(path)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(path.read_text(), 'existing output must remain untouched')
        self.assertFalse(path.with_name('summary.json.tmp').exists())

    def test_cli_failure_emits_no_json(self):
        helper = ROOT / 'analysis/summarize_pair.py'
        self.assertTrue(helper.exists(), 'summary helper is not implemented')
        result = subprocess.run([sys.executable, str(helper), str(self.root)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('ERROR:', result.stderr)


if __name__ == '__main__':
    unittest.main()
