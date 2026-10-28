import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare as c


class AccuracySchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_fixture(self, task='gsm8k', count=1319, filters=('strict-match', 'flexible-extract')):
        c.write(self.directory / 'results_fixture.json', {
            'results': {task: {'exact_match,' + name: 0.5 for name in filters}},
            'n-samples': {task: {'original': count, 'effective': count}}})
        rows = []
        for name in filters:
            for doc_id in range(count):
                rows.append({'doc_id': doc_id, 'doc': {'question': 'official-format fixture', 'answer': str(doc_id)},
                             'target': str(doc_id), 'arguments': {'gen_args_0': {'arg_0': f'prompt {doc_id}', 'arg_1': {'temperature': 0}}},
                             'resps': [[f'raw answer {doc_id}']], 'filtered_resps': [str(doc_id) if name == 'flexible-extract' else ''],
                             'filter': name, 'metrics': ['exact_match'], 'exact_match': int(name == 'flexible-extract'),
                             'doc_hash': f'doc-{doc_id}', 'prompt_hash': f'prompt-{doc_id}', 'target_hash': f'target-{doc_id}'})
        self.save_rows(task, rows)
        return rows

    def save_rows(self, task, rows):
        (self.directory / f'samples_{task}_fixture.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_full_gsm_two_filters_are_2638_rows_and_1319_unique_docs_each(self):
        self.write_fixture()
        result = c.audit_accuracy(self.directory, 'gsm8k', 1319)
        self.assertEqual(result, {'task': 'gsm8k', 'samples': 1319, 'full_dataset': True,
                                  'logged_sample_rows': 2638, 'filters': {'strict-match': 1319, 'flexible-extract': 1319}})

    def test_full_aime_default_none_filter_remains_30_rows(self):
        self.write_fixture('aime25', 30, ('none',))
        self.assertEqual(c.audit_accuracy(self.directory, 'aime25', 30)['filters'], {'none': 30})

    def test_missing_filter_cannot_look_like_complete_gsm(self):
        self.write_fixture(filters=('strict-match',))
        with self.assertRaisesRegex(ValueError, 'filter set'):
            c.audit_accuracy(self.directory, 'gsm8k', 1319)

    def test_duplicate_or_shifted_doc_inside_one_filter_is_rejected(self):
        rows = self.write_fixture()
        rows[-1]['doc_id'] = 0
        self.save_rows('gsm8k', rows)
        with self.assertRaisesRegex(ValueError, 'doc coverage'):
            c.audit_accuracy(self.directory, 'gsm8k', 1319)

    def test_cross_filter_raw_response_difference_is_rejected(self):
        rows = self.write_fixture()
        rows[-1]['resps'] = [['different model response']]
        self.save_rows('gsm8k', rows)
        with self.assertRaisesRegex(ValueError, 'cross-filter'):
            c.audit_accuracy(self.directory, 'gsm8k', 1319)

    def test_cross_filter_input_or_target_difference_is_rejected(self):
        rows = self.write_fixture()
        rows[-1]['arguments']['gen_args_0']['arg_0'] = 'changed prompt'
        self.save_rows('gsm8k', rows)
        with self.assertRaisesRegex(ValueError, 'cross-filter'):
            c.audit_accuracy(self.directory, 'gsm8k', 1319)


if __name__ == '__main__':
    unittest.main()
