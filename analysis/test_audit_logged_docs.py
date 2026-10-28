"""Synthetic two-document checks only; never load a dataset or its cache."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
# Independently derived with shasum over literal UTF-8 JSON lines, including final LF.
EXPECTED = '5cdfe5970505a5bdedc9eb69c56149985e9a8d926a07ff2a74d85f02ce9780d6'
FILTERS = ('strict-match', 'flexible-extract')


class LoggedDocTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-logged-docs-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic-only.jsonl'

    def module(self):
        path = ROOT / 'audit_logged_docs.py'
        self.assertTrue(path.exists(), 'logged-doc auditor is not implemented')
        spec = importlib.util.spec_from_file_location('logged_doc_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def rows(self):
        return [{'doc_id': n, 'filter': name, 'doc': doc}
                for name in FILTERS for n, doc in [(1, {'b': 2}), (0, {'z': 1, 'a': 'é'})]]

    def digest(self, rows):
        self.path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
        return self.module().canonical_docs(self.path, 2, FILTERS)

    def test_orders_unique_integer_ids_and_preserves_unicode_and_last_newline(self):
        result = self.digest(self.rows())
        self.assertEqual(result['canonical_rows_sha256'], EXPECTED)
        self.assertEqual(result['physical_log_rows'], 4)
        self.assertEqual(result['unique_docs'], 2)
        self.assertEqual(result['filter_counts'], {'strict-match': 2, 'flexible-extract': 2})

    def test_rejects_same_filter_duplicate_and_cross_filter_doc_change(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.digest(rows + [dict(rows[0])])
        rows[-1] = {**rows[-1], 'doc': {'z': 2, 'a': 'é'}}
        with self.assertRaises(ValueError):
            self.digest(rows)

    def test_rejects_missing_filter_coverage_and_boolean_doc_id(self):
        with self.assertRaises(ValueError):
            self.digest(self.rows()[:-1])
        rows = self.rows()
        rows[0] = {**rows[0], 'doc_id': True}
        with self.assertRaises(ValueError):
            self.digest(rows)

    def test_extra_fields_are_hashed_without_projection(self):
        rows = self.rows()
        for row in rows:
            row['doc'] = {**row['doc'], 'extra': 'must remain present'}
        self.assertNotEqual(self.digest(rows)['canonical_rows_sha256'], EXPECTED)


if __name__ == '__main__':
    unittest.main()
