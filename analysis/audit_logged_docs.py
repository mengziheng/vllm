#!/usr/bin/env python3
"""Compare four complete logged test-document sets with stored preparation digests.

Reads saved model logs only. Does not load datasets/caches, run tasks or scoring,
transform document fields, or reconstruct GSM8K training/few-shot documents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

TASKS = {
    'gsm8k': ('openai/gsm8k', 1319, ('strict-match', 'flexible-extract'),
              '32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595'),
    'aime25': ('math-ai/aime25', 30, ('none',),
               '93226328a3bc794c9b7cf8ccd90beb65491bafa0ac58b132f3c11fe9976131b9'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def canonical_docs(path, count, filters):
    docs, groups, key_sets = {}, {name: set() for name in filters}, set()
    physical_rows = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id, name = row['doc_id'], row['filter']
            require(type(doc_id) is int, 'doc_id must be an integer, not a boolean/string')
            require(name in groups and doc_id not in groups[name], 'unexpected filter or duplicate doc_id within filter')
            require(isinstance(row['doc'], dict), 'logged doc must be the full document object')
            value = json.dumps(row['doc'], ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
            require(doc_id not in docs or docs[doc_id] == value, 'different doc objects across filters')
            docs[doc_id] = value
            groups[name].add(doc_id)
            key_sets.add(tuple(sorted(row['doc'])))
            physical_rows += 1
    require(all(ids == set(range(count)) for ids in groups.values()), 'incomplete filter doc_id coverage')
    digest = hashlib.sha256()
    for doc_id in range(count):
        digest.update(docs[doc_id])
    return {'physical_log_rows': physical_rows, 'unique_docs': len(docs),
            'filter_counts': {name: len(ids) for name, ids in groups.items()},
            'document_key_sets': [list(keys) for keys in sorted(key_sets)],
            'canonical_rows_sha256': digest.hexdigest()}


def audit_pair(pair, manifest):
    sources = read(manifest)['datasets']
    identity = read(pair / 'identity.json')
    result = {'pair_id': identity['manifest']['pair_id'], 'tasks': {},
              'scope': 'Four full logged test doc sets only; not scoring or overall pair completion.',
              'canonical_format': 'Full doc JSON, sorted keys, UTF-8 unicode, comma/colon separators; LF after every unique doc in ascending integer doc_id order.',
              'gsm8k_train': 'Not reconstructed; reuse the preparation receipt for train/few-shot source data.'}
    for task, (repo, count, filters, pinned_digest) in TASKS.items():
        selected = [source for source in sources if source['repo_id'] == repo]
        require(len(selected) == 1, 'exactly one prepared dataset source required')
        source = selected[0]
        expected = source['splits']['test']
        require(expected['rows'] == count and expected['canonical_rows_sha256'] == pinned_digest,
                'stored preparation test split differs from the pinned expected count/digest')
        result['tasks'][task] = {}
        for arm in ('baseline', 'tuned'):
            job = pair / arm / ('accuracy-' + task)
            receipt = read(job / 'complete.json')
            require(re.fullmatch(r'attempt-[0-9]{3,}', receipt['attempt']) is not None, 'unsafe selected attempt')
            attempt = job / receipt['attempt']
            status = read(attempt / 'exit.json')
            require(status['returncode'] == 0 and not status.get('reason'), 'selected accuracy command did not finish successfully')
            samples = list((attempt / 'results').rglob('*samples*' + task + '*.jsonl'))
            require(len(samples) == 1, 'exactly one complete official sample log required')
            actual = canonical_docs(samples[0], count, filters)
            result['tasks'][task][arm] = {**actual, 'dataset_repo': repo, 'dataset_revision': source['revision'],
                'split': 'test', 'expected_sha256': pinned_digest,
                'matches_prepared_test_split': actual['canonical_rows_sha256'] == pinned_digest,
                'samples_file': str(samples[0].relative_to(pair))}
    result['status'] = 'matched' if all(row['matches_prepared_test_split']
        for arms in result['tasks'].values() for row in arms.values()) else 'mismatch'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_directory', type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] /
                        'evidence/offline-dataset-preparation/official-data-manifest.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = audit_pair(args.pair_directory, args.manifest)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            temporary = args.output.with_name(args.output.name + '.tmp')
            temporary.write_text(text)
            temporary.replace(args.output)
        else:
            print(text, end='')
        return int(result['status'] != 'matched')
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
