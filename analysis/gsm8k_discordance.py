#!/usr/bin/env python3
"""Compare existing official GSM8K per-sample scores without scoring again."""
import hashlib
import json
from pathlib import Path
import sys


ARMS = ('baseline', 'tuned')
FILTERS = ('strict-match', 'flexible-extract')
CATEGORIES = ('both_correct', 'baseline_only_correct', 'tuned_only_correct', 'both_wrong')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(sys.argv[1])
    samples, sources = {}, {}
    for arm in ARMS:
        paths = list((root / arm / 'accuracy-gsm8k/attempt-001/results/__model').glob('*samples*gsm8k*.jsonl'))
        require(len(paths) == 1, 'expected exactly one official GSM8K sample file: ' + arm)
        raw = paths[0].read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line]
        require(len(rows) == 2638, 'expected two filters over 1319 documents: ' + arm)
        require({row['filter'] for row in rows} == set(FILTERS), 'unexpected official filter set')
        samples[arm] = {}
        for name in FILTERS:
            selected = [row for row in rows if row['filter'] == name]
            require(len(selected) == 1319 and {row['doc_id'] for row in selected} == set(range(1319)),
                    'missing or duplicate per-filter document: ' + arm + '/' + name)
            require(all(type(row['exact_match']) in (int, float) and row['exact_match'] in (0, 1) for row in selected),
                    'invalid official per-sample exact_match score')
            samples[arm][name] = {row['doc_id']: row for row in selected}
        sources[arm] = {'path': str(paths[0]), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                        'physical_rows': len(rows)}
    report = {'schema_version': 1, 'pair_directory': str(root), 'task': 'gsm8k',
              'independent_documents': 1319, 'sources': sources,
              'method': 'Read existing official per-sample exact_match values; no model requests or rescoring.',
              'input_and_reference_comparison': 'doc, target, arguments, doc_hash, prompt_hash, target_hash must match across arms',
              'filters': {}, 'discordant_cases': []}
    for name in FILTERS:
        ids = {category: [] for category in CATEGORIES}
        for doc_id in range(1319):
            baseline, tuned = (samples[arm][name][doc_id] for arm in ARMS)
            for key in ('doc', 'target', 'arguments', 'doc_hash', 'prompt_hash', 'target_hash'):
                require(key in baseline and key in tuned and baseline[key] == tuned[key],
                        f'actual input/reference mismatch at {name}/{doc_id}/{key}')
            b, t = bool(baseline['exact_match']), bool(tuned['exact_match'])
            category = ('both_correct' if b and t else 'baseline_only_correct' if b else
                        'tuned_only_correct' if t else 'both_wrong')
            ids[category].append(doc_id)
            if b != t:
                report['discordant_cases'].append({'filter': name, 'doc_id': doc_id, 'category': category,
                    'reference': {key: baseline[key] for key in ('doc', 'target', 'doc_hash', 'prompt_hash', 'target_hash')},
                    'arguments': baseline['arguments'],
                    **{arm: {key: row[key] for key in ('exact_match', 'filtered_resps', 'resps', 'metrics')}
                       for arm, row in zip(ARMS, (baseline, tuned))}})
        counts = {category: len(values) for category, values in ids.items()}
        baseline_correct = counts['both_correct'] + counts['baseline_only_correct']
        tuned_correct = counts['both_correct'] + counts['tuned_only_correct']
        report['filters'][name] = {'counts': counts, 'doc_ids': ids, 'baseline_correct': baseline_correct,
            'tuned_correct': tuned_correct, 'tuned_minus_baseline_correct': tuned_correct - baseline_correct,
            'tuned_minus_baseline_percentage_points': 100 * (tuned_correct - baseline_correct) / 1319}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
