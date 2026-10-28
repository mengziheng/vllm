#!/usr/bin/env python3
"""Summarize a complete run_pair output; never emit a partial performance claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import compare as c
import run_pair

ARMS = ('baseline', 'tuned')
THROUGHPUT = ('request_throughput', 'output_throughput', 'total_token_throughput')


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sample_inputs(path):
    inputs, payloads, filters = {}, {}, {}
    sources = {key: set() for key in ('doc_hash', 'prompt_hash', 'target_hash')}
    physical_rows = 0
    for row in (json.loads(line) for line in path.read_text().splitlines() if line):
        fields = {}
        for key, fallback in (('doc_hash', 'doc'), ('prompt_hash', 'arguments'), ('target_hash', 'target')):
            if isinstance(row.get(key), str) and row[key]:
                value, source = row[key], 'official:' + key
            else:
                c.require(fallback in row, 'sample lacks actual input/reference: ' + fallback)
                value, source = stable_hash(row[fallback]), 'sha256:' + fallback
            fields[key] = value
            sources[key].add(source)
        doc_id, name = row['doc_id'], row.get('filter', '<unspecified>')
        c.require(isinstance(name, str) and name, 'invalid accuracy filter')
        seen = filters.setdefault(name, set())
        c.require(doc_id not in seen, 'duplicate accuracy doc_id within filter: ' + name)
        seen.add(doc_id)
        payload = stable_hash({key: row[key] for key in ('doc', 'arguments', 'target', 'resps') if key in row})
        if doc_id in inputs:
            c.require(inputs[doc_id] == fields and payloads[doc_id] == payload,
                      'accuracy input/reference/raw response differs across filters')
        inputs[doc_id], payloads[doc_id] = fields, payload
        physical_rows += 1
    counts = {'physical_log_rows': physical_rows, 'unique_docs': len(inputs),
              'filter_counts': {name: len(seen) for name, seen in filters.items()}}
    return inputs, {key: sorted(value) for key, value in sources.items()}, counts


def summarize(root):
    root = Path(root)
    identity, plan, pair = (c.read(root / name) for name in ('identity.json', 'plan.json', 'pair-audit.json'))
    m = identity['manifest']
    c.require(pair['status'] == 'paired-audits-passed' and pair['pair_id'] == m['pair_id'], 'complete paired audit required')
    c.require(tuple(c.SHAPES) == ((8192, 1), (2048, 512), (512, 1024), (512, 128)), 'unsupported benchmark shapes')
    result = {'status': 'paired-raw-results-reaudited', 'pair_id': m['pair_id'], 'commit': m['shared']['commit'],
              'accuracy': {}, 'benchmarks': {}, 'throughput_units': {'request_throughput': 'requests/s',
                  'output_throughput': 'output tokens/s', 'total_token_throughput': 'input + output tokens/s'},
              'benchmark_protocol': {'repeats': 3, 'measured_requests_per_run': 128, 'warmups_per_run': 16},
              'comparison': 'Percent changes compare the three-run medians; they are not confidence intervals.'}
    input_maps, audits = {}, {}
    for arm in ARMS:
        expected = c.make_workloads(m['client_python'], identity['model'], identity['endpoint'], root / arm)
        jobs = plan[arm]['workloads']
        fields = ('name', 'kind', 'expected')
        c.require([{k: j[k] for k in fields} for j in jobs] == [{k: j[k] for k in fields} for j in expected],
                  'workload names, shapes, seeds or three repeats differ: ' + arm)
        c.require(c.read(root / arm / 'status.json') == {'status': 'passed', 'jobs': 14}, 'arm incomplete: ' + arm)
        audits[arm] = []
        for job in jobs:
            directory = root / arm / job['name']
            receipt = c.read(directory / 'complete.json')
            c.require(re.fullmatch(r'attempt-[0-9]{3,}', receipt['attempt']) is not None, 'unsafe attempt path')
            attempt = directory / receipt['attempt']
            exit_status = c.read(attempt / 'exit.json')
            c.require(exit_status['returncode'] == 0 and not exit_status.get('reason'), 'selected command failed')
            context = c.read(attempt / 'server-context.json')
            saved_proof = Path(context['proof'])
            c.require(saved_proof.parent.parent.name == arm, 'server proof path belongs to wrong arm')
            proof_path = root / arm / saved_proof.parent.name / saved_proof.name
            proof = c.read(proof_path)
            c.require(c.sha256(proof_path) == context['proof_sha256'], 'server proof receipt changed')
            c.require(all(proof[k] == value for k, value in [('arm', arm), ('pair_id', m['pair_id']),
                          ('model', identity['model']), ('shared', m['shared']), ('source_verified', True)]), 'wrong server proof')
            audit = run_pair.audit_job(job, attempt)
            c.require(audit == receipt['audit'], 'raw audit differs from completion receipt: ' + job['name'])
            audits[arm].append({'name': job['name'], 'audit': audit})
            if job['kind'] == 'accuracy':
                task = job['expected']['task']
                raw = next((attempt / 'results').rglob('*results*.json'))
                samples = next((attempt / 'results').rglob('*samples*' + task + '*.jsonl'))
                obj = c.read(raw)
                metrics = obj['results'][task]
                accuracy_keys = [k for k in metrics if k.startswith('exact_match,')]
                c.require(accuracy_keys and all(k.replace('exact_match,', 'exact_match_stderr,', 1) in metrics
                          for k in accuracy_keys), 'official exact_match metric or standard error missing')
                inputs, sources, log_counts = sample_inputs(samples)
                c.require(log_counts['unique_docs'] == job['expected']['count'], 'accuracy independent document count differs')
                c.require(audit.get('logged_sample_rows') == log_counts['physical_log_rows'] and
                          audit.get('filters') == log_counts['filter_counts'], 'per-filter accuracy auditor required or counts differ')
                result['accuracy'].setdefault(task, {})[arm] = {
                    'metrics': {k: v for k, v in metrics.items() if ',' in k and '_stderr,' not in k},
                    'stderr': {k: v for k, v in metrics.items() if '_stderr,' in k},
                    'n_samples': obj['n-samples'][task], 'result_file': str(raw.relative_to(root)),
                    'samples_file': str(samples.relative_to(root)), 'sample_log': log_counts,
                    'protocol': c.ACCURACY_PROTOCOL[task]}
                input_maps[task, arm] = inputs, sources
            else:
                raw = attempt / 'results/benchmark.json'
                obj = c.read(raw)
                c.require(obj['num_prompts'] == 128, 'benchmark prompt count differs')
                values = {key: obj[key] for key in THROUGHPUT}
                c.require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values.values()),
                          'invalid throughput')
                shape = f"{job['expected']['input_len']}-{job['expected']['output_len']}"
                runs = result['benchmarks'].setdefault(shape, {}).setdefault(arm, {'runs': []})['runs']
                runs.append({'repeat': int(job['name'].split('-')[1][1:]), **values,
                             'duration_seconds': obj['duration'], 'result_file': str(raw.relative_to(root))})
        c.require(audits[arm] == c.read(root / arm / 'audit.json') == pair['audits'][arm], 'stale arm/pair audit')
    c.require(audits['baseline'] == audits['tuned'], 'baseline/tuned request/sample audit mismatch')
    for task, accuracy in result['accuracy'].items():
        baseline, baseline_sources = input_maps[task, 'baseline']
        tuned, tuned_sources = input_maps[task, 'tuned']
        c.require(baseline == tuned, 'accuracy actual input/reference mismatch: ' + task)
        accuracy['input_audit'] = {'status': 'matched-by-doc_id', 'samples': len(baseline),
            'sources': {key: sorted(set(baseline_sources[key] + tuned_sources[key])) for key in baseline_sources},
            'matched_inputs_sha256': stable_hash(sorted(baseline.items(), key=lambda item: str(item[0])))}
    for shape in result['benchmarks'].values():
        for arm in ARMS:
            c.require([row['repeat'] for row in shape[arm]['runs']] == [1, 2, 3], 'missing benchmark repeat')
            for key in THROUGHPUT:
                values = [row[key] for row in shape[arm]['runs']]
                shape[arm][key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
        ratios = {key: shape['tuned'][key]['median'] / shape['baseline'][key]['median'] for key in THROUGHPUT}
        shape['median_tuned_over_baseline_percent'] = {key: 100 * value for key, value in ratios.items()}
        shape['median_change_percent'] = {key: 100 * (value - 1) for key, value in ratios.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        text = json.dumps(summarize(args.pair_directory), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            temporary = args.output.with_name(args.output.name + '.tmp')
            temporary.write_text(text)
            temporary.replace(args.output)
        else:
            print(text, end='')
    except (ValueError, OSError, KeyError, TypeError, StopIteration, IndexError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
