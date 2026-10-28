#!/usr/bin/env python3
"""Concatenate selected official stdout bytes with relative source-path labels."""
import argparse
import json
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair_directory', type=Path)
    parser.add_argument('--kind', choices=['benchmark', 'accuracy'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pair = args.pair_directory
    plan = json.loads((pair / 'plan.json').read_text())
    selected = []
    for arm in ('baseline', 'tuned'):
        jobs = [job for job in plan[arm]['workloads'] if job['kind'] == args.kind]
        if len(jobs) != (12 if args.kind == 'benchmark' else 2):
            raise ValueError('unexpected selected workload count')
        for job in jobs:
            directory = pair / arm / job['name']
            complete = json.loads((directory / 'complete.json').read_text())
            if re.fullmatch(r'attempt-[0-9]{3,}', complete['attempt']) is None:
                raise ValueError('unsafe attempt path')
            attempt = directory / complete['attempt']
            status = json.loads((attempt / 'exit.json').read_text())
            if status['returncode'] != 0 or status.get('reason'):
                raise ValueError('selected official command failed')
            path = attempt / 'stdout.txt'
            selected.append((str(path.relative_to(pair)), path.read_bytes()))
    temporary = args.output.with_name(args.output.name + '.tmp')
    index = []
    with temporary.open('wb') as out:
        for name, data in selected:
            out.write(('\n===== SOURCE: ' + name + ' =====\n').encode())
            offset = out.tell()
            out.write(data)
            index.append({'source_relative_to_pair': name, 'payload_offset': offset, 'payload_bytes': len(data)})
            out.write(('\n===== END SOURCE: ' + name + ' =====\n').encode())
    temporary.replace(args.output)
    args.output.with_suffix('.sources.json').write_text(json.dumps({'kind': args.kind, 'files': index}, indent=2) + '\n')
    print(json.dumps({'selected_files': len(index), 'exact_source_payload_bytes': sum(row['payload_bytes'] for row in index)}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
