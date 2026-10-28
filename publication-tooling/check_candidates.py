#!/usr/bin/env python3
"""Update the candidate inventory and scan only new/changed file content.

Run: python publication/publication-tooling/check_candidates.py publication
The default stat cache stays outside publication; a clone without it scans all
content once. Findings contain filenames/counts, never matched secret values.
Bump POLICY_VERSION if content scan rules change. Adding a text format retains
previously accepted-format results; unsupported cached files are rescanned.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

POLICY_VERSION = 2
ALLOWED = {'.py', '.md', '.json', '.jsonl', '.txt', '.log', '.stdout', '.stderr', '.csv', '.sh', '.yaml', '.yml'}
PATTERNS = {
    'private_key_material': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'known_credential_prefix': r'\b(?:AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,})\b',
    'bearer_literal': r'(?i)\bBearer\s+[A-Za-z0-9._~+/-]{18,}={0,2}',
    'credential_in_url': r'https?://[^/\s:@\"\x27]+:[^/\s@\"\x27]+@',
    'credential_assignment': r'\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|API_KEY|PASSWORD)\s*[:=]\s*[\"\x27]?[A-Za-z0-9_./+\-]{8,}',
    'internal_dns': r'\b[A-Za-z0-9._-]+\.(?:bytedance\.net|bytedance\.com|byteintl\.net|volcengine\.com)\b',
    'private_ipv4': r'\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b',
    'local_path_metadata': r'/(?:Users|home)/[^/\s\"\x27]+|/data\d+/(?:tiger|docker)\b',
}
AUTH_KEY = re.compile(r'(^|_)(TOKEN|PASSWORD|PASSWD|SECRET|API_KEY|ACCESS_KEY|AUTHORIZATION|COOKIE|PRIVATE_KEY)(_|$)')
INFORMATIONAL = {'local_path_metadata', 'package_version_ip_shape', 'token_throughput_unit_label'}


def signature(path):
    s = path.lstat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def json_key_findings(value):
    if isinstance(value, dict):
        return sum(bool(AUTH_KEY.search(key.upper())) and isinstance(item, str) and bool(item.strip())
                   for key, item in value.items()) + sum(json_key_findings(item) for item in value.values())
    if isinstance(value, list):
        return sum(json_key_findings(item) for item in value)
    return 0


def scan(path):
    findings = Counter()
    if path.is_symlink():
        return {'findings': {'symlink': 1}, 'links': []}
    if path.suffix not in ALLOWED and path.name != 'LICENSE':
        return {'findings': {'unsupported_file_type': 1}, 'links': []}
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeError:
        return {'findings': {'non_utf8': 1}, 'links': []}
    for name, pattern in PATTERNS.items():
        findings[name] = len(re.findall(pattern, text))
    if path.suffix == '.json':
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        findings['credential_json_key'] = json_key_findings(obj)
        # Version 2: this exact public metric-unit label is not an auth token.
        if (path.name == 'pair2-summary.json' and isinstance(obj, dict)
                and obj.get('status') == 'paired-raw-results-reaudited'
                and obj.get('throughput_units', {}).get('total_token_throughput') == 'input + output tokens/s'):
            findings['credential_json_key'] -= 1
            findings['token_throughput_unit_label'] = 1
        # The prior official result contains a package version shaped like an IP.
        if isinstance(obj, dict) and isinstance(obj.get('pretty_env_info'), str):
            ignored = sum(len(re.findall(PATTERNS['private_ipv4'], line))
                          for line in obj['pretty_env_info'].splitlines() if '==' in line and '://' not in line)
            findings['private_ipv4'] = max(0, findings['private_ipv4'] - ignored)
            findings['package_version_ip_shape'] = ignored
    links = re.findall(r'\]\(([^)]+)\)', text) if path.suffix == '.md' else []
    return {'findings': {key: count for key, count in findings.items() if count}, 'links': links}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def candidates(root, excluded):
    return sorted((path for path in root.rglob('*')
                   if '.git' not in path.relative_to(root).parts and (path.is_file() or path.is_symlink())
                   and path.resolve() not in excluded), key=lambda path: str(path.relative_to(root)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--state', type=Path, help='Trusted local cache; defaults outside the candidate tree.')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('candidate root must be an existing directory')
    state_path = (args.state or root.parent / ('.' + root.name + '-candidate-scan-state.json')).resolve()
    inventory = root / 'CANDIDATE-FILES.json'
    report_path = root / 'INCREMENTAL-CHECKS.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Only the named summary's classification changed from policy 1 to 2.
    # Keep unaffected literal/key scan results and re-read every such summary.
    if old.get('policy_version') == 1 and old.get('root') == str(root):
        old['files'] = {name: entry for name, entry in old.get('files', {}).items()
                        if Path(name).name != 'pair2-summary.json'}
        old['policy_version'] = POLICY_VERSION
    if old.get('policy_version') != POLICY_VERSION or old.get('root') != str(root):
        old = {}
    previous, current = old.get('files', {}), {}
    scanned, reused = 0, 0
    findings, blocking, broken = [], [], []
    for path in candidates(root, {state_path, inventory.resolve(), report_path.resolve()}):
        name, before = str(path.relative_to(root)), signature(path)
        entry = previous.get(name, {})
        if entry.get('signature') == before and not entry.get('findings', {}).get('unsupported_file_type'):
            reused += 1
        else:
            entry = {'signature': before, **scan(path)}
            scanned += 1
        if signature(path) != before:
            blocking.append({'file': name, 'categories': {'changed_while_reading': 1}})
            continue
        current[name] = entry
        if entry['findings']:
            findings.append({'file': name, 'categories': entry['findings']})
            fatal = {key: count for key, count in entry['findings'].items() if key not in INFORMATIONAL}
            if fatal:
                blocking.append({'file': name, 'categories': fatal})
        missing = 0
        for href in entry['links']:
            if href.startswith('#') or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', href):
                continue
            target = href.split('#', 1)[0].strip('<>')
            missing += not (path.parent / target).exists()
        if missing:
            broken.append({'file': name, 'missing_targets': missing})
    now = datetime.now(timezone.utc).isoformat()
    report = {'snapshot_utc': now, 'policy_version': POLICY_VERSION,
              'status': 'passed' if not blocking and not broken else 'failed',
              'new_or_changed_files_scanned': scanned, 'unchanged_file_results_reused': reused,
              'removed_cached_files': len(set(previous) - set(current)),
              'blocking_findings': blocking, 'broken_local_markdown_links': broken, 'findings': findings,
              'scope': 'Literal/key checks on candidate text; stat reuse; no source hashes or matched values.',
              'generated_files_excluded_from_content_scan': [inventory.name, report_path.name],
              'inventory_excludes': ['.git/**', inventory.name, 'local scan state if placed inside root']}
    write_json(state_path, {'policy_version': POLICY_VERSION, 'root': str(root), 'snapshot_utc': now, 'files': current})
    write_json(report_path, report)
    rows = [{'file': str(path.relative_to(root)), 'bytes': path.lstat().st_size}
            for path in candidates(root, {state_path, inventory.resolve()})]
    write_json(inventory, {'snapshot_utc': now, 'excludes': ['.git/**', inventory.name, 'local scan state'],
                          'listed_files': len(rows), 'listed_bytes': sum(row['bytes'] for row in rows), 'files': rows})
    print(json.dumps({'status': report['status'], 'new_or_changed_files_scanned': scanned,
                      'unchanged_file_results_reused': reused, 'blocking_files': len(blocking),
                      'broken_markdown_files': len(broken), 'candidate_files': len(rows) + 1,
                      'candidate_bytes': sum(row['bytes'] for row in rows) + inventory.stat().st_size}))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('ERROR: publication check could not finish (' + type(error).__name__ + ')', file=sys.stderr)
        sys.exit(1)
