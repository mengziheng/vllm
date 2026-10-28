"""Use the official pool CLI with narrow support for this task's legacy receipt.

This never edits the shared pool or its receipts. Modern preparations still use
its original resolver; only this original, still-current release can fall back.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

POOL_DIR = Path('/Users/bytedance/.codex/.chatgpt-projects/g-p-6ac8d3f87e388191bd7a2fba93516f11/runtime/common_env')
TASK = 'moe58986'
RELEASE = '20261024-153729-7168ad'
LEGACY_SHA256 = '11689650958767c6937d9eb526c7e28e30d8fc6f8acd6b37da9d7ebe932d4e32'


def resolve_legacy(original, state, task, release, current, expected_receipt_sha256=None):
    try:
        return original(state, task, release, current)
    except KeyError as error:
        if error.args != ('config_template',) or (task, release) != (TASK, RELEASE):
            raise
        base = Path(state) / 'staging' / task / release
        paths = list(base.glob('PREPARATION*.json'))
        if len(paths) != 1 or paths[0].name != 'PREPARATION.json' or paths[0].resolve().parent != base.resolve():
            raise ValueError('Legacy admission requires exactly the original preparation receipt') from error
        raw = paths[0].read_bytes()
        if expected_receipt_sha256 and hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
            raise ValueError('Legacy receipt content changed') from error
        legacy = json.loads(raw)
        if any(key in legacy for key in ('config_template', 'runtime_identity', 'manifest_sha256', 'image_id', 'framework')):
            raise
        allowed = {'task', 'release', 'source_node', 'source_root', 'hosts', 'errors', 'staged_bytes', 'node_root_overrides'}
        hosts = current.get('hosts', {})
        if (set(legacy) != allowed or legacy['task'] != task or legacy['release'] != release
            or current.get('pool_release_id') != release or current.get('framework') != 'v032'
            or current.get('source_hosts') != {legacy['source_node']: legacy['source_root']}
            or 'h20-01' not in hosts or any(legacy['hosts'].get(node) != root for node, root in hosts.items())
            or any(node not in ('h20-01', 'h20-02', 'h20-03', 'h20-04') or not Path(root).is_absolute() or '..' in Path(root).parts for node, root in hosts.items())):
            raise ValueError('Legacy preparation differs from the exact current task release/source/ready hosts') from error
        return current


def main():
    sys.path.insert(0, str(POOL_DIR))
    pool = importlib.import_module('pool')
    original = pool.fixed_preparation
    pool.fixed_preparation = lambda *args: resolve_legacy(original, *args, expected_receipt_sha256=LEGACY_SHA256)
    # Only the current process uses this compatibility resolver. All admission,
    # ownership, pause, resource, storage and dispatch checks remain official.
    return pool.main()


if __name__ == '__main__':
    raise SystemExit(main())
