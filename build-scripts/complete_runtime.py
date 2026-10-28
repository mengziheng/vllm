"""Assemble once, probe in a fresh child process, then freeze for GPU admission."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/work')
for script in ('assemble_runtime.py', 'probe_assembled_runtime.py'):
    subprocess.check_call([sys.executable, str(root / 'scripts' / script)])
native_receipt = root / 'receipts/native-receipt.json'
native = json.loads(native_receipt.read_text())
probe_path = root / 'receipts/assembled-runtime-probe.json'
probe = json.loads(probe_path.read_text())
assert probe['status'] == 'passed' and not probe['cuda_initialized']
target = 'vllm/model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json'
source_checks = {}
for arm in ('baseline', 'tuned'):
    repository = root / 'assembled' / arm
    git = ['git', '-c', 'safe.directory=' + str(repository)]
    exclusions = [':(exclude)' + name for name in [target, *native['source_files']]]
    diff = subprocess.check_output([*git, 'diff', '--binary', 'HEAD', '--', '.', *exclusions], cwd=repository)
    assert not diff, arm
    extra = subprocess.check_output([*git, 'ls-files', '--others', '--exclude-standard'], cwd=repository, text=True).splitlines()
    assert all(name in native['source_files'] for name in extra), (arm, extra)
    target_diff = subprocess.check_output([*git, 'diff', '--name-status', 'HEAD', '--', target], cwd=repository, text=True)
    if arm == 'baseline':
        assert target_diff == 'D\t' + target + '\n'
    else:
        assert not target_diff
    source_checks[arm] = {'commit': subprocess.check_output([*git, 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
                          'common_diff_sha256': hashlib.sha256(diff).hexdigest(), 'target_diff': target_diff}
freeze = {
    'status': 'ready_for_gpu_smoke',
    'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'native_receipt_sha256': hashlib.sha256(native_receipt.read_bytes()).hexdigest(),
    'probe_receipt_sha256': hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    'source_checks': source_checks,
    'host_root': '/data03/tiger/vllm_pr58986_native_20261028_eb34e115',
    'readonly_bind': '/data03/tiger/vllm_pr58986_native_20261028_eb34e115:/native-runtime',
    'native_receipt': '/native-runtime/receipts/native-receipt.json',
    'gpu_execution_verified': False,
}
(root / 'receipts/FROZEN.json').write_text(json.dumps(freeze, indent=2))
print(json.dumps(freeze, indent=2))
