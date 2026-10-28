#!/usr/bin/env python3
"""Export this experiment's original text results only after the pool reports passed.

Without --collect, print the fixed scope and make no remote calls or archives.
The archive stays on the task's /data03 disk and is also copied into local evidence.
Existing hardware/maps evidence and every original result remain unchanged.
"""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import uuid

JOB = '48781766ecb1491487388e1ad00fe7c5'
PAIR_ID = 'glm53-h20-pr58986-eb34e115-native-full-mtp3-20261028-pair2'
BASE = '/data03/tiger/operator_sweep_pool_tasks/moe58986/20261024-153729-7168ad'
OUTPUT = BASE + '/pr58986-native-fullpair-eb34e115-mtp3-pair2'
LAUNCHER = BASE + '/unified-environment/sweep-unified-moe58986-bbbd3419b670.json'
EXPORT = '/data03/tiger/vllm_pr58986_evalprep_20261028_eb34e115/exports'
WORKSPACE = Path(__file__).resolve().parents[1]
EVIDENCE = WORKSPACE / 'evidence/fullpair-v10-pair2-48781766'
PYTHON = '/Users/bytedance/Documents/ChatGPT/b200优化/.venv/bin/python'
POOL = '/Users/bytedance/.codex/.chatgpt-projects/g-p-6ac8d3f87e388191bd7a2fba93516f11/runtime/common_env/pool.py'
STATUS_ARGV = [PYTHON, POOL, 'status', JOB]
EXCLUDED_DIRS = ['hf-cache', 'hf-cache.partial', 'cache', 'home', 'tmp', 'locks', 'weights', 'binaries', '__pycache__']
EXCLUDED_SUFFIXES = ['.so', '.a', '.o', '.cubin', '.ptx', '.bin', '.safetensors', '.arrow', '.parquet', '.pt', '.pth', '.npy', '.npz', '.pkl', '.pyc']

REMOTE = r'''
import datetime, hashlib, io, json, os, stat, tarfile
from pathlib import Path
root, launcher = Path(cfg['output']), Path(cfg['launcher'])
def read(p): return json.loads(p.read_text())
def require(ok, message):
    if not ok: raise RuntimeError(message)
require(root.is_dir() and not root.is_symlink(), 'output root missing or symlink')
pair = read(root/'pair/pair-audit.json')
require(pair['status']=='paired-audits-passed' and pair['pair_id']==cfg['pair_id'], 'pair audit not passed for expected pair')
checks = {'pair_audit': {'status':pair['status'], 'pair_id':pair['pair_id']}, 'arms':{}}
for arm in ('baseline','tuned'):
    armroot=root/'pair'/arm
    armstatus=read(armroot/'status.json')
    require(armstatus=={'status':'passed','jobs':14}, arm+' does not have 14 passed jobs')
    complete=list(armroot.glob('*/complete.json'))
    require(len(complete)==14, arm+' complete receipt count differs')
    server=list(armroot.glob('server-*/exit.json'))
    require(bool(server), arm+' server exit receipt missing')
    exits={str(p.relative_to(root)):read(p) for p in server}
    require(all(p['returncode']==0 for p in exits.values()), arm+' server did not exit cleanly')
    checks['arms'][arm]={'status':armstatus,'complete_receipts':len(complete),'server_exits':exits}
runner=list(root.glob('runner-*/exit.json'))
require(bool(runner) and all(read(p)['returncode']==0 for p in runner), 'outer runner exit not zero')
checks['runner_exits']={str(p.relative_to(root)):read(p) for p in runner}
launch=read(launcher)
require(launch['status']=='passed' and launch['exit_code']==0, 'actual launcher not passed')
state=launch['owned_exit']['state']
require(state['Running'] is False and state['Pid']==0 and state['OOMKilled'] is False and state['ExitCode']==0, 'container exit is not clean')
checks['launcher']={'path':str(launcher),'status':launch['status'],'exit_code':launch['exit_code'],'owned_exit':launch['owned_exit']}
archive_dir=Path(cfg['export'])
archive_dir.mkdir(exist_ok=True)
require(archive_dir.stat().st_dev==root.stat().st_dev, 'export is not on the same data filesystem as output')
archive=archive_dir/(cfg['capture']+'.tar.gz')
partial=archive.with_suffix(archive.suffix+'.partial')
require(not archive.exists() and not partial.exists(), 'export already exists')
exclude_dirs=set(cfg['excluded_dirs']); suffixes=tuple(cfg['excluded_suffixes'])
files=[]; skipped=[]
def include(path, arcname):
    st=path.lstat()
    if stat.S_ISLNK(st.st_mode): skipped.append({'path':str(path),'reason':'symlink'}); return
    if not stat.S_ISREG(st.st_mode): skipped.append({'path':str(path),'reason':'not-regular'}); return
    if path.name.endswith('.lock') or path.name.endswith(suffixes) or '.so.' in path.name:
        skipped.append({'path':str(path),'reason':'lock-or-binary'}); return
    files.append((path,arcname,st.st_size))
for child in sorted(root.iterdir()):
    if child.is_symlink(): skipped.append({'path':str(child),'reason':'symlink'}); continue
    if child.is_file(): include(child,'job/'+child.name); continue
    if child.name!='pair' and not child.name.startswith(('setup','runner')):
        skipped.append({'path':str(child),'reason':'outside-result-scope'}); continue
    for directory, dirs, names in os.walk(child,followlinks=False):
        keep=[]
        for name in sorted(dirs):
            p=Path(directory)/name
            if name in exclude_dirs or p.is_symlink(): skipped.append({'path':str(p),'reason':'cache-or-symlink-directory'})
            else: keep.append(name)
        dirs[:]=keep
        for name in sorted(names):
            p=Path(directory)/name
            include(p,'job/'+p.relative_to(root).as_posix())
for p in sorted(launcher.parent.glob(launcher.stem+'.*')):
    include(p,'launcher/'+p.name)
for pattern in ('hardware-before-bench-48781766.*','native-maps-baseline-48781766.*','native-maps-tuned-48781766.*'):
    for p in sorted(Path(cfg['base']).glob(pattern)):
        include(p,'live-process-evidence/'+p.name)
require(any(arc.startswith('launcher/') and arc.endswith('.log') for _,arc,_ in files), 'actual launcher log absent')
require(len({arc for _,arc,_ in files})==len(files), 'duplicate archive paths')
manifest={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':cfg['job_id'],'original_output':str(root),'original_launcher':str(launcher),'scope':'all root metadata, pair/**, setup*/**, runner*/**, actual launcher artifacts and existing hardware/maps; no metric redaction','excluded_dirs':sorted(exclude_dirs),'excluded_suffixes':list(suffixes),'skipped':skipped,'terminal_checks':checks,'files':[{'source':str(p),'archive_path':arc,'bytes':size} for p,arc,size in files]}
with tarfile.open(partial,'x:gz',compresslevel=6,dereference=False) as tf:
    for p,arc,size in files:
        require(p.lstat().st_size==size and not p.is_symlink(), 'result changed during final export: '+str(p))
        tf.add(p,arcname=arc,recursive=False)
    for name,obj in [('collection-manifest.json',manifest),('pool-job-status.json',cfg['pool_status']),('pool-status-command.json',cfg['pool_command'])]:
        content=(json.dumps(obj,indent=2,ensure_ascii=False)+'\n').encode()
        info=tarfile.TarInfo(name); info.size=len(content); info.mode=0o644
        tf.addfile(info,io.BytesIO(content))
    for name,content in [('pool-status.stdout.txt',cfg['pool_stdout']),('pool-status.stderr.txt',cfg['pool_stderr'])]:
        content=content.encode(); info=tarfile.TarInfo(name); info.size=len(content); info.mode=0o644
        tf.addfile(info,io.BytesIO(content))
os.rename(partial,archive)
h=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
receipt={'archive':str(archive),'sha256':h.hexdigest(),'bytes':archive.stat().st_size,'files':len(files),'original_bytes':sum(size for _,_,size in files),'terminal_checks':checks}
with archive.with_suffix(archive.suffix+'.receipt.json').open('x') as f: json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''

def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collect',action='store_true',help='require passed terminal receipts, archive and download originals')
    args=ap.parse_args()
    cfg={'job_id':JOB,'pair_id':PAIR_ID,'base':BASE,'output':OUTPUT,'launcher':LAUNCHER,'export':EXPORT,'excluded_dirs':EXCLUDED_DIRS,'excluded_suffixes':EXCLUDED_SUFFIXES}
    if not args.collect:
        print(json.dumps({'action':'print-plan-only','status_argv':STATUS_ARGV,'destination':str(EVIDENCE),'scope':cfg},indent=2)); return
    capture='pair2-final-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    dest=EVIDENCE/capture
    dest.mkdir(parents=True,exist_ok=False)
    status=subprocess.run(STATUS_ARGV,capture_output=True,text=True,timeout=45)
    (dest/'pool-status.stdout.txt').write_text(status.stdout)
    (dest/'pool-status.stderr.txt').write_text(status.stderr)
    command={'argv':STATUS_ARGV,'returncode':status.returncode}
    (dest/'pool-status-command.json').write_text(json.dumps(command,indent=2)+'\n')
    status.check_returncode()
    rows=json.loads(status.stdout)
    if not isinstance(rows,list) or len(rows)!=1:
        raise RuntimeError('official pool status did not return exactly one job')
    d=rows[0]
    result=d.get('allocation',{}).get('result',{})
    if not (d['id']==JOB and d['status']=='passed' and result.get('status')=='finished' and result.get('exit_code')==0 and result.get('foreign_gpu_contamination') is False and result.get('resource_busy_before_launch') is False and result.get('launch_receipts')==[LAUNCHER]):
        raise RuntimeError('expected pair2 has not passed cleanly; raw status preserved, no remote archive created')
    cfg.update(capture=capture,pool_status=d,pool_command=command,pool_stdout=status.stdout,pool_stderr=status.stderr)
    script='cfg='+repr(cfg)+'\n'+REMOTE
    (dest/'remote-collection.py').write_text(script)
    ssh_argv=['ssh','h20-01','python3 -']
    (dest/'collection-command.json').write_text(json.dumps({'argv':ssh_argv,'stdin':'remote-collection.py'},indent=2)+'\n')
    remote=subprocess.run(ssh_argv,input=script,capture_output=True,text=True,timeout=600)
    (dest/'collection.stdout.txt').write_text(remote.stdout)
    (dest/'collection.stderr.txt').write_text(remote.stderr)
    (dest/'collection-exit.json').write_text(json.dumps({'returncode':remote.returncode})+'\n')
    remote.check_returncode()
    receipt=json.loads(remote.stdout)
    (dest/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    archive=dest/Path(receipt['archive']).name
    scp_argv=['scp','h20-01:'+shlex.quote(receipt['archive']),str(archive)]
    copy=subprocess.run(scp_argv,capture_output=True,text=True,timeout=600)
    (dest/'copy-command.json').write_text(json.dumps({'argv':scp_argv,'returncode':copy.returncode},indent=2)+'\n')
    (dest/'copy.stdout.txt').write_text(copy.stdout); (dest/'copy.stderr.txt').write_text(copy.stderr)
    copy.check_returncode()
    import hashlib
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    if h.hexdigest()!=receipt['sha256']: raise RuntimeError('downloaded archive SHA differs')
    raw=dest/'raw'; raw.mkdir()
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            target=raw/member.name
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()) or not target.resolve().is_relative_to(raw.resolve()):
                raise RuntimeError('unsafe archive member: '+member.name)
        tf.extractall(raw,filter='data')
    (dest/'LOCAL-COMPLETE.json').write_text(json.dumps({'job_id':JOB,'archive_sha256':receipt['sha256'],'raw':str(raw)},indent=2)+'\n')
    print(json.dumps({'evidence':str(dest),'raw':str(raw),'archive':str(archive),'remote_archive':receipt['archive'],'bytes':receipt['bytes'],'sha256':receipt['sha256']},indent=2))

if __name__=='__main__':
    run()
