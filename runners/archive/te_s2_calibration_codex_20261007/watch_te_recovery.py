"""Local, read-only watcher for the authorized October 6 TE recovery chain.

One Athena SSH connection per 15 minutes, no submission/cancellation/cleanup.
Collects new scalar checkpoints and records decision-relevant state changes.
Stops on SSH failure, a failed job, completion, or after 72 hours. The laptop
must remain awake; SLURM dependencies continue independently of this watcher.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

JOBS = '170746,170750,170752,170754'
EXPECTED_TASKS = ({f'170746_{i}' for i in (1, 2, 13, 14)}
                  | {f'170750_{i}' for i in range(3, 13)}
                  | {'170752_0', '170752_1', '170754_2'})
REMOTE = r'''
import json,subprocess
from pathlib import Path
jobs='170746,170750,170752,170754'
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(code=r.returncode,out=r.stdout,err=r.stderr)
base=Path('/home/evyatarrubin/bragg_sim_athena')
out=dict(accounting=command(['sacct','-X','-n','-P','-j',jobs,'--format=JobID,State,ExitCode']),
         queue=command(['squeue','-r','-h','-j',jobs,'--format=%i|%T|%R']),
         quota=command(['quota','-s']),files={},logs={})
for study in ('recover_te_s2','recover_te_s2_peak'):
 for p in (base/'results'/study/'results').glob('lumopt2_te_s2_*/task_[0-9][0-9].json'):
  out['files'][study+'/'+p.name]=json.loads(p.read_text())
for job in jobs.split(','):
 for p in (base/'jobs/logs').glob('lum_array-'+job+'_*.out'):
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-12000))
   out['logs'][p.name]=f.read().decode('utf-8','replace')
print(json.dumps(out))
'''


def collect(ssh_hostname=None):
    script = "python3 - <<'PY'\n"+REMOTE+"\nPY\n"
    # An explicitly verified address can bypass local DNS without weakening
    # host-key checking or changing the user's SSH configuration.
    options = (['-o', f'HostName={ssh_hostname}', '-o',
                'HostKeyAlias=athena.technion.ac.il'] if ssh_hostname else [])
    result = subprocess.run(['ssh', *options, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=yes', 'evyatarrubin@athena.technion.ac.il', 'bash -s'],
        input=script.encode('utf-8'), capture_output=True, timeout=100)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace'))
    return json.loads(result.stdout)


def verdict(data):
    for name in ('accounting', 'queue'):
        if data[name]['code'] != 0:
            raise RuntimeError(f"{name} query failed: {data[name]['err']}")
    rows = [r.split('|') for r in data['accounting']['out'].splitlines() if r.strip()]
    states = [r[1].split()[0].rstrip('+') for r in rows]
    failed = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL'}
    if any(s in failed for s in states):
        return 'failed'
    # Missing accounting rows (or a failed squeue) must never look complete.
    if (EXPECTED_TASKS <= {r[0] for r in rows}
            and not data['queue']['out'].strip()
            and all(s == 'COMPLETED' for s in states)):
        return 'complete'
    return 'active'


def persist(root, data):
    for name, content in data['files'].items():
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('x', encoding='utf-8') as f:
                json.dump(content, f, indent=2)
        elif json.loads(target.read_text(encoding='utf-8')) != content:
            raise RuntimeError(f'Remote immutable checkpoint changed: {name}')
    state = dict(accounting=data['accounting'], queue=data['queue'],
                 checkpoints=sorted(data['files']), verdict=verdict(data))
    key = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # One full snapshot per change, never a model wake on elapsed-time changes.
    marker = root/('state_'+key+'.json')
    if not marker.exists():
        with marker.open('x', encoding='utf-8') as f:
            json.dump(dict(utc=stamp, **data), f, indent=2)
        with (root/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=stamp, **state))+'\n')
        print(stamp, state['verdict'], len(data['files']), 'checkpoints', flush=True)
    return state['verdict']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--ssh-hostname', help='Previously verified Athena address; retains hostname host-key verification')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+72*3600
    while True:
        try:
            state = persist(args.output, collect(args.ssh_hostname))
        except Exception as e:
            with (args.output/'errors.log').open('a', encoding='utf-8') as f:
                f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {e!r}\n')
            raise
        if args.once or state != 'active' or time.monotonic() >= deadline:
            return
        time.sleep(900)


if __name__ == '__main__':
    main()
