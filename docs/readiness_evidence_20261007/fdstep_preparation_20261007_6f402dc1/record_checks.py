from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
module='runners.lumopt2_design.diagnose_te_s2_fdstep'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
suffix='_raw_reference'
commands=[('six_checks'+suffix,[sys.executable,'-X','utf8',str(ROOT/'runners/lumopt2_design/gates/gate_s2_fdstep_diagnostic.py')])]
for cluster in ('athena','igum'):
    commands.append((cluster+'_tasklist'+suffix,[sys.executable,'-X','utf8',
        str(ROOT/cluster/'scripts/build_sweep_list.py'),'--kind','lumopt2_design',
        '--module',module,'--output',str(HERE/(cluster+'_tasklist'+suffix+'.txt'))]))
results=[]
for label,command in commands:
    run=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       encoding='utf8',timeout=60)
    log=HERE/(label+'.log');log.write_text(run.stdout,encoding='utf8')
    results.append(dict(check=label,command=command,exit_code=run.returncode,log=str(log)))
    if run.returncode!=0:
        raise RuntimeError(run.stdout)
    if 'tasklist' in label:
        assert (HERE/(label+'.txt')).read_text().splitlines()==['0','1','2']
sys.path.insert(0,str(ROOT))
from runners.lumopt2_design import diagnose_te_s2_fdstep as diag
raw=ROOT/'docs/readiness_evidence_20261007/s2_reference_task00_remote_original.json'
watch=ROOT/'results_from_athena/recover_te_s2_20261006/watch/recover_te_s2/task_00.json'
assert sha(raw)==diag.REFERENCE_SHA256
assert json.loads(raw.read_text())==json.loads(watch.read_text())
record=dict(date_utc=datetime.now(timezone.utc).isoformat(),checks=results,test_count=6,
    source={name:sha(ROOT/name) for name in (
        'runners/lumopt2_design/diagnose_te_s2_fdstep.py',
        'runners/lumopt2_design/gates/gate_s2_fdstep_diagnostic.py',
        'runners/lumopt2_design/lumopt2_design.py',
        'runners/lumopt2_design/recover_te_s2.py')},
    helper=True,menu_activation=False,CAD=False,FDTD=False,server_actions=False,
    reference_checkpoint_sha256=diag.REFERENCE_SHA256,
    raw_reference=str(raw),watch_copy_sha256=sha(watch),semantic_JSON_equal=True,
    note='Default pins exact remote raw bytes; local test fixtures use explicit SHA overrides')
(HERE/'fdstep_preparation_checks_raw_reference.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record['source'],indent=2))
print('Six tests and both explicit task-list generators: exit0; tasks0/1/2; helper remains True')
