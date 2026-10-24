"""Run the real integrated aggregate, capture its complete result."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1,campaign_te_s2 as s2
from runners.lumopt2_design import smoke_te_s2_restart as smoke

gate_dir=ROOT/'runners/lumopt2_design/gates'
aggregate=gate_dir/'run_all_gates.py'
tree=ast.parse(aggregate.read_text())
gates=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='GATES' for t in n.targets))
assert len(gates)==13 and len(set(gates))==13
assert eng.CampaignSpec().wgp_modal_guard is False
assert eng.CampaignSpec().best_metric=='fom'
assert all(s.SPEC_V3.wgp_modal_guard is False and s.SPEC_V3.best_metric=='fom' for s in (s1,s2))
assert s2.ADJ_FIX_PORT is None and smoke.CALIBRATION_SHA256 is None
command=[sys.executable,'-X','utf8',str(aggregate)]
suffix=sys.argv[1] if len(sys.argv)>1 else ''
log=HERE/('integrated_all13'+suffix+'.log')
record_path=HERE/('integration_gate_record'+suffix+'.json')
if log.exists() or record_path.exists():
    raise RuntimeError('Validation attempt exists; choose a unique suffix')
run=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                   encoding='utf8',timeout=300)
log.write_text(run.stdout,encoding='utf8')
record=dict(date_utc=datetime.now(timezone.utc).isoformat(),command=command,
    aggregate_exit_code=run.returncode,log=str(log),gate_count=len(gates),
    gates=[dict(gate=g,command=[sys.executable,str(gate_dir/g)],
                exit_code=0 if run.returncode==0 else None,
                sha256=hashlib.sha256((gate_dir/g).read_bytes()).hexdigest()) for g in gates],
    exit_code_basis='Aggregate fails immediately on any child nonzero; exit0 establishes each child exit0',
    defaults=dict(wgp_modal_guard=False,best_metric='fom'),
    s1_s2_flags_unchanged=True,calibration_unset=True,hardware_blocked=True,
    integrated_engine_sha256=hashlib.sha256(Path(eng.__file__).read_bytes()).hexdigest())
record_path.write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(dict(exit_code=run.returncode,gate_count=len(gates),log=str(log),
                     output_tail=run.stdout[-2200:]),indent=2))
sys.exit(run.returncode)
