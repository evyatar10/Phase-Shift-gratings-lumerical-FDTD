"""Record and archive the validated integrated source identity."""
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((ROOT/'docs/te_modal_guard_candidate_location.json').read_text())
g=json.loads((HERE/'integration_gate_record_attempt2.json').read_text())
assert g['aggregate_exit_code']==0 and g['gate_count']==13
record_path=HERE/'integration_final_identity.json'
archive_path=HERE/'integrated_validated_sources.zip'
if record_path.exists() or archive_path.exists():
    raise RuntimeError('Final integration artifacts already exist')
files=[]
for item in m['candidate_files']:
    path=ROOT/item['path'];candidate=Path(item['candidate_path'])
    same=path.read_text(encoding='utf8')==candidate.read_text(encoding='utf8')
    if item['path'].endswith('gate_modal_guard.py'):
        assert not same
        delta=''.join(difflib.unified_diff(candidate.read_text(encoding='utf8').splitlines(True),
            path.read_text(encoding='utf8').splitlines(True),
            fromfile='a/'+item['path'],tofile='b/'+item['path']))
        (HERE/'gate_modal_guard_standalone_bootstrap_fix.patch').write_text(delta,encoding='utf8',newline='\n')
    else:
        assert same,item['path']
    files.append(dict(path=item['path'],sha256=sha(path),
        candidate_sha256=item['candidate_sha256'],normalized_candidate_equal=same))
for item in m['shared_dependencies']:
    assert sha(ROOT/item['path'])==item['sha256'],item['path']
with zipfile.ZipFile(m['archive']['path']) as candidate:
    for name in ('campaign_te_s1.py','campaign_te_s2.py'):
        actual=ROOT/'runners/lumopt2_design'/name
        assert actual.read_bytes()==candidate.read('additional_context/'+name),name
record=dict(date_utc=datetime.now(timezone.utc).isoformat(),integration='LOCAL_ONLY',
    patch_applied_once=m['patch'],files=files,shared_dependencies=m['shared_dependencies'],
    test_record=str(HERE/'integration_gate_record_attempt2.json'),
    standalone_gate_repair='Added import sys and ROOT sys.path insertion; no assertions or physics changed',
    failed_first_attempt_preserved=str(HERE/'integrated_all13.log'),
    defaults=dict(wgp_modal_guard=False,best_metric='fom'),
    S1_S2_configuration_bytes_unchanged=True,calibration_installed=False,
    calibration_failure='170754_2 scientific gate failed; hardware remains blocked',
    CAD=False,FDTD=False,server_actions=False,Git_index_or_commit_mutations=False,
    pending='Parent review/commit; scientific qualification before hardware/upload',
    before_source_archive_sha256=sha(HERE/'integration_before_sources.zip'))
record_path.write_text(json.dumps(record,indent=2),encoding='utf8')
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for item in m['candidate_files']+m['shared_dependencies']:
        z.write(ROOT/item['path'],item['path'])
    for path in HERE.glob('*'):
        if path.is_file() and path.suffix not in ('.zip',):
            z.write(path,'evidence/'+path.name)
with zipfile.ZipFile(archive_path) as z:
    assert z.testzip() is None
print(json.dumps(dict(identity=str(record_path),archive=str(archive_path),
    archive_sha256=sha(archive_path),engine_sha256=files[0]['sha256'],
    gate_modal_guard_sha256=files[-1]['sha256'],all13_exit_code=g['aggregate_exit_code']),indent=2))
