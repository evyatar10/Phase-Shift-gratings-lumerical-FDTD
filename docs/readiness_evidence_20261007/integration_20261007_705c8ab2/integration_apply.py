"""One-time local integration after exact baseline/dependency/archive checks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MANIFEST=ROOT/'docs/te_modal_guard_candidate_location.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
RECORD=HERE/'integration_apply_record.json'
if RECORD.exists():
    raise RuntimeError('Integration attempt already recorded; do not apply again')
m=json.loads(MANIFEST.read_text())
record=dict(date_utc=datetime.now(timezone.utc).isoformat(),manifest=str(MANIFEST),
    manifest_sha256=sha(MANIFEST),phase='checking',verified=[],commands=[])
def save():
    RECORD.write_text(json.dumps(record,indent=2),encoding='utf8')
def verify(path,expected,kind):
    actual=sha(path) if Path(path).exists() else None
    record['verified'].append(dict(path=str(path),expected=expected,actual=actual,kind=kind))
    if actual!=expected:
        record['phase']='HASH_MISMATCH';save()
        raise RuntimeError(f'{kind}: {path}: expected {expected}, actual {actual}; no integration')
for item in m['candidate_files']:
    path=ROOT/item['path']
    verify(path,item['baseline_sha256'],'baseline')
    verify(item['candidate_path'],item['candidate_sha256'],'candidate')
for item in m['shared_dependencies']:
    verify(ROOT/item['path'],item['sha256'],'shared_dependency')
for item in (m['patch'],m['archive']):
    verify(item['path'],item['sha256'],'patch_or_archive')
frozen=ROOT/'docs/readiness_evidence_20261007/frozen_recovery_sources_20261006.zip'
verify(frozen,'339c182a464eb8830232359527af4a06a3d6cc493c980b88e5fa9dc496992cc6','frozen_archive')
with zipfile.ZipFile(m['archive']['path']) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Candidate archive integrity failure')
    for item in m['candidate_files']:
        assert hashlib.sha256(archive.read('candidate/'+item['path'])).hexdigest()==item['candidate_sha256']
    for item in m['shared_dependencies']:
        assert hashlib.sha256(archive.read('shared_dependencies/'+item['path'])).hexdigest()==item['sha256']
with zipfile.ZipFile(HERE/'integration_before_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for item in m['candidate_files']+m['shared_dependencies']:
        path=ROOT/item['path']
        if path.exists():
            archive.write(path,item['path'])
record['phase']='verified';save()
for command in (['git','apply','--check',m['patch']['path']],
                ['git','apply',m['patch']['path']]):
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    record['commands'].append(dict(command=command,exit_code=run.returncode,
                                   stdout=run.stdout,stderr=run.stderr))
    save()
    if run.returncode!=0:
        raise RuntimeError(run.stdout+run.stderr)
record['phase']='applied_once'
record['integrated_files']=[]
for item in m['candidate_files']:
    path=ROOT/item['path'];candidate=Path(item['candidate_path'])
    if path.read_text(encoding='utf8')!=candidate.read_text(encoding='utf8'):
        raise RuntimeError('Integrated text differs from candidate: '+str(path))
    record['integrated_files'].append(dict(path=item['path'],sha256=sha(path),
        candidate_sha256=item['candidate_sha256'],normalized_text_equal=True))
for item in m['shared_dependencies']:
    verify(ROOT/item['path'],item['sha256'],'shared_dependency_after')
save()
print(json.dumps(dict(phase=record['phase'],files=len(record['integrated_files']),
    record=str(RECORD),engine_sha256=sha(ROOT/'runners/lumopt2_design/lumopt2_design.py')),indent=2))
