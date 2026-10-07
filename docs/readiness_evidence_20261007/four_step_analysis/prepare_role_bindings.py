"""Bind measured historical checkpoint roles; no repository writes."""
import hashlib
import json
from pathlib import Path
import compare_four_steps as calc

HERE=Path(__file__).resolve().parent
ROOT=Path('C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes')
source=ROOT/'docs/readiness_evidence_20261007/four_step_analysis/existing_evidence_pins.json'
pins=json.loads(source.read_text())
center,plus,minus=pins['old_checkpoint_rows']
original=ROOT/'docs/readiness_evidence_20261007/s2_reference_task00_remote_original.json'
raw=json.loads(original.read_text())
reduced=dict(identity=raw['identity'],result=raw['result']['s2_gate'])
if reduced!=center:
    raise RuntimeError('Preserved exporter checkpoint does not match original pins')
bindings=dict(center_export=dict(role='stored_forward_export_no_solve',
    code=center['identity']['code'],semantic_sha256=calc.semantic_sha(center),
    original_checkpoint=str(original),original_checkpoint_sha256=calc.sha(original.read_bytes()),
    recorded_job='170745_0',qualification='Fingerprint of export code, not original FSP creation'),
    fd_endpoints={})
for row in (plus,minus):
    task=row['identity']['task']
    original=ROOT/f'results_from_athena/recover_te_s2_20261006/watch/recover_te_s2/task_{task:02d}.json'
    if json.loads(original.read_text())!=row:
        raise RuntimeError('Preserved FD checkpoint differs from pinned record')
    bindings['fd_endpoints'][str(task)]=dict(role='new_forward_fd_leg',
        code=row['identity']['code'],semantic_sha256=calc.semantic_sha(row),
        observed_checkpoint=str(original),observed_checkpoint_sha256=calc.sha(original.read_bytes()),
        recorded_job=f'170750_{task}',qualification='Later recovery-generated FD solve; watcher serialization bytes separately recorded')
pins['old_source_bindings']=bindings
target=HERE/'existing_evidence_pins_role_bound.json'
with target.open('x',encoding='utf8') as stream:
    json.dump(pins,stream,indent=2)
print('Role-bound pins SHA256',calc.sha(target.read_bytes()))
