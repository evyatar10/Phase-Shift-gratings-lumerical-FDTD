"""Prepared staged half-step class FD diagnostic; twelve forward-only tasks.

First wave0..3 repeats corr30 +/-0.5nm and measures shift30 +/-0.5nm.
Later4..11 measures corr1, avg1, shift1, wcav +/-0.5nm ONLY after parent
review of all four immutable first-wave checkpoints. No nominal, adjoint,
optimizer, fitting or automatic calibration verdict. No marker is published
by this runner. Existing corr30 half-step evidence is retained separately.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(os.environ.get('S2_HALF_REPO', Path.cwd())).resolve()))
import numpy as np
import config
from runners.lumopt2_design import diagnose_te_s2_fdstep as diag

eng, val, recovery = diag.eng, diag.val, diag.recovery
IS_HELPER = True
N_TASKS = 12
LABEL = 'lumopt2_te_s2_fdhalf_classcheck_20261007'
SPEC = dataclasses.replace(diag.SPEC, label=LABEL)
INDICES,LABELS=val._indices(SPEC)
if INDICES != [0,29,60,120,149,eng.layout(SPEC.n_free).I_CAV] or eng.layout(SPEC.n_free).I_CAV != 295:
    raise RuntimeError('Unexpected live TE60 index map')
BY_LABEL=dict(zip(LABELS,INDICES))
TASK_CLASSES=('corr_30','shift_30','corr_1','avg_1','shift_1','wcav')
TASK_MAP=tuple((label,BY_LABEL[label],.5,sign) for label in TASK_CLASSES for sign in (1,-1))
CENTER_SHA256 = 'cd2dd4a58be7b83c46ef6a6ac90a874e5fc33edda884edfc8ac50ce5e3be4a06'
APPROVAL_STATUS = {'first':'APPROVED_FOR_FIRSTWAVE_HALFSTEP_FD',
                  'later':'APPROVED_FOR_REMAINING_CLASS_FD_LEGS_AFTER_FIRSTWAVE_REVIEW'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def env_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(name + ' must be explicitly configured')
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(name + ' must be an absolute path')
    return path.resolve()


def task_params(task):
    if isinstance(task, bool) or not isinstance(task, (int, np.integer)) or task not in range(N_TASKS):
        raise ValueError('Task must be integer 0..11; got ' + repr(task))
    label,index,h,sign = TASK_MAP[task]
    p = val.te_point(SPEC).copy()
    p[index] += sign*h
    bounds = np.asarray(eng.param_bounds(SPEC), dtype=float)
    if not np.all((p >= bounds[:, 0]) & (p <= bounds[:, 1])):
        raise RuntimeError('Convergence leg is outside its own bounds')
    return p,index,h,sign,label


def dependency_hashes():
    # Existing diagnostic's exact frozen dependency list includes itself,
    # engine, recovery, validator, S2, builder, simulation_config, helpers, v3.
    code = dict(diag.identity(0)['code'])
    code[Path(__file__).name] = digest(__file__)
    code[Path(config.__file__).name] = digest(config.__file__)
    return code


def output_root():
    root = env_path('S2_HALF_OUTPUT_ROOT')
    if root.name != LABEL:
        raise RuntimeError('Output root must end with unique study label ' + LABEL)
    center = env_path('S2_HALF_CENTER_CHECKPOINT')
    if root == center.parent or root.is_relative_to(center.parent):
        raise RuntimeError('Output must be separate from center evidence')
    return root


def load_center():
    if dataclasses.asdict(SPEC) != dataclasses.asdict(dataclasses.replace(diag.SPEC, label=LABEL)):
        raise RuntimeError('New runner numerical family differs from frozen original diagnostic')
    path = env_path('S2_HALF_CENTER_CHECKPOINT')
    if digest(path) != CENTER_SHA256:
        raise RuntimeError('Pinned NEW center checkpoint hash mismatch')
    marker = path.parent/'center_reviewed.sha256'
    if not marker.is_file() or marker.read_text().strip() != CENTER_SHA256:
        raise RuntimeError('Exact original reviewed-center marker is required')
    row = json.loads(path.read_text())
    identity, result = row['identity'], row['result']
    point = val.te_point(SPEC)
    if identity.get('task') != 0 or identity.get('sign') != 0 or identity.get('index') != BY_LABEL['corr_30'] or identity.get('perturbation_nm') != .5:
        raise RuntimeError('Center checkpoint is not original diagnostic task00')
    if not np.array_equal(np.asarray(identity.get('point'), float), point):
        raise RuntimeError('Center parameter point mismatch')
    if identity.get('spec') != json.loads(json.dumps(dataclasses.asdict(diag.SPEC))):
        raise RuntimeError('Center complete geometry/numerical-family spec mismatch')
    if identity.get('code') != diag.identity(0)['code']:
        raise RuntimeError('Frozen center source hashes differ from current dependencies')
    if result.get('task') != 0 or result.get('sign') != 0 or not np.array_equal(np.asarray(result.get('params'), float), point):
        raise RuntimeError('Center result point/task mismatch')
    recovery_metrics = {key: float(result[key]) for key in diag.METRICS}
    if not all(np.isfinite(v) for v in recovery_metrics.values()):
        raise RuntimeError('Center has nonfinite metrics')
    diag.verify_artifacts(result)
    return dict(checkpoint=str(path), sha256=CENTER_SHA256, result=result,
                metrics=recovery_metrics)


def review_template(stage, root, center, provenance_path):
    if stage not in APPROVAL_STATUS:
        raise ValueError(stage)
    template=dict(schema=1, family=LABEL, stage=stage, status='NOT_REVIEWED',
                center_sha256=center['sha256'], runner_sha256=digest(__file__),
                code=dependency_hashes(), task_map=[list(x) for x in TASK_MAP],
                output_root=str(root), allowed_tasks=[],
                provenance_report=str(Path(provenance_path).resolve()),
                provenance_report_sha256=digest(provenance_path),
                note='Parent must review this stage and publish a NEW immutable marker. Failed calibration unchanged.')
    if stage=='later':
        template.update(first_wave=load_firstwave(root,center),lead_verdict='NOT_REVIEWED')
    return template


def reviewed_stage(stage, root, center):
    path = env_path('S2_HALF_FIRST_REVIEW_MARKER' if stage=='first' else 'S2_HALF_LATER_REVIEW_MARKER')
    if stage=='later' and path==env_path('S2_HALF_FIRST_REVIEW_MARKER'):
        raise RuntimeError('Later-stage marker must be separate; never mutate the first marker')
    row = json.loads(path.read_text())
    expected = dict(schema=1, family=LABEL, stage=stage, status=APPROVAL_STATUS[stage],
                    center_sha256=CENTER_SHA256, runner_sha256=digest(__file__),
                    code=dependency_hashes(), task_map=[list(x) for x in TASK_MAP],
                    output_root=str(root))
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Explicit parent provenance-review marker identity/status mismatch')
    allowed = row.get('allowed_tasks')
    stage_tasks=list(range(4)) if stage=='first' else list(range(4,12))
    if not isinstance(allowed,list) or allowed!=stage_tasks or any(type(t) is not int for t in allowed):
        raise RuntimeError('Task is not authorized by bounded parent-review marker')
    provenance = Path(row.get('provenance_report', ''))
    if not provenance.is_absolute() or not provenance.is_file() or digest(provenance) != row.get('provenance_report_sha256'):
        raise RuntimeError('Parent-reviewed provenance report hash mismatch/unavailable')
    # Parent approval is meaningful only after the no-solve probe completed its
    # required extraction and preserved its originals. Coordinate differences
    # themselves are for parent interpretation, not an automatic shape gate.
    evidence = json.loads(provenance.read_text())
    if evidence.get('execution_error') or evidence.get('critical_evidence_available') is not True or evidence.get('original_file_identity_unchanged') is not True:
        raise RuntimeError('Provenance report lacks completed/preserved critical evidence')
    review=dict(path=str(path),sha256=digest(path),stage=stage,provenance_report=str(provenance),
                provenance_report_sha256=row['provenance_report_sha256'],allowed_tasks=allowed)
    if stage=='later':
        first=load_firstwave(root,center)
        if row.get('first_wave')!=first:
            raise RuntimeError('Later-stage marker first-wave checkpoint identities mismatch')
        verdict=row.get('lead_verdict')
        if not isinstance(verdict,str) or not verdict.strip() or verdict.strip()=='NOT_REVIEWED':
            raise RuntimeError('Later-stage marker needs an explicit lead review verdict')
        if (review['provenance_report'],review['provenance_report_sha256'])!=(first['provenance_report'],first['provenance_report_sha256']):
            raise RuntimeError('Stage provenance identity must remain shared')
        review.update(first_wave=first,lead_verdict=verdict)
    return review


def reviewed_family(task,root,center):
    return reviewed_stage('first' if task<4 else 'later',root,center)


def load_firstwave(root,center):
    first_review=reviewed_stage('first',root,center)
    checkpoints={}
    for task in range(4):
        path=root/f'task_{task:02d}.json'
        saved=recovery.read_checkpoint(path,identity(task,center,first_review))
        if saved is None:
            raise RuntimeError('Later stage requires completed first-wave task '+str(task))
        verify_result(saved,task,root)
        checkpoints[str(task)]=dict(path=str(path),sha256=digest(path),
            peak3_tout=float(saved['peak3_tout']),peak3_modal=float(saved['peak3_modal']),lam_nm=float(saved['lam_nm']))
    return dict(first_marker=first_review['path'],first_marker_sha256=first_review['sha256'],
                checkpoints=checkpoints,provenance_report=first_review['provenance_report'],
                provenance_report_sha256=first_review['provenance_report_sha256'])


def identity(task, center, review):
    p,index,h,sign,label=task_params(task)
    return dict(schema=1, task=int(task), spec=dataclasses.asdict(SPEC),
                point=p.tolist(),index=index,class_label=label,sign=sign,perturbation_nm=h,
                task_map=[list(x) for x in TASK_MAP], center_sha256=CENTER_SHA256,
                center_checkpoint=center['checkpoint'], parent_review=review,
                code=dependency_hashes())


def verify_result(result,task,root):
    p,index,h,sign,label=task_params(task)
    expected=dict(task=task,index=index,perturbation_nm=h,sign=sign,delta_nm=h*sign,class_label=label,
        reviewed_center_sha256=CENTER_SHA256,existing_overall_gradient_gate='FAILED_UNCHANGED',
        comparison_status='REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS')
    if any(result.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.asarray(result.get('params'),float),p):
        raise RuntimeError('Completed half-step result identity mismatch')
    diag.verify_artifacts(result)
    if digest(result['ports_npz']) != result['ports_sha256']:
        raise RuntimeError('Completed half-step ports hash mismatch')
    source=Path(result['source']).resolve();ports=Path(result['ports_npz']).resolve()
    if not ports.parent.is_relative_to(root.resolve()) or not ports.parent.name.startswith(f'task_{task:02d}_') or not source.is_relative_to(ports.parent) or source.name!='fwd_default.fsp' or Path(result['h5']).resolve()!=source.with_suffix('')/(source.stem+'_output.h5'):
        raise RuntimeError('Completed half-step artifact path mismatch')
    recovery.successful_log(source)
    with np.load(ports,allow_pickle=False) as archive:
        metrics=recovery.score({k:archive[k] for k in archive.files})
    if any(not np.isfinite(float(result[k])) or abs(float(metrics[k])-float(result[k]))>1e-12 for k in diag.METRICS):
        raise RuntimeError('Completed half-step scalar/ports evidence mismatch')


def forward_only(task, root, center, review):
    p,index,h,sign,label=task_params(task)
    attempt = Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_', dir=root))
    project, _ = eng.make_project(SPEC, str(attempt), eng.import_lumopt2())
    try:
        project.set_fwd(p)
        f = project.fdtd_session.fdtd
        mesh = {a: np.asarray(f.getresult('FDTD', a)).copy() for a in ('x', 'y', 'z')}
        f.setnamed('optimization_dft', 'enabled', 0)
        for a, grid in mesh.items():
            if not np.array_equal(grid, np.asarray(f.getresult('FDTD', a))):
                raise RuntimeError('Disabling FD field storage changed '+a+' mesh')
        configs = list(project.fom.config_map.config_map.values())
        if len(configs) != 1:
            raise RuntimeError('Only one forward configuration is supported')
        source = Path(project.fom.config_map.project_folder)/configs[0]['fwd_filename']
        if not source.resolve().is_relative_to(attempt.resolve()):
            raise RuntimeError('Forward destination escaped unique attempt')
        f.save(str(source))  # only this attempt's new forward setup
        project.fdtd_session.invalidate_loaded_state()
        project.run_forward()  # exactly ONE forward; no adjoint/optimizer
        recovery.successful_log(source)
        h5 = source.with_suffix('')/(source.stem+'_output.h5')
        if not h5.is_file():
            raise RuntimeError('Preserved companion HDF5 is required')
        f.load(str(source))
        arrays = recovery.port_arrays(f)
        ports = attempt/'ports.npz'
        with ports.open('xb') as stream:
            np.savez(stream, **arrays)
        measured = recovery.score(arrays)
        return dict(task=int(task),index=index,class_label=label,perturbation_nm=h,sign=sign,
                    delta_nm=h*sign, params=p.tolist(), source=str(source), h5=str(h5),
                    ports_npz=str(ports), ports_sha256=digest(ports), **measured,
                    reviewed_center_sha256=CENTER_SHA256, parent_review=review,
                    new_center_comparison=diag.differences(measured, center['metrics']),
                    comparison_status='REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS',
                    existing_overall_gradient_gate='FAILED_UNCHANGED',
                    mesh_points={a:len(grid.ravel()) for a,grid in mesh.items()})
    finally:
        project.fdtd_session.fdtd.close()


def main(task_idx):
    task_params(task_idx)  # reject out-of-range/type before any checkpoint/CAD
    task = int(task_idx)
    root = output_root()
    center = load_center()
    review = reviewed_family(task, root, center)  # no marker => no CAD/solve
    expected = identity(task, center, review)
    root.mkdir(parents=True, exist_ok=True)
    path = root/f'task_{task:02d}.json'
    saved = recovery.read_checkpoint(path, expected)
    if saved is not None:
        verify_result(saved,task,root)
        print('REUSED half-step checkpoint',path,flush=True)
        return saved
    if os.environ.get('KEEP_H5') != '1':
        raise RuntimeError('Half-step diagnostic requires KEEP_H5=1 before CAD/solve')
    result = forward_only(task, root, center, review)
    recovery.checkpoint(path, expected, result)  # atomic exclusive publication
    print('HALFSTEP checkpoint; stage review required',path,'class',result['class_label'],'delta_nm',result['delta_nm'],flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-template', metavar='PROVENANCE_JSON',
                        help='Print NOT_REVIEWED template; writes nothing and runs no CAD')
    parser.add_argument('--task', type=int)
    parser.add_argument('--stage',choices=('first','later'))
    args = parser.parse_args()
    if args.review_template:
        if not args.stage:parser.error('--stage required for review template')
        print(json.dumps(review_template(args.stage,output_root(),load_center(),args.review_template),indent=2))
    elif args.task is not None:
        main(args.task)
    else:
        parser.error('--review-template or --task is required')
