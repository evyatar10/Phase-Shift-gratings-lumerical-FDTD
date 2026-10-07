"""Prepared, gated, four-task forward-only corr30 step-convergence diagnostic.

Task 0/1: +/-1 nm. Task 2/3: +/-0.25 nm. No nominal, adjoint, optimizer,
six-class batch, fitting or automatic calibration verdict. Import with the
existing Athena wrapper, then call main(task_idx). Parent must review exact
provenance and publish the explicit marker before any new forward solve.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(os.environ.get('S2_CONV_REPO', Path.cwd())).resolve()))
import numpy as np
import config
from runners.lumopt2_design import diagnose_te_s2_fdstep as diag

eng, val, recovery = diag.eng, diag.val, diag.recovery
IS_HELPER = True
N_TASKS = 4
LABEL = 'lumopt2_te_s2_corr30_stepconvergence_20261007'
SPEC = dataclasses.replace(diag.SPEC, label=LABEL)
INDEX = eng.layout(SPEC.n_free).SL_CORR.start + 29
TASK_MAP = ((1.0, 1), (1.0, -1), (.25, 1), (.25, -1))
CENTER_SHA256 = 'cd2dd4a58be7b83c46ef6a6ac90a874e5fc33edda884edfc8ac50ce5e3be4a06'
APPROVAL_STATUS = 'APPROVED_FOR_BOUNDED_FORWARD_STEP_DIAGNOSTIC'


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
        raise ValueError('Task must be integer 0..3; got ' + repr(task))
    h, sign = TASK_MAP[task]
    p = val.te_point(SPEC).copy()
    p[INDEX] += sign*h
    bounds = np.asarray(eng.param_bounds(SPEC), dtype=float)
    if not np.all((p >= bounds[:, 0]) & (p <= bounds[:, 1])):
        raise RuntimeError('Convergence leg is outside its own bounds')
    return p, h, sign


def dependency_hashes():
    # Existing diagnostic's exact frozen dependency list includes itself,
    # engine, recovery, validator, S2, builder, simulation_config, helpers, v3.
    code = dict(diag.identity(0)['code'])
    code[Path(__file__).name] = digest(__file__)
    code[Path(config.__file__).name] = digest(config.__file__)
    return code


def output_root():
    root = env_path('S2_CONV_OUTPUT_ROOT')
    if root.name != LABEL:
        raise RuntimeError('Output root must end with unique study label ' + LABEL)
    center = env_path('S2_CONV_CENTER_CHECKPOINT')
    if root == center.parent or root.is_relative_to(center.parent):
        raise RuntimeError('Output must be separate from center evidence')
    return root


def load_center():
    if dataclasses.asdict(SPEC) != dataclasses.asdict(dataclasses.replace(diag.SPEC, label=LABEL)):
        raise RuntimeError('New runner numerical family differs from frozen original diagnostic')
    path = env_path('S2_CONV_CENTER_CHECKPOINT')
    if digest(path) != CENTER_SHA256:
        raise RuntimeError('Pinned NEW center checkpoint hash mismatch')
    marker = path.parent/'center_reviewed.sha256'
    if not marker.is_file() or marker.read_text().strip() != CENTER_SHA256:
        raise RuntimeError('Exact original reviewed-center marker is required')
    row = json.loads(path.read_text())
    identity, result = row['identity'], row['result']
    point = val.te_point(SPEC)
    if identity.get('task') != 0 or identity.get('sign') != 0 or identity.get('index') != INDEX or identity.get('perturbation_nm') != .5:
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


def review_template(root, center, provenance_path):
    return dict(schema=1, family=LABEL, status='NOT_REVIEWED',
                center_sha256=center['sha256'], runner_sha256=digest(__file__),
                code=dependency_hashes(), task_map=[list(x) for x in TASK_MAP],
                output_root=str(root), allowed_tasks=[],
                provenance_report=str(Path(provenance_path).resolve()),
                provenance_report_sha256=digest(provenance_path),
                note='Parent must inspect provenance, decide tasks, then set explicit approval status. Failed calibration unchanged.')


def reviewed_family(task, root, center):
    path = env_path('S2_CONV_REVIEW_MARKER')
    row = json.loads(path.read_text())
    expected = dict(schema=1, family=LABEL, status=APPROVAL_STATUS,
                    center_sha256=CENTER_SHA256, runner_sha256=digest(__file__),
                    code=dependency_hashes(), task_map=[list(x) for x in TASK_MAP],
                    output_root=str(root))
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Explicit parent provenance-review marker identity/status mismatch')
    allowed = row.get('allowed_tasks')
    if not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed) or any(type(t) is not int or t not in range(N_TASKS) for t in allowed) or task not in allowed:
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
    return dict(path=str(path), sha256=digest(path), provenance_report=str(provenance),
                provenance_report_sha256=row['provenance_report_sha256'], allowed_tasks=allowed)


def identity(task, center, review):
    p, h, sign = task_params(task)
    return dict(schema=1, task=int(task), spec=dataclasses.asdict(SPEC),
                point=p.tolist(), index=INDEX, sign=sign, perturbation_nm=h,
                task_map=[list(x) for x in TASK_MAP], center_sha256=CENTER_SHA256,
                center_checkpoint=center['checkpoint'], parent_review=review,
                code=dependency_hashes())


def verify_result(result):
    diag.verify_artifacts(result)
    if digest(result['ports_npz']) != result['ports_sha256']:
        raise RuntimeError('Completed convergence ports hash mismatch')


def forward_only(task, root, center, review):
    p, h, sign = task_params(task)
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
        return dict(task=int(task), index=INDEX, perturbation_nm=h, sign=sign,
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
        verify_result(saved)
        print('REUSED convergence checkpoint', path, flush=True)
        return saved
    if os.environ.get('KEEP_H5') != '1':
        raise RuntimeError('Convergence diagnostic requires KEEP_H5=1 before CAD/solve')
    result = forward_only(task, root, center, review)
    recovery.checkpoint(path, expected, result)  # atomic exclusive publication
    print('CONVERGENCE checkpoint; review required', path, 'delta_nm', result['delta_nm'], flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-template', metavar='PROVENANCE_JSON',
                        help='Print NOT_REVIEWED template; writes nothing and runs no CAD')
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    if args.review_template:
        print(json.dumps(review_template(output_root(), load_center(), args.review_template), indent=2))
    elif args.task is not None:
        main(args.task)
    else:
        parser.error('--review-template or --task is required')
