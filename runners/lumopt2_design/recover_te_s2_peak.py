"""Validate S2's actual v3 peak3 objective using recovered broadband fields.

Tasks 0/1: Re/Im gradients, no FDTD solves. Task 2: compare with the peak3
values from recover_te_s2's twelve FD legs, fit C, and write a verdict.
The vendor port adjoint source is independent of the objective Jacobian;
the Jacobian weights are applied during gradient-field assembly. Therefore
the original broadband port forward/adjoint fields can be reused exactly.
Keep this study separate from the immutable in-flight legacy recovery.
"""
import dataclasses
import hashlib
from pathlib import Path
import tempfile

import numpy as np

import config
from runners.lumopt2_design import recover_te_s2 as recovery

eng, val = recovery.eng, recovery.val
N_TASKS = 3
SPEC = dataclasses.replace(recovery.SPEC, label='lumopt2_te_s2_peak_calibration_20261006')


def identity(task):
    return dict(base=recovery.identity(task), objective='peak3_abs_T_out',
                code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def gradient(task, root):
    imaginary = task == 1
    spec = dataclasses.replace(SPEC, adj_phase_fix=imaginary,
                               adj_fix_re=0.0 if imaginary else 1.0,
                               adj_fix_im=1.0 if imaginary else 0.0)
    attempt = Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_', dir=root))
    project, wl = eng.make_project(spec, str(attempt), eng.import_lumopt2())
    project.fom.fct = eng.make_fct_peak(wl)
    project.generate()
    folder = recovery.OLD/('lumopt2_te_s2_cport_im_c_files' if imaginary else
                           'lumopt2_te_s2_cport_fd_c_files')
    forward, adjoint = folder/'fwd_default.fsp', folder/'adj_default_Port_2.fsp'
    recovery.successful_log(forward)
    recovery.successful_log(adjoint)
    cm = project.fom.config_map
    for c in cm.config_map.values():
        c['fwd_filename'] = str(forward)
        for a in c['adjoints']:
            a['adj_filename'] = str(adjoint)
    for info in cm._sim_result_info.values():
        info['fwd_filename'], info['adj_filename'] = str(forward), str(adjoint)
    fields = project.fom.calculate_gradient_fields(
        fdtd_session=project.fdtd_session, parametrization=project.parametrization,
        symmetry_factors=project.fom_symmetry_factors)
    p = val.te_point(spec)
    project.fdtd_session.fdtd.switchtolayout()
    project.parametrization.update_structure(project.fdtd_session, p)
    project.fdtd_session.invalidate_loaded_state()
    g = project.parametrization.compute_gradient_from_fields(fields, project.fdtd_session, p)
    return dict(gradient=np.asarray(g)[recovery.INDICES].tolist(),
                indices=recovery.INDICES, forward=str(forward), adjoint=str(adjoint))


def fit(root):
    legacy = Path(config.BASE_SAVE_DIR)/'recover_te_s2/results'/recovery.SPEC.label
    legs = {i: recovery.read_checkpoint(legacy/f'task_{i:02d}.json', recovery.identity(i))
            for i in range(1, 13)}
    quadratures = [recovery.read_checkpoint(root/f'task_{i:02d}.json', identity(i)) for i in (0, 1)]
    if any(r is None for r in list(legs.values())+quadratures):
        raise RuntimeError('Peak calibration requires every FD leg and both quadratures')
    fd = np.array([(legs[i]['peak3_tout']-legs[i+1]['peak3_tout'])/(2*recovery.PERTURBATION)
                   for i in range(1, 13, 2)])
    re, im = [q['gradient'] for q in quadratures]
    coeff, worst = val.fit_port(fd, re, im, recovery.LABELS)
    A = np.array([re, im]).T
    loo = []
    for i in range(len(fd)):
        keep = np.arange(len(fd)) != i
        c = np.linalg.lstsq(A[keep], fd[keep], rcond=None)[0]
        loo.append(float((A[i]@c-fd[i])/fd[i]) if fd[i] != 0 else None)
    model = A@coeff
    passed = (np.all(np.isfinite(fd)) and np.linalg.matrix_rank(A) == 2
              and all(v <= .1 for v in worst.values())
              and all(v is not None and abs(v) <= .1 for v in loo)
              and np.all(np.sign(model) == np.sign(fd)))
    return dict(fd=fd.tolist(), re=re, im=im, coefficients=coeff, worst=worst,
                held_out_relative=loo, passed=bool(passed),
                condition_number=float(np.linalg.cond(A)),
                note='Do not install coefficients unless passed, then run S2 smoke/toy.')


def main(task_idx):
    task = int(task_idx)
    if task not in range(N_TASKS):
        raise ValueError(task)
    root = Path(config.RESULTS_DIR)/SPEC.label
    root.mkdir(parents=True, exist_ok=True)
    path, expected = root/f'task_{task:02d}.json', identity(task)
    result = recovery.read_checkpoint(path, expected)
    if result is None:
        if task == 2:
            legacy = Path(config.BASE_SAVE_DIR)/'recover_te_s2/results'/recovery.SPEC.label
            legacy_path = legacy/'task_15.json'
            if recovery.read_checkpoint(legacy_path, recovery.identity(15)) is None:
                recovery.checkpoint(legacy_path, recovery.identity(15), recovery.fit(legacy))
        result = gradient(task, root) if task < 2 else fit(root)
        recovery.checkpoint(path, expected, result)
    print('PEAK CALIBRATION', task, result, flush=True)
    if task == 2 and not result['passed']:
        raise RuntimeError('S2 peak3 calibration failed; campaign remains blocked')
    return result
