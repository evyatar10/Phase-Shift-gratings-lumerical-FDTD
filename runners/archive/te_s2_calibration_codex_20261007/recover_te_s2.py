"""Recover interrupted S2 port calibration without overwriting the old study.

Tasks: 0 exports stored port spectra (no solve); 1..12 are individual central
FD legs, ordered like validate_te._indices, positive then negative. Task 2
reuses the completed p0 negative leg. Tasks 13/14 reconstruct Re/Im from
stored forward/adjoint fields (no solve); 15 fits the completed evidence.

TE N=98/free=60, h=350 nm, pitch=491.06 nm, bare. Calibration resonance
1560.878859628509 nm, 2 nm window/501 points, original softmax functional.
This is calibration recovery, not authorization to start a campaign.
Each numerical result is an immutable, fingerprinted checkpoint. A cold
restart loses at most one solve. Unfinished attempts are kept in unique
directories. FD-only runs disable optimization_dft after verifying that the
locked mesh is identical; geometry, ports and other monitors are unchanged.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import validate_te as val

N_TASKS = 16
SPEC = val._port_spec(val.S2M, 1, '_cport_recovery_20261006')
INDICES, LABELS = val._indices(SPEC)
PERTURBATION = 2.0
OLD = Path('/work/results/validate_te/results/validate_te_s2')


def identity(task):
    """Reject checkpoints after any physics, objective or recovery-code change."""
    sources = [Path(__file__), Path(eng.__file__), Path(val.__file__),
               Path(val.S2M.__file__)]
    import bragg_device
    import simulation_config
    import sim_helpers
    from runners.lumopt2_design import v3_step
    sources += [Path(m.__file__) for m in (bragg_device, simulation_config, sim_helpers, v3_step)]
    return dict(schema=1, task=int(task), spec=dataclasses.asdict(SPEC),
                point=val.te_point(SPEC).tolist(), indices=INDICES,
                perturbation=PERTURBATION,
                code={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def read_checkpoint(path, expected):
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding='utf-8'))
    # Round-trip normalizes tuples in the dataclass to JSON lists.
    if row['identity'] != json.loads(json.dumps(expected)):
        raise RuntimeError(f'Checkpoint identity mismatch: {path}')
    return row['result']


def checkpoint(path, expected, result):
    """Atomic publication, without replacing an earlier completed result."""
    payload = json.dumps(dict(identity=expected, result=result), allow_nan=False, indent=2)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    # Hard-link publication is atomic and fails if the target already exists.
    # Keep the small temporary evidence file, in line with preservation policy.
    os.link(tmp, path)


def port_arrays(fdtd):
    out = {}
    for port in (1, 2):
        data = fdtd.getresult(f'FDTD::ports::Port_{port}', 'expansion for port monitor')
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                out[f'p{port}_{key}'] = value
    return out


def score(arrays):
    wl = np.asarray(arrays['p2_lambda']).ravel() * 1e9
    tout = np.abs(np.asarray(arrays['p2_T_out']).ravel())
    modal = np.abs(np.asarray(arrays['p2_S']).ravel())**2
    if len(wl) != SPEC.n_wl_points or not np.all(np.isfinite(tout)):
        raise RuntimeError('Missing, nonfinite or wrong-size port spectrum')
    lp, tp, width = eng.measure_peak(wl, modal)
    if width is None or tp < eng.DEAD_T_FLOOR:
        raise RuntimeError('Dead or off-window calibration result')
    k = int(np.argmin(abs(wl-lp)))
    result = dict(fom=float(eng.make_fct(wl)(tout)),
                  peak3_tout=float(eng.make_fct_peak(wl)(tout)),
                  peak3_modal=float(eng.make_fct_peak(wl)(modal)),
                  lam_nm=float(lp), modal_t=float(tp),
                  tout_at_modal_peak=float(tout[k]),
                  ratio=float(tout[k]/modal[k]), width_nm=float(width))
    for key in ('p1_T_in', 'p1_T_out', 'p1_T_total'):
        if key in arrays:
            result[key] = float(np.real(np.asarray(arrays[key]).ravel()[k]))
    return result


def successful_log(fsp):
    log = fsp.with_name(fsp.stem+'_p0.log')
    text = log.read_text(errors='replace')
    if 'Simulation completed successfully' not in text:
        raise RuntimeError(f'No successful solve marker in {log}')


def export_stored(root, lmpt):
    cases = {
        's1_anchor': OLD.parent/'validate_te_s1/lumopt2_te_s1_anchor_files/fwd_default.fsp',
        's1_v3_toy': OLD.parent/'validate_te_s1_v3/lumopt2_te_s1_v3_toy_files/fwd_default.fsp',
        's2_gate': OLD/'lumopt2_te_s2_cport_fd_c_files/fwd_default.fsp',
        's2_p0_neg': OLD/'lumopt2_te_s2_cport_fd_c_files/fwd_default_fd_p0_neg.fsp',
    }
    import lumapi
    results = {}
    with lumapi.FDTD(hide=True) as fdtd:
        for label, source in cases.items():
            h5 = source.parent/source.stem/(source.stem+'_output.h5')
            if not h5.exists():
                results[label] = dict(source=str(source), unavailable=f'Missing companion {h5}')
                print(label, results[label], flush=True)
                continue
            successful_log(source)
            fdtd.load(str(source))
            arrays = port_arrays(fdtd)
            np.savez(root/(label+'_ports.npz'), **arrays)
            results[label] = dict(source=str(source), **score(arrays))
            print(label, results[label], flush=True)
    return results


def fd_leg(task, root, lmpt):
    index = INDICES[(task-1)//2]
    sign = 1 if task % 2 else -1
    p = val.te_point(SPEC)
    eng._assert_fd_legs_in_bounds(SPEC, p, INDICES, PERTURBATION)
    p[index] += sign*PERTURBATION
    attempt = Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_', dir=root))
    if task == 2:
        # Verified completed solver log from 170505_14. Never resave this file.
        source = OLD/'lumopt2_te_s2_cport_fd_c_files/fwd_default_fd_p0_neg.fsp'
        successful_log(source)
        import lumapi
        with lumapi.FDTD(hide=True) as f:
            f.load(str(source))
            arrays = port_arrays(f)
    else:
        project, _ = eng.make_project(SPEC, str(attempt), lmpt)
        project.set_fwd(p)
        f = project.fdtd_session.fdtd
        mesh = {a: np.asarray(f.getresult('FDTD', a)) for a in ('x', 'y', 'z')}
        f.setnamed('optimization_dft', 'enabled', 0)
        for a, grid in mesh.items():
            if not np.array_equal(grid, np.asarray(f.getresult('FDTD', a))):
                raise RuntimeError(f'Disabling FD field storage changed {a} mesh')
        configs = list(project.fom.config_map.config_map.values())
        if len(configs) != 1:
            raise RuntimeError('Recovery supports one forward configuration only')
        source = Path(project.fom.config_map.project_folder)/configs[0]['fwd_filename']
        # set_fwd created this attempt's new file; originals are never touched.
        f.save(str(source))
        project.fdtd_session.invalidate_loaded_state()
        project.run_forward()
        successful_log(source)
        f.load(str(source))
        arrays = port_arrays(f)
    np.savez(attempt/'ports.npz', **arrays)
    return dict(index=index, sign=sign, params=p.tolist(), source=str(source), **score(arrays))


def recover_quadrature(task, root, lmpt):
    """Reproduce vendor gradient assembly from saved fields; zero engine solves."""
    imaginary = task == 14
    spec = dataclasses.replace(SPEC, adj_phase_fix=imaginary,
                               adj_fix_re=0.0 if imaginary else 1.0,
                               adj_fix_im=1.0 if imaginary else 0.0)
    attempt = Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_', dir=root))
    project, _ = eng.make_project(spec, str(attempt), lmpt)
    project.generate()
    folder = OLD/('lumopt2_te_s2_cport_im_c_files' if imaginary else
                  'lumopt2_te_s2_cport_fd_c_files')
    forward, adjoint = folder/'fwd_default.fsp', folder/'adj_default_Port_2.fsp'
    successful_log(forward)
    successful_log(adjoint)
    cm = project.fom.config_map
    for c in cm.config_map.values():
        c['fwd_filename'] = str(forward)
        for a in c['adjoints']:
            a['adj_filename'] = str(adjoint)
    for info in cm._sim_result_info.values():
        info['fwd_filename'], info['adj_filename'] = str(forward), str(adjoint)
    # No compute_gradient() call: that would set_fwd and overwrite old evidence.
    fields = project.fom.calculate_gradient_fields(
        fdtd_session=project.fdtd_session, parametrization=project.parametrization,
        symmetry_factors=project.fom_symmetry_factors)
    p = val.te_point(spec)
    project.fdtd_session.fdtd.switchtolayout()
    project.parametrization.update_structure(project.fdtd_session, p)
    project.fdtd_session.invalidate_loaded_state()
    grad = project.parametrization.compute_gradient_from_fields(fields, project.fdtd_session, p)
    selected = np.asarray(grad)[INDICES]
    # Compare with the independently preserved pre-cancellation printouts.
    # Re was printed to eight decimal places; Im to nine significant digits.
    reference = np.array([-8.71524851e-4, 3.73127120e-3, -6.67376252e-2,
                          -1.93385823e-3, -1.28606055, -1.72773639e-2]) if imaginary else np.array(
                          [.00022048, -.00043038, .00807626, .00022821, .1568861, .00204085])
    if not np.allclose(selected, reference, rtol=2e-6, atol=6e-9):
        raise RuntimeError(f'Recovered gradient differs from original log: {selected!r}')
    return dict(gradient=selected.tolist(), indices=INDICES,
                forward=str(forward), adjoint=str(adjoint))


def fit(root):
    rows = {i: read_checkpoint(root/f'task_{i:02d}.json', identity(i)) for i in range(1, 15)}
    missing = [i for i, r in rows.items() if r is None]
    if missing:
        raise RuntimeError(f'Incomplete calibration tasks: {missing}')
    fd = [(rows[i]['fom']-rows[i+1]['fom'])/(2*PERTURBATION) for i in range(1, 13, 2)]
    re, im = rows[13]['gradient'], rows[14]['gradient']
    coeff, worst = val.fit_port(fd, re, im, LABELS)
    return dict(fd=fd, re=re, im=im, coefficients=coeff, worst=worst,
                passed=all(v <= .10 for v in worst.values()),
                note='Original softmax calibration; v3 functional validation still required.')


def main(task_idx):
    task = int(task_idx)
    if task not in range(N_TASKS):
        raise ValueError(task)
    root = Path(config.RESULTS_DIR)/SPEC.label
    root.mkdir(parents=True, exist_ok=True)
    path, expected = root/f'task_{task:02d}.json', identity(task)
    saved = read_checkpoint(path, expected)
    if saved is not None:
        print('REUSED', path, saved, flush=True)
        return saved
    lmpt = eng.import_lumopt2()
    if task == 0:
        result = export_stored(root, lmpt)
    elif task <= 12:
        result = fd_leg(task, root, lmpt)
    elif task <= 14:
        result = recover_quadrature(task, root, lmpt)
    else:
        result = fit(root)
    checkpoint(path, expected, result)
    print('CHECKPOINT', path, result, flush=True)
    return result
