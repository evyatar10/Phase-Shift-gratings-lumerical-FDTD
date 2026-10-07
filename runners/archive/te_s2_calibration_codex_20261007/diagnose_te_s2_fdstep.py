"""Forward-only S2 FD-step diagnosis; nominal repeat precedes reviewed ±legs.

Task0: recovered calibration te_point, unchanged. Task1/2: corr30 ±0.5nm.
This measures nominal reproducibility and local FD response, not an optimizer
or a replacement for the failed 10% gradient gate. Preserve FSP/HDF5/ports.
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
from runners.lumopt2_design import recover_te_s2 as recovery
from runners.lumopt2_design import validate_te as val

IS_HELPER = True
N_TASKS = 3
SPEC = dataclasses.replace(recovery.SPEC,label='lumopt2_te_s2_fdstep_diagnostic_20261007')
INDEX = eng.layout(SPEC.n_free).SL_CORR.start+29
PERTURBATION_NM = .5
REFERENCE_SHA256 = '732ca5752ff2b0449a892ebafe7ba6a87a5ee1fce63bd929f8f7343bf50df13f'
METRICS = ('fom','peak3_tout','peak3_modal','lam_nm','modal_t','width_nm',
           'tout_at_modal_peak','p1_T_in')


def output_root():
    return Path(config.RESULTS_DIR)/SPEC.label


def reference_pin():
    path = os.environ.get('S2_FDSTEP_REFERENCE_CHECKPOINT')
    default = Path(config.BASE_SAVE_DIR)/'recover_te_s2/results'/recovery.SPEC.label/'task_00.json'
    digest = os.environ.get('S2_FDSTEP_REFERENCE_SHA256',REFERENCE_SHA256)
    return Path(path) if path else default,digest


def task_params(task):
    if int(task) not in range(N_TASKS):
        raise ValueError(task)
    p = val.te_point(SPEC).copy()
    sign = {0:0,1:1,2:-1}[int(task)]
    p[INDEX] += sign*PERTURBATION_NM
    b = np.asarray(eng.param_bounds(SPEC),dtype=float)
    if not np.all((p>=b[:,0]) & (p<=b[:,1])):
        raise RuntimeError('Diagnostic point is outside its own bounds')
    return p,sign


def identity(task):
    p,sign = task_params(task)
    import bragg_device,simulation_config,sim_helpers
    from runners.lumopt2_design import v3_step
    modules=(eng,recovery,val,val.S2M,bragg_device,simulation_config,sim_helpers,v3_step)
    reference,digest=reference_pin()
    sources=[Path(__file__)]+[Path(m.__file__) for m in modules]
    return dict(schema=1,task=int(task),spec=dataclasses.asdict(SPEC),point=p.tolist(),
        index=INDEX,sign=sign,perturbation_nm=PERTURBATION_NM,
        reference_checkpoint=str(reference),reference_sha256=digest,
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def load_reference_center():
    path,digest=reference_pin()
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise RuntimeError('Reference-center checkpoint hash mismatch')
    row=json.loads(raw)
    p,_=task_params(0)
    if row.get('identity',{}).get('task')!=0 or not np.array_equal(
            np.asarray(row['identity']['point'],dtype=float),p):
        raise RuntimeError('Reference is not the original recovered calibration point')
    prior=row['identity']['spec']
    keys=('polarization','n_periods_side','n_free','bare','pitch_nm','scan_center_nm',
          'scan_width_nm','n_wl_points','mesh_refinement','region_dx_nm',
          'region_y_half_nm','box_y_um','box_z_mult')
    if any(prior.get(k)!=getattr(SPEC,k) for k in keys):
        raise RuntimeError('Reference-center numerics/geometry family mismatch')
    metrics=row['result']['s2_gate']
    if not all(k in metrics and np.isfinite(float(metrics[k])) for k in METRICS):
        raise RuntimeError('Reference center lacks finite comparison metrics')
    return dict(path=str(path),sha256=digest,old_source=metrics['source'],
                metrics={k:float(metrics[k]) for k in METRICS})


def reviewed_center(root):
    path=root/'task_00.json'
    marker=root/'center_reviewed.sha256'
    if not path.is_file() or not marker.is_file():
        raise RuntimeError('Task1/2 require task0 checkpoint and explicit reviewed-center marker')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if marker.read_text().strip()!=digest:
        raise RuntimeError('Reviewed-center marker does not match exact task0 checkpoint')
    result=recovery.read_checkpoint(path,identity(0))
    if result is None or result.get('task')!=0 or result.get('sign')!=0:
        raise RuntimeError('Invalid nominal-center checkpoint')
    verify_artifacts(result)
    return result,digest


def verify_artifacts(result):
    for key in ('source','h5','ports_npz'):
        if not Path(result[key]).is_file():
            raise RuntimeError('Completed diagnostic artifact missing: '+key)


def differences(new,base):
    return {k:dict(new=float(new[k]),reference=float(base[k]),
        signed_delta=float(new[k]-base[k]),absolute_delta=float(abs(new[k]-base[k])),
        relative_delta=(float((new[k]-base[k])/base[k]) if base[k]!=0 else None))
        for k in METRICS}


def forward_only(task,root,lmpt,reference,center=None,center_digest=None):
    p,sign=task_params(task)
    attempt=Path(tempfile.mkdtemp(prefix=f'task_{int(task):02d}_',dir=root))
    project,_=eng.make_project(SPEC,str(attempt),lmpt)
    try:
        project.set_fwd(p)
        f=project.fdtd_session.fdtd
        mesh={a:np.asarray(f.getresult('FDTD',a)).copy() for a in ('x','y','z')}
        f.setnamed('optimization_dft','enabled',0)
        for a,grid in mesh.items():
            if not np.array_equal(grid,np.asarray(f.getresult('FDTD',a))):
                raise RuntimeError('Disabling FD field storage changed '+a+' mesh')
        configs=list(project.fom.config_map.config_map.values())
        if len(configs)!=1:
            raise RuntimeError('Diagnostic supports one forward configuration only')
        source=Path(project.fom.config_map.project_folder)/configs[0]['fwd_filename']
        if not source.resolve().is_relative_to(attempt.resolve()):
            raise RuntimeError('Forward destination escaped unique attempt')
        f.save(str(source))
        project.fdtd_session.invalidate_loaded_state()
        project.run_forward()               # no adjoint/gradient/optimizer call
        recovery.successful_log(source)
        h5=source.with_suffix('')/(source.stem+'_output.h5')
        if not h5.is_file():
            raise RuntimeError('Companion HDF5 missing; diagnostic cannot claim completion')
        f.load(str(source))
        arrays=recovery.port_arrays(f)
        ports=attempt/'ports.npz'
        np.savez(ports,**arrays)
        measured=recovery.score(arrays)
        result=dict(task=int(task),index=INDEX,sign=sign,delta_nm=sign*PERTURBATION_NM,
            params=p.tolist(),source=str(source),h5=str(h5),ports_npz=str(ports),
            **measured,reference=reference,
            old_center_comparison=differences(measured,reference['metrics']),
            comparison_status='REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS',
            existing_overall_gradient_gate='FAILED_UNCHANGED',
            mesh_points={a:len(np.asarray(grid).ravel()) for a,grid in mesh.items()})
        if center is not None:
            result.update(reviewed_center_sha256=center_digest,
                          new_center_comparison=differences(measured,center))
        return result
    finally:
        project.fdtd_session.fdtd.close()


def main(task_idx):
    task=int(task_idx)
    task_params(task)
    root=output_root()
    root.mkdir(parents=True,exist_ok=True)
    reference=load_reference_center()        # scalar checkpoint only; no old CAD
    center,digest=reviewed_center(root) if task else (None,None)
    expected=identity(task)
    if digest is not None:
        expected['reviewed_center_sha256']=digest
    path=root/f'task_{task:02d}.json'
    saved=recovery.read_checkpoint(path,expected)
    if saved is not None:
        verify_artifacts(saved)
        print('REUSED diagnostic checkpoint',path,flush=True)
        return saved
    if os.environ.get('KEEP_H5')!='1':
        raise RuntimeError('Diagnostic requires KEEP_H5=1 before CAD/solve')
    result=forward_only(task,root,eng.import_lumopt2(),reference,center,digest)
    recovery.checkpoint(path,expected,result)
    print('DIAGNOSTIC checkpoint, review required',path,
          {k:result[k] for k in ('task','delta_nm','peak3_tout','peak3_modal','lam_nm')},
          flush=True)
    return result
