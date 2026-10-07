"""Strict offline first-wave review evidence; never fit or approve later tasks."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sys
import zipfile
sys.dont_write_bytecode=True
import numpy as np
import numerical_helpers as h

CENTER_SHA='cd2dd4a58be7b83c46ef6a6ac90a874e5fc33edda884edfc8ac50ce5e3be4a06'
MARKER_SHA='6722e1883338c0259b6fd2aa7386a2d12f1c5124708f590c1ce30a7c266f8d02'
RUNNER_SHA='4f5f9d21cb95f5cfbcb1e0fc91ae960dc0dd57449c973369051c551c07c7bba1'
ARCHIVE_SHA='b996142caf4a66d73336ed2845a564626e6fbd9e0e9a695ee89bf498e7522204'
EXPORT_SHA='732ca5752ff2b0449a892ebafe7ba6a87a5ee1fce63bd929f8f7343bf50df13f'
FAMILY='lumopt2_te_s2_fdhalf_classcheck_20261007'
CENTER_REMOTE='/work/results/diagnose_te_s2_fdstep/results/lumopt2_te_s2_fdstep_diagnostic_20261007/task_00.json'
MARKER_REMOTE='/work/data/s2_half_classes_first_review_20261007.json'
TASK_MAP=[[name,index,.5,sign] for name,index in [('corr_30',29),('shift_30',149),('corr_1',0),('avg_1',60),('shift_1',120),('wcav',295)] for sign in [1,-1]]
EXISTING={
 's2_fdstep_task00_original.json':CENTER_SHA,
 's2_fdstep_task00_ports.npz':'4c3c526b51973a2c8889608c69958b6d60e4dc27ddcf638c9932598629a035e4',
 's2_fdstep_task01_original.json':'021ce215c64ccc13020f74af8d775f65ab97f73290e4b06f1f159297f8c7b04e',
 's2_fdstep_task01_ports.npz':'9f6d80043881877926d40a17f18e5c0e3863d8c93a6279a2a65277b96c626613',
 's2_fdstep_task02_original.json':'9e72cdc72197254594688dc0102d8a4b10155435244b545719ec4fd3fda4e808',
 's2_fdstep_task02_ports.npz':'eb140dc60e401b9472e942833437eebb74ac60f6f2641a7cd2a55c94fe1e5fe7'}


def sha(path):return h.sha(Path(path).read_bytes())


def build_pins(evidence,archive=None,watch=None):
    evidence=Path(evidence)
    for name,value in EXISTING.items():h.require(sha(evidence/name)==value,'Existing trusted evidence hash mismatch: '+name)
    h.load_json(evidence/'s2_half_classes_first_review_original.json',MARKER_SHA)
    pins=dict(schema=1,existing_files=EXISTING,marker_sha256=MARKER_SHA,center_sha256=CENTER_SHA,
              old_shift30=dict(available=False,reason='Optional original archive/watch comparison not configured'))
    if archive is not None and watch is not None:
        archive=Path(archive);watch=Path(watch)
        if not archive.is_file() or any(not(watch/f'task_{i:02d}.json').is_file() for i in (9,10)) or not(evidence/'s2_reference_task00_remote_original.json').is_file():
            pins['old_shift30']=dict(available=False,reason='Configured original archive/checkpoints/export unavailable',archive=str(archive),watch=str(watch))
        else:
            h.require(sha(archive)==ARCHIVE_SHA,'Original archive hash mismatch')
            exporter,export_sha=h.load_json(evidence/'s2_reference_task00_remote_original.json',EXPORT_SHA)
            roles={}
            for task in (9,10):
                path=watch/f'task_{task:02d}.json';row,digest=h.load_json(path)
                roles[str(task)]=dict(checkpoint=str(path.resolve()),sha256=digest,semantic_sha256=h.semantic_sha(row),
                                     code=row['identity']['code'],role='recovery_generated_forward_fd_leg')
            with zipfile.ZipFile(archive) as z:
                h.require(z.testzip() is None,'Original ZIP integrity failure')
                members={name:h.sha(z.read(name)) for name in ('base_ports.npz','task_09_ports.npz','task_10_ports.npz')}
            pins['old_shift30']=dict(available=True,archive=str(archive.resolve()),archive_sha256=ARCHIVE_SHA,members=members,
                center_export_checkpoint=str((evidence/'s2_reference_task00_remote_original.json').resolve()),
                center_export_sha256=export_sha,center_export_role='stored_forward_export_not_fsp_creation',
                center_export_semantic_sha256=h.semantic_sha(exporter),fd_roles=roles)
    return pins


def validate_new(row,task,center,marker,ports_sha):
    ident,result=row['identity'],row['result'];label,index,step,sign=TASK_MAP[task]
    h.require(ident.get('schema')==1 and type(ident.get('task')) is int and ident['task']==task and
        ident.get('class_label')==label and ident.get('index')==index and ident.get('sign')==sign and
        ident.get('perturbation_nm')==step and ident.get('task_map')==TASK_MAP,'First-wave task/index/sign/map mismatch')
    for key,value in dict(task=task,class_label=label,index=index,sign=sign,perturbation_nm=step,delta_nm=step*sign).items():
        h.require(result.get(key)==value,'First-wave result metadata mismatch: '+key)
    h.require(ident.get('center_sha256')==CENTER_SHA and result.get('reviewed_center_sha256')==CENTER_SHA and
        ident.get('center_checkpoint')==CENTER_REMOTE,'First-wave center identity mismatch')
    h.require(ident['spec'].get('label')==FAMILY and h.spec_equal(ident['spec'],center['identity']['spec']),
        'Complete numerical-family spec mismatch')
    h.require(ident.get('code')==marker['code'] and ident['code'].get('s2_fdhalf_classes.py')==RUNNER_SHA,
        'Frozen first-wave source hash mismatch')
    h.require(all(marker['code'].get(k)==v for k,v in center['identity']['code'].items()),'Frozen center source code mismatch')
    expected=dict(path=MARKER_REMOTE,sha256=MARKER_SHA,stage='first',provenance_report=marker['provenance_report'],
        provenance_report_sha256=marker['provenance_report_sha256'],allowed_tasks=[0,1,2,3])
    h.require(ident.get('parent_review')==expected and result.get('parent_review')==expected,'Parent marker provenance mismatch')
    h.require(result.get('ports_sha256')==ports_sha,'Transferred NPZ hash differs from checkpoint')
    h.require(result.get('existing_overall_gradient_gate')=='FAILED_UNCHANGED' and
              result.get('comparison_status')=='REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS','Diagnostic status mismatch')
    point=np.asarray(center['identity']['point'],float).copy();point[index]+=step*sign
    h.match_point(ident.get('point'),point,'First-wave full point mismatch')
    h.match_point(result.get('params'),point,'First-wave full result point mismatch')
    paths={k:PurePosixPath(result[k]) for k in ('source','h5','ports_npz')}
    root=PurePosixPath(marker['output_root']);attempt=paths['ports_npz'].parent
    h.require(all(p.is_absolute() and '..' not in p.parts and p.is_relative_to(root) for p in paths.values()) and
              attempt.name.startswith(f'task_{task:02d}_') and paths['source'].is_relative_to(attempt) and
              paths['source'].name=='fwd_default.fsp' and paths['ports_npz'].name=='ports.npz' and
              paths['h5']==paths['source'].with_suffix('')/(paths['source'].stem+'_output.h5'),
              'First-wave source/H5/NPZ path identity mismatch')


def old_shift_inputs(pins,center,baseline,score):
    info=pins.get('old_shift30',{})
    if info.get('available') is not True:return dict(available=False,reason=info.get('reason','Not configured')),None
    archive=Path(info['archive'])
    if not archive.is_file() or not Path(info['center_export_checkpoint']).is_file() or any(not Path(v['checkpoint']).is_file() for v in info['fd_roles'].values()):
        return dict(available=False,reason='Pinned optional archive/checkpoint no longer locally accessible'),None
    h.require(sha(archive)==info['archive_sha256']==ARCHIVE_SHA,'Optional old archive identity mismatch')
    exporter,_=h.load_json(info['center_export_checkpoint'],info['center_export_sha256'])
    h.require(info['center_export_sha256']==EXPORT_SHA and h.semantic_sha(exporter)==info['center_export_semantic_sha256'],
              'Historical center-export binding mismatch')
    oldcenter=dict(identity=exporter['identity'],result=exporter['result']['s2_gate'])
    rows=[]
    for task,sign in ((9,1),(10,-1)):
        role=info['fd_roles'][str(task)];row,_=h.load_json(role['checkpoint'],role['sha256'])
        h.require(role.get('role')=='recovery_generated_forward_fd_leg' and row['identity']['code']==role['code'] and
                  h.semantic_sha(row)==role['semantic_sha256'],'Historical FD source/content role mismatch')
        ident,result=row['identity'],row['result']
        h.require(ident['task']==task and ident['perturbation']==2. and ident['indices']==[0,29,60,120,149,295] and
                  result['index']==149 and result['sign']==sign,'Historical shift30 task/sign/index mismatch')
        h.match_point(ident['point'],center['identity']['point'],'Historical shift30 nominal point mismatch')
        point=np.asarray(ident['point'],float).copy();point[149]+=sign*2.
        h.match_point(result['params'],point,'Historical shift30 endpoint point mismatch')
        common=set(ident['spec']) & set(center['identity']['spec'])-{'label'}
        h.require(all(ident['spec'][k]==center['identity']['spec'][k] for k in common),'Historical shared numerical-family mismatch')
        source=PurePosixPath(result['source'])
        h.require(source.name=='fwd_default.fsp' and any(p.startswith(f'task_{task:02d}_') for p in source.parts),
                  'Historical forward source path mismatch')
        rows.append(row)
    h.match_point(oldcenter['identity']['point'],center['identity']['point'],'Historical center point mismatch')
    with zipfile.ZipFile(archive) as z:
        arrays=[]
        for name in ('base_ports.npz','task_09_ports.npz','task_10_ports.npz'):
            raw=z.read(name);h.require(h.sha(raw)==info['members'][name],'Historical NPZ member hash mismatch')
            arrays.append(h.load_npz(raw))
    for row,a in zip([oldcenter]+rows,arrays):
        h.check_grid(a,baseline,center['identity']['spec']['n_wl_points']);verify_scores(row,a,score)
    return dict(available=True,half_step_nm=2.,source_roles=info,
        qualification='Center fingerprints an export; FD endpoints have separately pinned recovery creators. Code-map equality and complete original-solve provenance are not claimed.'),arrays


def verify_scores(row,arrays,score):
    measured=score(arrays)
    for key,value in measured.items():
        supplied=row['result'].get(key)
        h.require(isinstance(supplied,(int,float)) and not isinstance(supplied,bool) and np.isfinite(supplied)
                  and abs(float(value)-supplied)<=1e-12,'Saved scalar not reproduced: '+key)


def selected_peak(spectrum,wavelengths_m,eng):
    wl=np.asarray(wavelengths_m).ravel()*1e9
    wavelength,transmission,width=eng.measure_peak(wl,np.asarray(spectrum))
    h.require(transmission>=eng.DEAD_T_FLOOR and width is not None,'Dead/off-window selected resonance')
    index=int(np.argmin(abs(wl-wavelength)))
    return h.peak3(spectrum,index)


def pair_report(plus,minus,center,step,selector=None):
    select=selector or h.peak3
    p,m,c=select(plus),select(minus),select(center)
    spectral=(plus-minus)/(2*step);terms=h.terms(p[1],m[1],c[1],step)
    projection=float(c[2]@spectral[c[0]-1:c[0]+2])
    return dict(half_step_nm=step,plus_peak3=p[1],minus_peak3=m[1],center_peak3=c[1],
        peak_indices_plus_minus=[p[0],m[0]],center_peak_index=c[0],center_peak3_weights=c[2].tolist(),
        **terms,exact_center_peak3_jacobian_projection=projection,
        moving_peak_minus_projection=terms['central_derivative']-projection,
        spectral_fd_max_abs=float(np.max(abs(spectral))),spectral_fd_rms=float(np.sqrt(np.mean(spectral**2)))),spectral


def difference(a,b):
    delta=np.asarray(a)-np.asarray(b)
    return dict(max_abs=float(np.max(abs(delta))),rms=float(np.sqrt(np.mean(abs(delta)**2))))


def analyze(evidence,pins,repo):
    evidence=Path(evidence);sys.path.insert(0,str(Path(repo).resolve()))
    from runners.lumopt2_design import recover_te_s2 as recovery
    h.require(pins.get('schema')==1 and pins.get('existing_files')==EXISTING and pins.get('center_sha256')==CENTER_SHA and
              pins.get('marker_sha256')==MARKER_SHA,'Trusted existing evidence pin identity mismatch')
    marker,_=h.load_json(evidence/'s2_half_classes_first_review_original.json',MARKER_SHA)
    h.require(marker.get('schema')==1 and marker.get('family')==FAMILY and marker.get('stage')=='first' and
        marker.get('status')=='APPROVED_FOR_FIRSTWAVE_HALFSTEP_FD' and marker.get('allowed_tasks')==[0,1,2,3] and
        marker.get('task_map')==TASK_MAP and marker.get('center_sha256')==CENTER_SHA and marker.get('runner_sha256')==RUNNER_SHA,
        'Exact first-wave marker scope mismatch')
    h.require(sha(recovery.eng.__file__)==marker['code']['lumopt2_design.py'] and
              sha(recovery.__file__)==marker['code']['recover_te_s2.py'],'Local scoring source differs from frozen runtime')
    originals=[];oldarrays=[];provenance={}
    for task in range(3):
        name=f's2_fdstep_task{task:02d}_original.json';row,digest=h.load_json(evidence/name,EXISTING[name])
        npz=f's2_fdstep_task{task:02d}_ports.npz';raw=(evidence/npz).read_bytes()
        h.require(h.sha(raw)==EXISTING[npz],'Existing half-step NPZ pin mismatch')
        a=h.load_npz(raw);originals.append(row);oldarrays.append(a);provenance[name]=digest;provenance[npz]=h.sha(raw)
    center=originals[0];baseline=oldarrays[0];ident,result=center['identity'],center['result']
    h.require(ident['task']==0 and ident['index']==29 and ident['sign']==0 and result['task']==0 and result['sign']==0 and result['delta_nm']==0,'Pinned center metadata mismatch')
    h.match_point(result['params'],ident['point'],'Pinned center point mismatch')
    for task,sign in ((1,1),(2,-1)):h.check_half(originals[task],center,task,sign,CENTER_SHA)
    new=[];rows=[]
    for task in range(4):
        name=f's2_half_classes_task{task:02d}_original.json';row,digest=h.load_json(evidence/name)
        npz=f's2_half_classes_task{task:02d}_ports.npz';raw=(evidence/npz).read_bytes()
        validate_new(row,task,center,marker,h.sha(raw));a=h.load_npz(raw);rows.append(row);new.append(a)
        provenance[name]=digest;provenance[npz]=h.sha(raw)
    for row,a in list(zip(originals,oldarrays))+list(zip(rows,new)):
        h.check_grid(a,baseline,ident['spec']['n_wl_points']);verify_scores(row,a,recovery.score)
    oldinfo,oldshift=old_shift_inputs(pins,center,baseline,recovery.score)
    report=dict(schema=1,diagnostic_status='REVIEW_REQUIRED_CALIBRATION_FAILED_UNCHANGED',
        no_automatic_later_approval=True,provenance=dict(marker_sha256=MARKER_SHA,center_sha256=CENTER_SHA,
            runner_sha256=RUNNER_SHA,observed_files=provenance,old_shift30=oldinfo),observables={},
        limitations=['No coefficient fit or later-stage marker creation.',
            'Two corr30 pairs show observed repeat differences, not a noise sigma or statistical bound.',
            'Two shift30 steps do not establish asymptotic order or exact infinitesimal derivative.',
            'Fixed-center Jacobian and moving-peak FD may differ at finite step.',
            'Saved scalar tolerance1e-12 is arithmetic verification, not a scientific gate.',
            'Source paths/H5 relationships are checked; this offline tool does not open remote FSP/H5 or verify remote solver logs.'])
    spectra={}
    select=lambda spectrum:selected_peak(spectrum,baseline['p2_lambda'],recovery.eng)
    for metric,transform in h.METRICS.items():
        f0=transform(baseline);cp,cm=map(transform,new[:2]);op,om=map(transform,oldarrays[1:])
        newcorr,dnew=pair_report(cp,cm,f0,.5,select);oldcorr,dold=pair_report(op,om,f0,.5,select)
        sp,sm=map(transform,new[2:]);shift,dshift=pair_report(sp,sm,f0,.5,select)
        data=dict(corr30_repeat=dict(previous=oldcorr,new=newcorr,
            central_derivative_difference=newcorr['central_derivative']-oldcorr['central_derivative'],
            even_difference=newcorr['even']-oldcorr['even'],
            plus_endpoint_peak_difference=newcorr['plus_peak3']-oldcorr['plus_peak3'],
            minus_endpoint_peak_difference=newcorr['minus_peak3']-oldcorr['minus_peak3'],
            plus_spectral_difference=difference(cp,op),minus_spectral_difference=difference(cm,om),
            spectral_fd_difference=difference(dnew,dold)),shift30_half_step=shift)
        spectra[metric]=dict(corr30_new_spectral_fd=dnew.tolist(),corr30_previous_spectral_fd=dold.tolist(),shift30_half_spectral_fd=dshift.tolist())
        if oldshift is not None:
            oldbase,plus,minus=map(transform,oldshift)
            large,dlarge=pair_report(plus,minus,oldbase,2.,select)
            repeated,_=pair_report(plus,minus,f0,2.,select)
            data['shift30_old_h2']=dict(native_old_center=large,repeated_center_even_terms=repeated,
                hhalf_minus_h2_derivative=shift['central_derivative']-large['central_derivative'],
                spectral_fd_difference=difference(dshift,dlarge),center_peak_difference=select(f0)[1]-select(oldbase)[1])
            spectra[metric]['shift30_h2_spectral_fd']=dlarge.tolist()
        else:data['shift30_old_h2']=oldinfo
        report['observables'][metric]=data
    return report,spectra


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',required=True);parser.add_argument('--repo',required=True)
    parser.add_argument('--prepare-pins',action='store_true');parser.add_argument('--old-spectra');parser.add_argument('--old-watch-dir')
    parser.add_argument('--pins');parser.add_argument('--pins-sha256');parser.add_argument('--output',required=True);parser.add_argument('--spectra-output')
    args=parser.parse_args()
    if args.prepare_pins:write(args.output,build_pins(args.evidence_dir,args.old_spectra,args.old_watch_dir))
    else:
        if not args.pins or not args.pins_sha256 or not args.spectra_output:parser.error('--pins, --pins-sha256, --spectra-output required')
        pins,_=h.load_json(args.pins,args.pins_sha256);report,spectra=analyze(args.evidence_dir,pins,args.repo)
        report['pins_sha256']=args.pins_sha256;write(args.spectra_output,spectra)
        report['spectra_file']=str(Path(args.spectra_output).resolve());report['spectra_sha256']=sha(args.spectra_output)
        write(args.output,report)
