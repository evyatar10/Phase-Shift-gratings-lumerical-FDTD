"""Strict offline four-step comparison; no solver/imported project/approval.

Inputs: --evidence-dir DIR --old-spectra ZIP --pins JSON --output NEW.json.
DIR: s2_fdstep_task00/01/02_original.json + *_ports.npz,
     s2_corr30_convergence_task00..03_original.json + *_ports.npz,
     corr30_convergence_parent_review.json.
Use pins prepared from preserved existing evidence, not newly computed hashes
of an unverified replacement. Output uses exclusive creation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

TASK_MAP=[[1.,1],[1.,-1],[.25,1],[.25,-1]]
METRICS={'peak3_tout':lambda a:np.abs(a['p2_T_out'].ravel()),
         'peak3_modal':lambda a:np.abs(a['p2_S'].ravel())**2}
GRIDS=('p1_lambda','p1_f','p2_lambda','p2_f')


class EvidenceError(RuntimeError):
    pass


def require(ok,message):
    if not ok:
        raise EvidenceError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic_sha(row):
    return sha(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8'))


def load_json(path,expected=None):
    raw=Path(path).read_bytes()
    require(expected is None or sha(raw)==expected,'Raw checkpoint/marker hash mismatch: '+str(path))
    row=json.loads(raw)
    require(isinstance(row,dict),'Expected JSON object')
    return row,sha(raw)


def load_npz(raw):
    with np.load(io.BytesIO(raw),allow_pickle=False) as source:
        arrays={k:source[k].copy() for k in source.files}
    for k in GRIDS+('p2_T_out','p2_S'):
        require(k in arrays and np.isfinite(arrays[k]).all(),'Missing/nonfinite array '+k)
    return arrays


def peak3(t):
    t=np.asarray(t,dtype=float).ravel()
    require(len(t)>=3 and np.isfinite(t).all(),'Invalid spectrum')
    k=int(np.argmax(t))
    require(0<k<len(t)-1,'Peak on edge')
    b=t[k+1]-2*t[k]+t[k-1];d=t[k+1]-t[k-1]
    require(b<0,'Peak not concave')
    r=-d/(2*b)
    require(abs(r)<=.5+1e-9,'Peak vertex outside cell')
    w=np.array([r*(r-1)/2,1-r*r,r*(r+1)/2])
    return k,float(t[k]-d*d/(8*b)),w


def terms(plus,minus,center,h):
    require(np.isfinite([plus,minus,center,h]).all() and h>0,'Invalid scalar difference')
    odd=(plus-minus)/2;even=(plus+minus-2*center)/2
    return dict(odd=float(odd),even=float(even),central_derivative=float(odd/h),
                second_difference=float(2*even/(h*h)))


def match_point(point,expected,message):
    a=np.asarray(point,dtype=float);b=np.asarray(expected,dtype=float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.array_equal(a,b),message)


def spec_equal(a,b):
    a=dict(a);b=dict(b);a.pop('label',None);b.pop('label',None)
    return a==b


def check_grid(arrays,baseline,n):
    for key in GRIDS:
        require(arrays[key].shape==baseline[key].shape and
                np.array_equal(arrays[key],baseline[key]),'Full grid mismatch: '+key)
        require(arrays[key].size==n and np.all(arrays[key]>0),'Invalid grid size/value: '+key)
    require(arrays['p2_T_out'].size==n and arrays['p2_S'].size==n,'Spectrum length mismatch')


def reproduce(row,arrays):
    result=row['result']
    for metric,transform in METRICS.items():
        measured=peak3(transform(arrays))[1]
        supplied=result.get(metric)
        require(isinstance(supplied,(int,float)) and not isinstance(supplied,bool)
                and np.isfinite(supplied) and abs(measured-supplied)<=1e-12,
                'Saved peak3 score not reproduced: '+metric)


def check_half(row,center,task,sign,center_sha):
    ident,result=row['identity'],row['result']
    require(ident['task']==task and ident['index']==29 and ident['sign']==sign
            and ident['perturbation_nm']==.5,'Half-step identity mismatch')
    require(result['task']==task and result['index']==29 and result['sign']==sign
            and result['delta_nm']==sign*.5,'Half-step result/sign mismatch')
    require(ident['reviewed_center_sha256']==center_sha and
            result['reviewed_center_sha256']==center_sha,'Half-step center hash mismatch')
    require(ident['code']==center['identity']['code'] and
            ident['spec']==center['identity']['spec'],'Half-step source/spec mismatch')
    point=np.asarray(center['identity']['point'],float).copy();point[29]+=sign*.5
    match_point(ident['point'],point,'Half-step parameter delta mismatch')
    match_point(result['params'],point,'Half-step result parameters mismatch')


def check_convergence(row,center,task,marker,marker_sha,center_sha,pins,ports_sha):
    ident,result=row['identity'],row['result'];h,sign=TASK_MAP[task]
    require(ident['task']==task and ident['index']==29 and ident['sign']==sign
            and ident['perturbation_nm']==h and ident['task_map']==TASK_MAP,'Convergence task/step mismatch')
    require(result['task']==task and result['index']==29 and result['sign']==sign
            and result['perturbation_nm']==h and result['delta_nm']==h*sign,'Convergence result/step mismatch')
    require(ident['center_sha256']==center_sha and result['reviewed_center_sha256']==center_sha,
            'Convergence center hash mismatch')
    require(ident['center_checkpoint']==pins['center_remote_checkpoint'],'Center checkpoint path mismatch')
    require(ident['spec']['label']==marker['family'] and
            spec_equal(ident['spec'],center['identity']['spec']),'Complete convergence spec mismatch')
    code=ident['code']
    require(code==marker['code'],'Convergence exact source hashes differ from marker')
    require(all(code.get(k)==v for k,v in center['identity']['code'].items()),
            'Convergence source differs from frozen center')
    require(code.get('s2_corr30_convergence.py')==marker['runner_sha256'],'Runner source hash mismatch')
    review=ident['parent_review']
    expected=dict(path=pins['review_remote_path'],sha256=marker_sha,
        provenance_report=marker['provenance_report'],
        provenance_report_sha256=marker['provenance_report_sha256'],allowed_tasks=marker['allowed_tasks'])
    require(review==expected and result['parent_review']==expected,'Raw parent-review provenance mismatch')
    require(result['ports_sha256']==ports_sha,'Saved convergence NPZ hash mismatch')
    point=np.asarray(center['identity']['point'],float).copy();point[29]+=sign*h
    match_point(ident['point'],point,'Convergence parameter delta mismatch')
    match_point(result['params'],point,'Convergence result parameters mismatch')
    for name in ('source','h5','ports_npz'):
        require(isinstance(result.get(name),str) and result[name].startswith(marker['output_root']+'/'),
                'Convergence artifact outside declared output family')


def validated_inputs(evidence,archive,pins):
    require(pins.get('schema')==1,'Unknown pin schema')
    original=[];arrays=[];provenance={}
    for i in range(3):
        name=f's2_fdstep_task{i:02d}_original.json'
        row,digest=load_json(evidence/name,pins['existing_files'][name]);original.append(row)
        name_npz=f's2_fdstep_task{i:02d}_ports.npz';raw=(evidence/name_npz).read_bytes()
        require(sha(raw)==pins['existing_files'][name_npz],'Pinned existing NPZ mismatch: '+name_npz)
        arrays.append(load_npz(raw));provenance[name]=digest;provenance[name_npz]=sha(raw)
    center=original[0];center_sha=provenance['s2_fdstep_task00_original.json']
    require(center_sha==pins['center_sha256'],'Pinned nominal center mismatch')
    ident,result=center['identity'],center['result']
    require(ident['task']==0 and ident['index']==29 and ident['sign']==0
            and ident['perturbation_nm']==.5 and result['task']==0 and result['sign']==0
            and result['delta_nm']==0 and result['index']==29,'Nominal center identity mismatch')
    match_point(result['params'],ident['point'],'Nominal result point mismatch')
    require(ident['code']==pins['center_code'],'Pinned center code mismatch')
    for task,sign in ((1,1),(2,-1)):
        check_half(original[task],center,task,sign,center_sha)
    marker,marker_sha=load_json(evidence/'corr30_convergence_parent_review.json',pins['review_marker_sha256'])
    require(marker['schema']==1 and marker['center_sha256']==center_sha
            and marker['task_map']==TASK_MAP and marker['allowed_tasks']==[0,1,2,3]
            and marker['status']=='APPROVED_FOR_BOUNDED_FORWARD_STEP_DIAGNOSTIC',
            'Parent marker is not the exact bounded family approval')
    new=[];new_arrays=[]
    for task in range(4):
        name=f's2_corr30_convergence_task{task:02d}_original.json'
        row,digest=load_json(evidence/name)
        name_npz=f's2_corr30_convergence_task{task:02d}_ports.npz';raw=(evidence/name_npz).read_bytes()
        check_convergence(row,center,task,marker,marker_sha,center_sha,pins,sha(raw))
        new.append(row);new_arrays.append(load_npz(raw))
        provenance[name]=digest;provenance[name_npz]=sha(raw)
    archive_raw=archive.read_bytes()
    require(sha(archive_raw)==pins['old_archive_sha256'],'Original spectra archive hash mismatch')
    old=[]
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as z:
        require(z.testzip() is None,'Original archive failed integrity')
        for name in ('base_ports.npz','task_03_ports.npz','task_04_ports.npz'):
            raw=z.read(name)
            require(sha(raw)==pins['old_members'][name],'Original archive member hash mismatch')
            old.append(load_npz(raw))
    n=ident['spec']['n_wl_points']
    for a in arrays+new_arrays+old:
        check_grid(a,old[0],n)
    for row,a in list(zip(original,arrays))+list(zip(new,new_arrays)):
        reproduce(row,a)
    for row,a in zip(pins['old_checkpoint_rows'],old):
        reproduce(row,a)
    old_center,old_plus,old_minus=pins['old_checkpoint_rows']
    bindings=pins.get('old_source_bindings',{})
    exporter=bindings.get('center_export',{})
    require(exporter.get('role')=='stored_forward_export_no_solve'
            and old_center['identity']['code']==exporter.get('code')
            and semantic_sha(old_center)==exporter.get('semantic_sha256'),
            'Old center exporter source/content binding mismatch')
    match_point(old_center['identity']['point'],ident['point'],'Old/new nominal parameter mismatch')
    for row,task,sign in ((old_plus,3,1),(old_minus,4,-1)):
        old_i,old_r=row['identity'],row['result']
        require(old_i['task']==task and old_i['perturbation']==2. and old_r['index']==29
                and old_r['sign']==sign,'Old two-nm label/sign mismatch')
        match_point(old_i['point'],ident['point'],'Old two-nm base point mismatch')
        p=np.asarray(ident['point'],float).copy();p[29]+=sign*2.
        match_point(old_r['params'],p,'Old two-nm endpoint mismatch')
        endpoint=bindings.get('fd_endpoints',{}).get(str(task),{})
        require(endpoint.get('role')=='new_forward_fd_leg'
                and old_i['code']==endpoint.get('code')
                and semantic_sha(row)==endpoint.get('semantic_sha256'),
                'Old two-nm FD source/content binding mismatch')
        common=set(old_i['spec']) & set(ident['spec'])-{'label'}
        require(all(old_i['spec'][k]==ident['spec'][k] for k in common),'Old numerical-family mismatch')
    provenance['old_archive']=sha(archive_raw);provenance['review_marker']=marker_sha
    provenance['old_source_roles']=bindings
    return arrays,new_arrays,old,provenance


def halving_report(values):
    hs=[2.,1.,.5,.25]
    d=[float(values[h]) for h in hs]
    differences=[d[i]-d[i+1] for i in range(3)]
    ratios=[]
    for i in range(2):
        ratio=differences[i]/differences[i+1] if differences[i+1]!=0 else None
        if ratio is not None and not np.isfinite(ratio):
            ratio=None
        ratios.append(dict(steps_nm=hs[i:i+3],signed_ratio=ratio,
            absolute_ratio=abs(ratio) if ratio is not None else None,
            conditional_implied_order=(float(np.log2(abs(ratio))) if ratio not in (None,0.) else None)))
    rich=[dict(h_large=hs[i],h_small=hs[i+1],value=(4*d[i+1]-d[i])/3)
          for i in range(3)]
    return dict(halving_differences=differences,observed_ratios=ratios,
        conditional_Richardson_assuming_O_h2=rich,
        caveat='Ratios are observations, not established convergence order. Richardson assumes an asymptotic h^2 error expansion; cancellation, center mismatch and unresolved solver error can invalidate it.')


def analyze(evidence,archive,pins):
    a,new,old,provenance=validated_inputs(Path(evidence),Path(archive),pins)
    output=dict(steps_nm=[2.,1.,.5,.25],provenance=provenance,observables={},
        grids={key:old[0][key].tolist() for key in GRIDS},
        verdict='DIAGNOSTIC_ONLY_OVERALL_CALIBRATION_REMAINS_FAILED',
        limitations=['No noise bounds inferred from a single repeat or halving differences.',
            'No global calibration, new C coefficient, spatial-width validation or optimization claim.',
            'Saved Tout/modal peak3 reproduction tolerance 1e-12 is an arithmetic check, not a scientific tolerance.',
            'Old/new center values may differ despite identical nominal parameters; even/curvature terms depend on the chosen center.',
            'Fixed-center peak3 Jacobian projection and moving-peak FD need not agree at finite h.'])
    output['limitations'].append('The h2 nominal code map fingerprints a read-only export, not creation of its surviving FSP. h2 endpoint generation has separately pinned later recovery/engine identities; full parameter/spec/grid/score agreement is checked, but code equality or complete original-solve provenance is not claimed.')
    for metric,transform in METRICS.items():
        f0new,f0old=transform(a[0]),transform(old[0])
        cnew,cold=peak3(f0new),peak3(f0old)
        pairs={2.:(transform(old[1]),transform(old[2])),1.:(transform(new[0]),transform(new[1])),
               .5:(transform(a[1]),transform(a[2])),.25:(transform(new[2]),transform(new[3]))}
        reports={};derivatives={};curv_new={};curv_old={};spectral_derivatives={}
        for h,(plus,minus) in pairs.items():
            pp,pm=peak3(plus),peak3(minus)
            spectral_fd=(plus-minus)/(2*h)
            spectral_derivatives[h]=spectral_fd
            scalar_new=terms(pp[1],pm[1],cnew[1],h)
            scalar_old=terms(pp[1],pm[1],cold[1],h)
            derivatives[h]=scalar_new['central_derivative']
            curv_new[h]=scalar_new['second_difference'];curv_old[h]=scalar_old['second_difference']
            projections={}
            for label,c in (('new',cnew),('old',cold)):
                k,_,w=c;projected=float(w@spectral_fd[k-1:k+2])
                projections[label]=dict(center_index=k,weights=w.tolist(),
                    center_sample_fd=float(spectral_fd[k]),exact_center_peak3_Jacobian_projection=projected,
                    moving_peak_minus_projection=scalar_new['central_derivative']-projected)
            reports[str(h)]=dict(native_center='old' if h==2. else 'new',
                scalar_with_new_center=scalar_new,scalar_with_old_center=scalar_old,
                peak_indices_plus_minus=[pp[0],pm[0]],center_projections=projections,
                full_spectral=dict(plus_minus= (plus-minus).tolist(),central_derivative=spectral_fd.tolist(),
                    odd=((plus-minus)/2).tolist(),even_with_new_center=((plus+minus-2*f0new)/2).tolist(),
                    even_with_old_center=((plus+minus-2*f0old)/2).tolist(),
                    fd_max_abs=float(np.max(np.abs(spectral_fd)))))
        hs=[2.,1.,.5,.25];spectral_halving=[]
        for h_large,h_small in zip(hs[:-1],hs[1:]):
            difference=spectral_derivatives[h_large]-spectral_derivatives[h_small]
            spectral_halving.append(dict(h_large=h_large,h_small=h_small,
                difference=difference.tolist(),max_abs=float(np.max(np.abs(difference))),
                l2_norm=float(np.linalg.norm(difference))))
        output['observables'][metric]=dict(center_values=dict(new=cnew[1],old=cold[1],
            repeat_delta=cnew[1]-cold[1]),steps=reports,
            derivative_halving=halving_report(derivatives),
            curvature_halving_with_new_center=halving_report(curv_new),
            curvature_halving_with_old_center=halving_report(curv_old),
            full_spectral_derivative_halving=spectral_halving,
            full_spectral_center_difference=(f0new-f0old).tolist())
    return output


def write_exclusive(path,result):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(result,stream,allow_nan=False,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-dir','old-spectra','pins','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Output already exists')
    pins,pins_digest=load_json(args.pins)
    result=analyze(args.evidence_dir,args.old_spectra,pins)
    result['provenance']['pins_json_sha256']=pins_digest
    write_exclusive(args.output,result)
    print('Validated four steps; diagnostic-only report:',args.output)
