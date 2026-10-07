"""Staged exact-center/shift30 +/-0.0625nm fixed100/150ps forwards; no fit."""
import dataclasses,hashlib,json,os,re,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(os.environ.get('S2_TIME_REPO',Path.cwd())).resolve()))
import numpy as np
from runners.lumopt2_design import s2_shift30_convergence as prior
eng,val,recovery,diag=prior.eng,prior.val,prior.recovery,prior.diag
IS_HELPER=True
N_TASKS=6
LABEL='s2_shift30_temporal_20261007'
SPEC=dataclasses.replace(prior.SPEC,label=LABEL)
TASK_MAP=[(t,s) for t in (100.,150.) for s in (0,1,-1)]
CENTER_SHA=prior.CENTER_SHA256
CENTER_PORT_SHA='4c3c526b51973a2c8889608c69958b6d60e4dc27ddcf638c9932598629a035e4'
PRIOR_MARKER_SHA='50ef62531e338f57b5cbc78d05905a2cd5d805297407a4ba0913cedf3a5404df'
PRIOR_TASK_SHA={0:'5de37e6a2709ee2214bf64fea1a17421704c9e23f6b442edb7d4d32ec1c4622b',1:'82a537302e7e89262d95a67966c948a2f57880c22853c39309bdb6893b39df8e',2:'0711bf7b9cc7776ba9e2617b469d915ac624aca8e1efacf00043b340b4d1b2f2',3:'4529bc20329c5f9de15c1564268a88ec5fa16376f205779dc0a7d73154d9f2aa'}
STATUS={100.:'APPROVED_FOR_FIXED100PS_TRIPLET_ONLY',150.:'APPROVED_AFTER_FIXED100PS_REVIEW_FOR_FIXED150PS_TRIPLET_ONLY'}
TIMING_PROPS=('simulation time','use early shutoff','auto shutoff min','use divergence checking','auto shutoff max','down sample time','max source time signal length','dt')
GLOBAL_SOURCE_PROPS=('pulselength','offset','frequency','bandwidth','source type','set time domain','set frequency','wavelength start','wavelength stop','optimize for short pulse')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def semantic(v):return hashlib.sha256(json.dumps(plain(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def need(ok,message):
    if not ok:raise RuntimeError(message)
def envpath(name):
    value=os.environ.get(name);need(bool(value),name+' required');p=Path(value);need(p.is_absolute(),name+' must be absolute');return p.resolve()
def task_params(task):
    if isinstance(task,bool) or not isinstance(task,(int,np.integer)) or task not in range(6):raise ValueError('Task must be integer0..5')
    duration,sign=TASK_MAP[task];p=val.te_point(SPEC).copy();index=29 if sign==0 else 149
    if sign:p[index]+=sign*.0625
    need(prior.INDEX==149 and prior.CENTER_INDEX==29,'Live layout changed')
    b=np.asarray(eng.param_bounds(SPEC));need(np.all((p>=b[:,0])&(p<=b[:,1])),'Point outside bounds')
    return p,index,duration,sign
def code():return {**prior.dependency_hashes(),Path(__file__).name:sha(__file__)}
def vendor_sources():
    # No CAD/import required. Fingerprint the exact configured API tree before
    # markers or checkpoint reuse, using relative keys portable across hosts.
    import config
    api=Path(config.LUMAPI_PATH).resolve().parent;tree=api/'lumopt2'
    need(tree.is_dir() and (api/'lumapi.py').is_file(),'Configured vendor API tree unavailable')
    paths=sorted(tree.rglob('*.py'))+[api/'lumapi.py']
    if (api/'lumslurm.py').is_file():paths.append(api/'lumslurm.py')
    return {str(p.relative_to(api)).replace('\\','/'):sha(p) for p in paths}
def root_path():
    root=envpath('S2_TIME_OUTPUT_ROOT');need(root.name==LABEL,'Unique temporal output label required')
    for old in (envpath('S2_SHIFT_CENTER_CHECKPOINT').parent,envpath('S2_TIME_PRIOR_ROOT')):
        need(not root.is_relative_to(old) and not old.is_relative_to(root),'Output overlaps existing evidence')
    return root
def load_center():
    need(dataclasses.asdict(SPEC)==dataclasses.asdict(dataclasses.replace(prior.SPEC,label=LABEL)),'Frozen numerical spec changed')
    center=prior.load_center();need(center['sha256']==CENTER_SHA,'Exact center SHA mismatch')
    need(sha(center['result']['ports_npz'])==CENTER_PORT_SHA,'Exact center port NPZ byte pin mismatch');return center
def baseline_bindings(center):
    path=envpath('S2_TIME_PRIOR_MARKER');need(sha(path)==PRIOR_MARKER_SHA,'Exact shift30 prior marker required')
    m=json.loads(path.read_text());need(m['code']==prior.dependency_hashes() and m['status']==prior.APPROVAL_STATUS and m['allowed_tasks']==[0,1,2,3] and m['center_sha256']==CENTER_SHA,'Frozen prior family mismatch')
    root=envpath('S2_TIME_PRIOR_ROOT');need(str(root)==m['output_root'],'Exact prior root mismatch')
    review=dict(path=str(path),sha256=PRIOR_MARKER_SHA,first_wave=m['first_wave'],lead_verdict=m['lead_verdict'],allowed_tasks=[0,1,2,3])
    rows={}
    for t in range(4):
        cp=root/f'task_{t:02d}.json';need(sha(cp)==PRIOR_TASK_SHA[t],'Exact completed171162 checkpoint SHA mismatch')
        r=recovery.read_checkpoint(cp,prior.identity(t,center,review));need(r is not None,'Completed prior checkpoint missing')
        prior.verify_result(r,t,root)
        rows[str(t)]=dict(checkpoint=str(cp),sha256=sha(cp),ports_sha256=r['ports_sha256'],source=r['source'],h5=r['h5'])
    return dict(marker=str(path),marker_sha256=PRIOR_MARKER_SHA,checkpoints=rows)
def scope(duration,root,center):
    tasks=[0,1,2] if duration==100. else [3,4,5]
    return dict(schema=1,family=LABEL,duration_ps=duration,center_sha256=CENTER_SHA,runner_sha256=sha(__file__),
        code=code(),vendor_api_sha256=vendor_sources(),spec=plain(dataclasses.asdict(SPEC)),task_map=plain(TASK_MAP),output_root=str(root),allowed_tasks=tasks,
        timing_changes={'simulation time':duration*1e-12,'use early shutoff':0},prior=baseline_bindings(center))
def identity(task,center,review):
    p,index,duration,sign=task_params(task)
    return dict(schema=1,task=int(task),spec=plain(dataclasses.asdict(SPEC)),point=p.tolist(),index=index,sign=sign,
        delta_nm=sign*.0625,duration_ps=duration,task_map=plain(TASK_MAP),center_checkpoint=center['checkpoint'],
        center_sha256=CENTER_SHA,code=code(),vendor_api_sha256=vendor_sources(),parent_review=review,timing_changes={'simulation time':duration*1e-12,'use early shutoff':0})
def first_stage_bindings(root,center):
    path=envpath('S2_TIME_FIRST_MARKER');m=json.loads(path.read_text());expected=scope(100.,root,center)
    need(all(m.get(k)==v for k,v in expected.items()) and m.get('status')==STATUS[100.],'First100ps marker scope mismatch')
    need(isinstance(m.get('lead_verdict'),str) and m['lead_verdict'].strip() not in ('','NOT_REVIEWED'),'First marker lead review absent')
    review=dict(path=str(path),sha256=sha(path),duration_ps=100.,lead_verdict=m['lead_verdict'],allowed_tasks=[0,1,2],first_stage=None)
    rows={}
    for t in range(3):
        cp=root/f'task_{t:02d}.json';r=recovery.read_checkpoint(cp,identity(t,center,review));need(r is not None,'Completed100ps triplet missing')
        verify_result(r,t,root)
        rows[str(t)]=dict(checkpoint=str(cp),sha256=sha(cp),ports_sha256=r['ports_sha256'],fsp_sha256=r['fsp_sha256'],h5_sha256=r['h5_sha256'],termination=r['termination'])
    return dict(marker=str(path),marker_sha256=sha(path),checkpoints=rows)
def review_template(duration,root,center):
    need(duration in (100.,150.),'Duration not in bounded family')
    m=scope(duration,root,center);m.update(status='NOT_REVIEWED',allowed_tasks=[],lead_verdict='NOT_REVIEWED',
        first_stage=first_stage_bindings(root,center) if duration==150. else None);return m
def reviewed(task,root,center):
    duration=task_params(task)[2];path=envpath('S2_TIME_FIRST_MARKER' if duration==100. else 'S2_TIME_LATER_MARKER')
    if duration==150.:need(path!=envpath('S2_TIME_FIRST_MARKER'),'150ps requires separate immutable review marker')
    m=json.loads(path.read_text());expected=scope(duration,root,center)
    need(all(m.get(k)==v for k,v in expected.items()) and m.get('status')==STATUS[duration],'Explicit duration review scope mismatch')
    need(isinstance(m.get('lead_verdict'),str) and m['lead_verdict'].strip() not in ('','NOT_REVIEWED'),'Substantive lead verdict required')
    first=first_stage_bindings(root,center) if duration==150. else None
    need(m.get('first_stage')==first,'Exact completed100ps provenance mismatch')
    return dict(path=str(path),sha256=sha(path),duration_ps=duration,lead_verdict=m['lead_verdict'],allowed_tasks=expected['allowed_tasks'],first_stage=first)

def snapshot(f,p):
    def read(obj,key):return plain(f.getnamed(obj,key))
    geometry={key:read(*key.rsplit('::',1)) for key in eng.make_func(SPEC)(p)}
    return dict(geometry=geometry,mesh={a:plain(f.getresult('FDTD',a)) for a in ('x','y','z')},
        timing={k:read('FDTD',k) for k in TIMING_PROPS},source={k:plain(f.getglobalsource(k)) for k in GLOBAL_SOURCE_PROPS},
        ports={obj:{k:read(obj,k) for k in ('x','y','z','x span','y span','z span','phase','injection axis','direction','enabled')} for obj in ('FDTD::ports::Port_1','FDTD::ports::Port_2')},
        refinement={k:read('FDTD',k) for k in ('mesh refinement','meshing refinement')},
        optimization_dft_enabled=read('optimization_dft','enabled'),
        monitor={k:read('FDTD::ports',k) for k in ('monitor frequency points','source port','source mode')})
def controls(f,p,duration):
    before=snapshot(f,p)
    need(all(abs(float(np.asarray(before['geometry'][key]).squeeze())-float(value))<1e-18 for key,value in eng.make_func(SPEC)(p).items()),'Generated geometry does not match requested exact point')
    need(before['timing']['simulation time']==2000e-12 and before['timing']['use early shutoff']==1 and before['timing']['auto shutoff min']==1e-7,'Unexpected baseline timing; refuse intervention')
    need(before['timing']['use divergence checking']==1,'Divergence safety checking must stay enabled')
    f.setnamed('FDTD','use early shutoff',0);f.setnamed('FDTD','simulation time',duration*1e-12)
    f.setnamed('optimization_dft','enabled',0)
    need(before['optimization_dft_enabled']==1,'Unexpected initial optimization DFT state')
    after=snapshot(f,p);expected=json.loads(json.dumps(before));expected['timing']['simulation time']=duration*1e-12;expected['timing']['use early shutoff']=0;expected['optimization_dft_enabled']=0
    need(after==expected,'Timing intervention changed frozen geometry/mesh/source/ports/numerics')
    dt=float(after['timing']['dt']);need(np.isfinite(dt) and dt>0,'Recorded solver dt unavailable')
    pulse=float(after['source']['pulselength']);offset=float(after['source']['offset'])
    need(np.isfinite([pulse,offset]).all() and pulse>0 and offset>=2*pulse and offset+10*pulse<duration*1e-12,'Source pulse does not fit target window safely')
    return dict(before=before,after=after,before_sha256=semantic(before),after_sha256=semantic(after),
        changed_properties={'FDTD::simulation time':[before['timing']['simulation time'],after['timing']['simulation time']],
            'FDTD::use early shutoff':[1,0],'optimization_dft::enabled':[1,0]},
        dt_s=dt,source_pulse_length_s=pulse,source_offset_s=offset,pulse_fit_status='CAD_PARAMETER_CHECK_ONLY_NOT_TEMPORAL_CONVERGENCE')
def persisted_check(expected,actual):
    need(set(actual)==set(expected),'Persisted snapshot key mismatch')
    need(all(actual[k]==expected[k] for k in expected if k!='geometry'),'Persisted timing/mesh/source/ports/numerics changed')
    need(set(actual['geometry'])==set(expected['geometry']),'Persisted geometry keys changed')
    differences={}
    for key,left in expected['geometry'].items():
        a=float(np.asarray(left).squeeze());b=float(np.asarray(actual['geometry'][key]).squeeze());delta=b-a
        if delta:
            need(np.isfinite([a,b]).all() and abs(delta)<=1e-20 and abs(delta)<=64*np.finfo(float).eps*max(abs(a),abs(b)),
                'Material persisted geometry difference: '+key)
            differences[key]=dict(before_m=a,reloaded_m=b,delta_m=delta)
    return dict(mapped_geometry_roundoff_differences=differences,max_abs_geometry_delta_m=max([abs(d['delta_m']) for d in differences.values()]+[0.]),
        all_other_snapshot_sections_exact=True,qualification='Bounded FSP geometry serialization roundoff retained; no physical convergence claim')
def termination(source,duration,dt,status):
    log=Path(source).with_name(Path(source).stem+'_p0.log');text=log.read_text(errors='replace')
    m=re.search(r'Completed (\d+) iterations, or ([\deE.+-]+)s of Simulation Time',text)
    need(m is not None and 'Simulation completed successfully' in text,'Successful physical-time termination record missing')
    if isinstance(status,dict):status=status.get('status')
    need(status is not None and 'Early termination of simulation' not in text and float(np.asarray(status).squeeze())==1.,'Did not complete fixed-time window')
    elapsed=float(m[2]);need(abs(elapsed-duration*1e-12)<=dt,'Actual duration differs by more than recorded one timestep')
    return dict(log=str(log),log_sha256=sha(log),iterations=int(m[1]),physical_time_s=elapsed,requested_time_s=duration*1e-12,
        recorded_dt_s=dt,delta_s=elapsed-duration*1e-12,status=1,reason='completed_full_simulation_time')
def verify_result(r,task,root):
    p,index,duration,sign=task_params(task)
    expected=dict(task=task,index=index,duration_ps=duration,sign=sign,delta_nm=sign*.0625,reviewed_center_sha256=CENTER_SHA,
        existing_overall_gradient_gate='FAILED_UNCHANGED',comparison_status='TEMPORAL_DIAGNOSTIC_REVIEW_REQUIRED_NO_PHYSICS_PASS')
    need(all(r.get(k)==v for k,v in expected.items()) and np.array_equal(np.asarray(r['params']),p),'Full temporal result identity mismatch')
    need(r.get('vendor_api_sha256')==vendor_sources(),'Completed vendor API identity changed')
    diag.verify_artifacts(r)
    source=Path(r['source']).resolve();ports=Path(r['ports_npz']).resolve()
    need(ports.parent.is_relative_to(root.resolve()) and ports.parent.name.startswith(f'task_{task:02d}_') and source.is_relative_to(ports.parent) and source.name=='fwd_default.fsp' and Path(r['h5']).resolve()==source.with_suffix('')/(source.stem+'_output.h5'),'Artifact path escaped unique attempt')
    for key,digest in (('source','fsp_sha256'),('h5','h5_sha256'),('ports_npz','ports_sha256')):need(sha(r[key])==r[digest],'Completed artifact SHA mismatch: '+key)
    t=termination(source,duration,r['controls']['dt_s'],1);need(t==r['termination'],'Termination log identity mismatch')
    c=r['controls'];need(c['before_sha256']==semantic(c['before']) and c['after_sha256']==semantic(c['after']),'Control snapshot fingerprint mismatch')
    expected_snapshot=json.loads(json.dumps(c['before']));expected_snapshot['timing']['simulation time']=duration*1e-12;expected_snapshot['timing']['use early shutoff']=0;expected_snapshot['optimization_dft_enabled']=0
    need(c['after']==expected_snapshot,'Frozen control snapshot changed')
    need(c['persisted_sha256']==semantic(c['persisted']) and persisted_check(c['after'],c['persisted'])==c['serialization'],'Persisted snapshot fingerprint mismatch')
    with np.load(ports,allow_pickle=False) as ar:arrays={k:ar[k] for k in ar.files}
    scores=recovery.score(arrays);need(all(abs(float(scores[k])-float(r[k]))<=1e-12 for k in diag.METRICS),'Saved scalar not reproduced')
    center=load_center();need(sha(center['result']['ports_npz'])==CENTER_PORT_SHA,'Exact center port NPZ byte pin mismatch')
    with np.load(center['result']['ports_npz'],allow_pickle=False) as baseline:
        for key in ('p1_f','p1_lambda','p2_f','p2_lambda'):need(arrays[key].size==SPEC.n_wl_points and np.isfinite(arrays[key]).all() and np.array_equal(arrays[key],baseline[key]),'Exact port grid mismatch')
def forward_only(task,root,center,review):
    p,index,duration,sign=task_params(task);attempt=Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_',dir=root))
    project,_=eng.make_project(SPEC,str(attempt),eng.import_lumopt2())
    try:
        project.set_fwd(p);f=project.fdtd_session.fdtd;control=controls(f,p,duration)
        configs=list(project.fom.config_map.config_map.values());need(len(configs)==1,'Exactly one forward configuration required')
        source=Path(project.fom.config_map.project_folder)/configs[0]['fwd_filename'];need(source.resolve().is_relative_to(attempt.resolve()),'New forward path escaped attempt')
        f.save(str(source));setup_sha=sha(source);project.fdtd_session.invalidate_loaded_state();f.load(str(source))
        persisted=snapshot(f,p);control.update(persisted=persisted,persisted_sha256=semantic(persisted),serialization=persisted_check(control['after'],persisted))
        project.fdtd_session.invalidate_loaded_state();project.run_forward()
        h5=source.with_suffix('')/(source.stem+'_output.h5');need(h5.is_file(),'Companion HDF5 required')
        f.load(str(source));t=termination(source,duration,control['dt_s'],f.getresult('FDTD','status'));arrays=recovery.port_arrays(f)
        # Exact baseline grid is read from preserved original center ports only.
        with np.load(center['result']['ports_npz'],allow_pickle=False) as ar:
            for key in ('p1_f','p1_lambda','p2_f','p2_lambda'):need(np.array_equal(arrays[key],ar[key]),'Temporal port grid changed: '+key)
        ports=attempt/'ports.npz'
        with ports.open('xb') as stream:np.savez(stream,**arrays)
        return dict(task=task,index=index,duration_ps=duration,sign=sign,delta_nm=sign*.0625,params=p.tolist(),source=str(source),h5=str(h5),
            ports_npz=str(ports),ports_sha256=sha(ports),fsp_sha256=sha(source),h5_sha256=sha(h5),setup_fsp_sha256=setup_sha,
            controls=control,termination=t,parent_review=review,reviewed_center_sha256=CENTER_SHA,**recovery.score(arrays),
            FDTD_version=plain(f.version()),vendor_api_sha256=vendor_sources(),
            comparison_status='TEMPORAL_DIAGNOSTIC_REVIEW_REQUIRED_NO_PHYSICS_PASS',existing_overall_gradient_gate='FAILED_UNCHANGED')
    finally:project.fdtd_session.fdtd.close()
def main(task_idx):
    task_params(task_idx);task=int(task_idx);root=root_path();center=load_center();review=reviewed(task,root,center);ident=identity(task,center,review)
    root.mkdir(parents=True,exist_ok=True);cp=root/f'task_{task:02d}.json';saved=recovery.read_checkpoint(cp,ident)
    if saved is not None:verify_result(saved,task,root);print('REUSED temporal checkpoint',cp);return saved
    need(os.environ.get('KEEP_H5')=='1','KEEP_H5=1 required before CAD')
    # A retained incomplete attempt is not discarded; make a separate new attempt.
    result=forward_only(task,root,center,review);verify_result(result,task,root);recovery.checkpoint(cp,ident,result);print('Temporal diagnostic checkpoint; review required',cp);return result
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',type=int);parser.add_argument('--review-template',type=float,choices=(100.,150.));a=parser.parse_args()
    if a.review_template is not None:print(json.dumps(review_template(a.review_template,root_path(),load_center()),indent=2))
    elif a.task is not None:main(a.task)
    else:parser.error('--task or --review-template required')
