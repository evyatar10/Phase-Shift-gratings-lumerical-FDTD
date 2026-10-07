"""Two explicitly reviewed150ps shift30 +/-0.125nm forward diagnostics; no optimizer."""
import copy,dataclasses,hashlib,json,os,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(os.environ.get('S2_STEP150_REPO',Path.cwd())).resolve()))
import numpy as np
from runners.lumopt2_design import s2_temporal_forward as base
eng,val,recovery,diag=base.eng,base.val,base.recovery,base.diag
IS_HELPER=True
N_TASKS=2
LABEL='s2_shift30_step150_20261007'
SPEC=dataclasses.replace(base.SPEC,label=LABEL)
TASK_MAP=((.125,1),(.125,-1))
INDEX=149
DURATION_PS=150.
BASE_SHA='555f3d13b42c494171e7658cc4263c588ce418e5d937da1497d690824b41d44a'
CENTER_SHA='a0ff55eacaa52e00a2aab17dcc43d0e5fff9a72566e15da371ff25279c172663'
PARENT_SHA={3:CENTER_SHA,4:'31ec8f2f72bfb1a5aa991240cec45c5fb515d18e82fc37d7cfac89703e75efe4',5:'a0e9d73a0f8f0162c9f7a843ca24eb4c4c29e444b1fca2b692ffd5002feb571d'}
PARENT_PORT_SHA={3:'a9a4b2e87379c645786ac8fc179852ad3bdec2a040c7fe2c8c1ae21342a5d90d',4:'a92bb781482ec2cd01e688fb3dc25f7e86f33a16e3fc7289f629ba4267caac90',5:'54694438904b08eaf9bdc7ef2c8a0a1b7c56183a39bdc3e1bcf6fa8309e4c688'}
PARENT_MARKER_SHA='6cc6d9fd8cab5419b436b433725fbb2cc88b49f538628f897929217940527f51'
PARENT_ROOT='/work/results/diagnose_te_s2_temporal/results/s2_shift30_temporal_20261007'
PARENT_MARKER='/work/data/s2_temporal150_review_20261007_1257.json'
DEFAULT_OUTPUT='/work/results/diagnose_te_s2_step150/results/'+LABEL
APPROVAL_STATUS='APPROVED_FOR_TWO_FIXED150PS_SHIFT30_STEP0125_FORWARDS_ONLY'
COMPARISON_STATUS='STEP150_DIAGNOSTIC_REVIEW_REQUIRED_NO_PHYSICS_PASS'
plain,semantic,sha,need=base.plain,base.semantic,base.sha,base.need
snapshot,controls,persisted_check,termination=base.snapshot,base.controls,base.persisted_check,base.termination

def envpath(name,default=None):
 v=os.environ.get(name,default);need(bool(v),name+' explicitly required');p=Path(v);need(p.is_absolute(),name+' must be absolute');return p.resolve()
def frozen_base():
 need(sha(base.__file__)==BASE_SHA,'Frozen temporal helper SHA mismatch')
 need(dataclasses.asdict(SPEC)==dataclasses.asdict(dataclasses.replace(base.SPEC,label=LABEL)),'Numerical spec changed')
def code():
 frozen_base();return {**base.code(),Path(__file__).name:sha(__file__)}
def task_params(task):
 if isinstance(task,bool) or not isinstance(task,(int,np.integer)) or task not in range(N_TASKS):raise ValueError('Task must be integer0..1')
 need(base.prior.INDEX==INDEX and base.prior.CENTER_INDEX==29,'Live parameter map changed')
 h,s=TASK_MAP[task];p=val.te_point(SPEC).copy();p[INDEX]+=h*s;b=np.asarray(eng.param_bounds(SPEC))
 need(np.all((p>=b[:,0])&(p<=b[:,1])),'Step outside bounds');return p,INDEX,DURATION_PS,s

def phase_compare(left,right):
 """Recorded cross-run Port_1 phase roundoff only; introduction cause unknown."""
 def get(x):
  v=x['ports']['FDTD::ports::Port_1']['phase'];need(np.asarray(v).shape==() and not isinstance(v,(bool,str)),'Scalar phase required');v=np.float64(v);need(np.isfinite(v),'Finite phase required');return v,int(v.view(np.uint64))
 a,ab=get(left);b,bb=get(right)
 if a==0 or b==0:need(ab==bb,'Zero phase must match bitwise');n=0
 else:
  order=lambda bits:(~bits)&((1<<64)-1) if bits&(1<<63) else bits|(1<<63)
  n=abs(order(ab)-order(bb));need(n<=2,'Port_1 cross-run phase exceeds2ULP')
 checked=copy.deepcopy(right);checked['ports']['FDTD::ports::Port_1']['phase']=left['ports']['FDTD::ports::Port_1']['phase']
 detail=persisted_check(left,checked)
 return dict(property='/ports/FDTD::ports::Port_1/phase',left_phase_deg=float(a),right_phase_deg=float(b),signed_difference_deg=float(b-a),ULP_distance=n,maximum_allowed_ULP=2,zeros_must_match_bitwise=True,existing_geometry_comparison=detail,qualification='Recorded cross-run phase roundoff only; cause unknown; no physics/noise tolerance')

def validate_control(c,p):
 before,after=c['before'],c['after'];expected=copy.deepcopy(before);expected['timing'].update({'simulation time':150e-12,'use early shutoff':0});expected['optimization_dft_enabled']=0
 need(before['timing']['simulation time']==2000e-12 and before['timing']['use early shutoff']==1 and before['timing']['auto shutoff min']==1e-7 and before['timing']['use divergence checking']==1 and before['optimization_dft_enabled']==1,'Baseline timing/divergence/monitor changed')
 need(c['changed_properties']=={'FDTD::simulation time':[2000e-12,150e-12],'FDTD::use early shutoff':[1,0],'optimization_dft::enabled':[1,0]},'Recorded timing intervention mismatch')
 need(after==expected and c['before_sha256']==semantic(before) and c['after_sha256']==semantic(after),'Control snapshots/intervention hash mismatch')
 need(c['persisted_sha256']==semantic(c['persisted']) and persisted_check(after,c['persisted'])==c['serialization'],'Persisted controls changed')
 geometry=eng.make_func(SPEC)(p)
 need(set(before['geometry'])==set(geometry) and all(abs(float(np.asarray(before['geometry'][k]).squeeze())-v)<1e-18 for k,v in geometry.items()),'Actual geometry differs from exact mapped point')
 dt=float(c['dt_s']);pulse=float(c['source_pulse_length_s']);offset=float(c['source_offset_s'])
 need(np.isfinite([dt,pulse,offset]).all() and dt>0 and dt==after['timing']['dt'] and pulse==after['source']['pulselength'] and offset==after['source']['offset'] and pulse>0 and offset>=2*pulse and offset+10*pulse<150e-12,'Pulse/timestep mismatch')

def saved_files(r,root,task):
 source=Path(r['source']).resolve();ports=Path(r['ports_npz']).resolve();h5=Path(r['h5']).resolve();root=Path(root).resolve()
 need(ports.parent.is_relative_to(root) and ports.parent.name.startswith(f'task_{task:02d}_') and ports.name=='ports.npz' and source.is_relative_to(ports.parent) and source.name=='fwd_default.fsp' and h5==source.with_suffix('')/'fwd_default_output.h5','Artifacts escaped unique family/attempt')
 for key,digest in [('source','fsp_sha256'),('h5','h5_sha256'),('ports_npz','ports_sha256')]:need(Path(r[key]).is_file() and sha(r[key])==r[digest],'Saved artifact missing/SHA mismatch: '+key)
 need(termination(source,150.,r['controls']['dt_s'],1)==r['termination'],'Saved termination/log mismatch')
 with np.load(ports,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 need(all(np.isfinite(v).all() for v in arrays.values()),'Saved nonfinite port arrays')
 scores=recovery.score(arrays);need(all(np.isfinite(r[k]) and abs(float(scores[k])-float(r[k]))<=1e-12 for k in diag.METRICS),'Saved port scores not reproduced')
 return arrays

def validate_parent_metadata(row,task,marker,marker_path):
 i,r=row['identity'],row['result'];p,index,d,s=base.task_params(task)
 review=dict(path=str(marker_path),sha256=PARENT_MARKER_SHA,duration_ps=150.,lead_verdict=marker['lead_verdict'],allowed_tasks=[3,4,5],first_stage=marker['first_stage'])
 expected=dict(schema=1,task=task,spec=marker['spec'],point=p.tolist(),index=index,sign=s,delta_nm=s*.0625,duration_ps=150.,task_map=marker['task_map'],center_checkpoint='/work/results/diagnose_te_s2_fdstep/results/lumopt2_te_s2_fdstep_diagnostic_20261007/task_00.json',center_sha256=base.CENTER_SHA,code=marker['code'],vendor_api_sha256=marker['vendor_api_sha256'],parent_review=review,timing_changes=marker['timing_changes'])
 need(i==expected and r['params']==p.tolist() and r['task']==task and r['index']==index and r['sign']==s and r['delta_nm']==s*.0625 and r['duration_ps']==150. and r['parent_review']==review and r['ports_sha256']==PARENT_PORT_SHA[task] and r['vendor_api_sha256']==marker['vendor_api_sha256'],'Exact150ps ancestry checkpoint metadata mismatch')
 validate_control(r['controls'],p)

def load_ancestry():
 frozen_base();root=envpath('S2_STEP150_PARENT_ROOT',PARENT_ROOT);mp=envpath('S2_STEP150_PARENT_MARKER',PARENT_MARKER)
 need(sha(mp)==PARENT_MARKER_SHA,'Exact reviewed150ps parent marker SHA required');m=json.loads(mp.read_text())
 need(m['family']==base.LABEL and m['duration_ps']==150. and m['allowed_tasks']==[3,4,5] and m['status']==base.STATUS[150.] and m['code']==base.code() and m['vendor_api_sha256']==base.vendor_sources() and m['spec']==plain(dataclasses.asdict(base.SPEC)) and m['timing_changes']=={'simulation time':150e-12,'use early shutoff':0},'Frozen150ps parent family/source/API mismatch')
 need(str(root)==m['output_root'],'Exact parent root required')
 rows={};bindings={};arrays={}
 for t in [3,4,5]:
  cp=root/f'task_{t:02d}.json';need(sha(cp)==PARENT_SHA[t],'Exact150ps checkpoint byte SHA mismatch');row=json.loads(cp.read_text());validate_parent_metadata(row,t,m,mp)
  arrays[t]=saved_files(row['result'],root,t);rows[t]=row;z=row['result']
  bindings[str(t)]=dict(checkpoint=str(cp),sha256=PARENT_SHA[t],ports_sha256=z['ports_sha256'],fsp_sha256=z['fsp_sha256'],h5_sha256=z['h5_sha256'],termination=z['termination'])
 for t in [4,5]:
  for k in ['p1_f','p1_lambda','p2_f','p2_lambda']:need(np.array_equal(arrays[t][k],arrays[3][k]) and arrays[t][k].size==501,'Parent exact501 grids changed')
 return dict(checkpoint=str(root/'task_03.json'),sha256=CENTER_SHA,result=rows[3]['result'],bindings=dict(marker=str(mp),marker_sha256=PARENT_MARKER_SHA,checkpoints=bindings))

def root_path():
 root=envpath('S2_STEP150_OUTPUT_ROOT',DEFAULT_OUTPUT);old=envpath('S2_STEP150_PARENT_ROOT',PARENT_ROOT)
 need(root.name==LABEL and not root.is_relative_to(old) and not old.is_relative_to(root),'Exclusive new output family required');return root

def scope(root,center):
 return dict(schema=1,family=LABEL,duration_ps=150.,step_nm=.125,index=149,task_map=plain(TASK_MAP),center_checkpoint=center['checkpoint'],center_sha256=CENTER_SHA,code=code(),base_helper_sha256=BASE_SHA,vendor_api_sha256=base.vendor_sources(),spec=plain(dataclasses.asdict(SPEC)),output_root=str(root),allowed_tasks=[0,1],timing_changes={'simulation time':150e-12,'use early shutoff':0},parent150=center['bindings'])
def review_template(root,center):
 m=scope(root,center);m.update(status='NOT_REVIEWED',allowed_tasks=[],lead_verdict='NOT_REVIEWED');return m

def reviewed(task,root,center):
 mp=envpath('S2_STEP150_REVIEW_MARKER');expected_sha=os.environ.get('S2_STEP150_REVIEW_SHA256');need(isinstance(expected_sha,str) and len(expected_sha)==64,'Explicit new review SHA required');need(sha(mp)==expected_sha,'New review marker byte SHA mismatch');m=json.loads(mp.read_text());expected=scope(root,center)
 need(all(m.get(k)==v for k,v in expected.items()) and m.get('status')==APPROVAL_STATUS,'Explicit two-task150ps step review scope mismatch')
 need(isinstance(m.get('lead_verdict'),str) and m['lead_verdict'].strip() not in ('','NOT_REVIEWED'),'Substantive leadreview required');need(task in m['allowed_tasks'],'Task not reviewed')
 return dict(path=str(mp),sha256=expected_sha,lead_verdict=m['lead_verdict'],allowed_tasks=[0,1],parent150=center['bindings'])
def identity(task,center,review):
 p,index,d,s=task_params(task);return dict(schema=1,task=int(task),spec=plain(dataclasses.asdict(SPEC)),point=p.tolist(),index=index,sign=s,delta_nm=s*.125,duration_ps=d,task_map=plain(TASK_MAP),center_checkpoint=center['checkpoint'],center_sha256=CENTER_SHA,code=code(),vendor_api_sha256=base.vendor_sources(),parent_review=review,timing_changes={'simulation time':150e-12,'use early shutoff':0})

def center_control_comparison(before,center):
 # Geometry intentionally changes at exactly mapped index149,validated separately.
 expected=copy.deepcopy(center['result']['controls']['before']);expected['geometry']=copy.deepcopy(before['geometry'])
 out=phase_compare(expected,before);out['geometry_qualification']='Full new mapped point validated separately; intentional shift30 step excluded from nominal control comparison';return out

def verify_result(r,task,root,center,review):
 p,index,d,s=task_params(task);expected=dict(task=task,index=index,duration_ps=d,sign=s,delta_nm=s*.125,params=p.tolist(),parent_review=review,reviewed_center_sha256=CENTER_SHA,vendor_api_sha256=base.vendor_sources(),existing_overall_gradient_gate='FAILED_UNCHANGED',comparison_status=COMPARISON_STATUS)
 need(all(r.get(k)==v for k,v in expected.items()),'Completed new-step result identity mismatch');validate_control(r['controls'],p)
 need(r['center_control_comparison']==center_control_comparison(r['controls']['before'],center),'Saved nominal control comparison changed')
 arrays=saved_files(r,root,task)
 with np.load(center['result']['ports_npz'],allow_pickle=False) as z:
  for k in ['p1_f','p1_lambda','p2_f','p2_lambda']:need(arrays[k].size==501 and np.array_equal(arrays[k],z[k]),'Exact150ps center port grid mismatch')

def forward_only(task,root,center,review):
 p,index,d,s=task_params(task);attempt=Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_',dir=root));project,_=eng.make_project(SPEC,str(attempt),eng.import_lumopt2())
 try:
  project.set_fwd(p);f=project.fdtd_session.fdtd;c=controls(f,p,150.);comparison=center_control_comparison(c['before'],center)
  configs=list(project.fom.config_map.config_map.values());need(len(configs)==1,'Exactly one port-only forward required');source=Path(project.fom.config_map.project_folder)/configs[0]['fwd_filename'];need(source.resolve().is_relative_to(attempt.resolve()),'New source escaped exclusive attempt')
  f.save(str(source));setup_sha=sha(source);project.fdtd_session.invalidate_loaded_state();f.load(str(source));persisted=snapshot(f,p);c.update(persisted=persisted,persisted_sha256=semantic(persisted),serialization=persisted_check(c['after'],persisted))
  project.fdtd_session.invalidate_loaded_state();project.run_forward();h5=source.with_suffix('')/'fwd_default_output.h5';need(h5.is_file(),'Companion H5 required');f.load(str(source));t=termination(source,150.,c['dt_s'],f.getresult('FDTD','status'));arrays=recovery.port_arrays(f)
  with np.load(center['result']['ports_npz'],allow_pickle=False) as z:
   for k in ['p1_f','p1_lambda','p2_f','p2_lambda']:need(np.array_equal(arrays[k],z[k]),'Exact center grid changed')
  ports=attempt/'ports.npz'
  with ports.open('xb') as stream:np.savez(stream,**arrays)
  return dict(task=task,index=index,duration_ps=d,sign=s,delta_nm=s*.125,params=p.tolist(),source=str(source),h5=str(h5),ports_npz=str(ports),ports_sha256=sha(ports),fsp_sha256=sha(source),h5_sha256=sha(h5),setup_fsp_sha256=setup_sha,controls=c,center_control_comparison=comparison,termination=t,parent_review=review,reviewed_center_sha256=CENTER_SHA,**recovery.score(arrays),FDTD_version=plain(f.version()),vendor_api_sha256=base.vendor_sources(),comparison_status=COMPARISON_STATUS,existing_overall_gradient_gate='FAILED_UNCHANGED')
 finally:project.fdtd_session.fdtd.close()
def main(task_idx):
 task_params(task_idx);task=int(task_idx);root=root_path();center=load_ancestry();review=reviewed(task,root,center);ident=identity(task,center,review);root.mkdir(parents=True,exist_ok=True);cp=root/f'task_{task:02d}.json';saved=recovery.read_checkpoint(cp,ident)
 if saved is not None:verify_result(saved,task,root,center,review);print('REUSED150ps step checkpoint',cp);return saved
 need(os.environ.get('KEEP_H5')=='1','KEEP_H5=1 required before CAD');r=forward_only(task,root,center,review);verify_result(r,task,root,center,review);recovery.checkpoint(cp,ident,r);print('New150ps step diagnostic; review required',cp);return r
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=int);p.add_argument('--tasklist',action='store_true');p.add_argument('--review-template',action='store_true');a=p.parse_args()
 if a.tasklist:print(json.dumps(dict(family=LABEL,N_TASKS=N_TASKS,task_map=plain(TASK_MAP),duration_ps=150.,IS_HELPER=True)))
 elif a.review_template:print(json.dumps(review_template(root_path(),load_ancestry()),indent=2))
 elif a.task is not None:main(a.task)
 else:p.error('--task,--tasklist or --review-template required')
