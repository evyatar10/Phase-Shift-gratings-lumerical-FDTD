"""Preparation-only native IGUM N98/N80 fixed-domain150ps forwards; explicit review required."""
import copy,dataclasses,hashlib,json,os,re,sys,tempfile,traceback
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(os.environ.get('S2_LEN_REPO',Path.cwd())).resolve()))
import numpy as np
from runners.lumopt2_design import s2_temporal_forward as base
try:
 from . import s2_length_fixed_domain_build as build
 from . import s2_length_reference_checks as ref
except ImportError:
 import s2_length_fixed_domain_build as build
 import s2_length_reference_checks as ref
eng,val,recovery,diag=base.eng,base.val,base.recovery,base.diag
IS_HELPER=True
N_TASKS=2
LABEL='s2_length_N98_N80_fixed150_20261007'
SPEC=dataclasses.replace(base.SPEC,label=LABEL)
TASK_MAP=(98,80)
DURATION_PS=150.
BASE_SHA='555f3d13b42c494171e7658cc4263c588ce418e5d937da1497d690824b41d44a'
BUILDER_ORIGINAL_SHA='1788f251da546c1f6a2a2a956325f9fdf6dbb9fc3c345b84cfe2231fe490e798'
BUILDER_SHA='d5230121273d73a3fff4b7efe971ab945220be55c7b0682eb1b4a1a1bcb97c8c'
GUARD_SHA='40dd487831da68080b29813cdbef2e1ce9e6629528dccd3d7c431dfc1229e99d'
REF_SHA='1498643c2718607c07a2ccbe50a8278fdf90900041936867266c03380fb8d59f'
APPROVAL_STATUS='APPROVED_FOR_TWO_IGUM_FIXED150PS_N98_N80_FIXED_DOMAIN_FORWARDS_ONLY'
QUALIFICATION='FINITE_DURATION_LENGTH_SENSITIVITY_ONLY_NO_TEMPORAL_MESH_OR_GRADIENT_PASS'
plain,semantic,sha,need=base.plain,base.semantic,base.sha,base.need

def task_spec(task):
 if isinstance(task,bool) or not isinstance(task,(int,np.integer)) or task not in range(N_TASKS):raise ValueError('Task must be integer0..1')
 return dataclasses.replace(SPEC,n_periods_side=TASK_MAP[task])
def task_params(task,reference):
 spec=task_spec(task);p=np.asarray(reference['point'],float).copy();bounds=np.asarray(eng.param_bounds(spec))
 need(p.shape==(296,) and np.isfinite(p).all() and np.all((p>=bounds[:,0])&(p<=bounds[:,1])),'Exact296point/bounds required')
 need(spec.n_free==60 and spec.bare and spec.polarization=='TE' and spec.n_wl_points==501,'Retained TE/free60/bare501 family required')
 return p,spec
def canonical(properties):
 out={}
 for key,value in properties.items():
  obj,prop=key.split('::');match=re.fullmatch(r'([LR])_(narrow|wide)_(\d+)_\d+',obj)
  name='_'.join(match.groups()) if match else 'cavity' if re.fullmatch(r'cavity_\d+',obj) else obj
  result=name+'::'+prop;need(result not in out,'Duplicate semantic geometry');out[result]=float(value)
 return out
def code():
 need(sha(base.__file__)==BASE_SHA and sha(build.__file__)==BUILDER_SHA,'Frozen temporal/builder helper changed')
 guard=Path(__file__).with_name('s2_length_no_solve_guard.py');need(sha(guard)==GUARD_SHA and sha(ref.__file__)==REF_SHA,'Private guard/reference helper changed')
 return {**base.code(),Path(__file__).name:sha(__file__),Path(build.__file__).name:BUILDER_SHA,guard.name:GUARD_SHA,Path(ref.__file__).name:REF_SHA}
def envpath(name):
 value=os.environ.get(name);need(bool(value),name+' explicitly required');p=Path(value);need(p.is_absolute(),name+' must be absolute');return p.resolve()
def native_paths(for_review=True):
 import config
 need(sys.platform.startswith('linux'),'Fresh execution requires native IGUM Linux; no local FDTD')
 root=envpath('S2_LEN_OUTPUT_ROOT');repo=envpath('S2_LEN_REPO');host=envpath('S2_LEN_NATIVE_BASE')
 need(PurePosixPath(os.environ['S2_LEN_NATIVE_BASE'])==PurePosixPath('/home/evyatarrubin/research/bragg_sim_igum'),'Explicit native IGUM study base required')
 need(repo==host/'project' and root.name==LABEL and root.is_relative_to(host/'results') and root!=host/'results','Exclusive native result family required')
 cp=envpath('S2_LEN_REFERENCE_CP');profile=envpath('S2_LEN_REFERENCE_PROFILE');grid=envpath('S2_LEN_REFERENCE_GRIDS');marker=envpath('S2_LEN_REVIEW_MARKER') if for_review else None
 metadata=[cp,profile,grid]+([marker] if marker is not None else [])
 need(all(x.is_relative_to(host/'data') for x in metadata),'Only supplied native metadata/review paths under data are allowed')
 need(not any(root.is_relative_to(x.parent) for x in metadata),'Outputs overlap supplied metadata')
 api=Path(config.LUMAPI_PATH).resolve();need(api.is_file() and api.name=='lumapi.py' and not api.is_relative_to('/work'),'Native configured official API required')
 need(Path(eng.__file__).resolve().parents[2]==repo,'Live source root differs from reviewed native repo')
 return dict(root=root,repo=repo,host=host,cp=cp,profile=profile,grid=grid,marker=marker,api=api)
def load_reference(paths):
 reference=ref.load(paths['cp'],paths['profile'],paths['grid']);need(reference['checkpoint']['identity']['code']==base.code(),'Original150 source family differs')
 need(np.array_equal(reference['point'],val.te_point(SPEC)),'Reference validation point differs')
 a,b=[canonical(eng.make_func(task_spec(t))(np.asarray(reference['point']))) for t in [0,1]]
 need(a==b and len(a)==723,'Mapped inner60/cavity geometry changed')
 return reference
def scope(paths,reference):
 return dict(schema=1,family=LABEL,platform='NATIVE_IGUM',duration_ps=150.,task_map=list(TASK_MAP),allowed_tasks=[0,1],point=reference['point'],reference=reference['binding'],specs={str(t):plain(dataclasses.asdict(task_spec(t))) for t in [0,1]},output_root=str(paths['root']),repo=str(paths['repo']),API_path=str(paths['api']),code=code(),vendor_api_sha256=base.vendor_sources(),builder_original_sha256=BUILDER_ORIGINAL_SHA,termination_policy={'simulation_time_s':150e-12,'early_shutoff':0,'divergence_checking':1,'optimization_dft':0,'field_profile_points':501},qualification=QUALIFICATION)
def review_template(paths,reference):
 m=scope(paths,reference);m.update(status='NOT_REVIEWED',allowed_tasks=[],lead_verdict='NOT_REVIEWED');return m
def reviewed(task,paths,reference):
 digest=os.environ.get('S2_LEN_REVIEW_SHA256');need(isinstance(digest,str) and re.fullmatch('[a-f0-9]{64}',digest),'Explicit review byte SHA required')
 need(sha(paths['marker'])==digest,'Review marker byte SHA mismatch');m=json.loads(paths['marker'].read_text());expected=scope(paths,reference)
 need(all(m.get(k)==v for k,v in expected.items()) and m.get('status')==APPROVAL_STATUS,'Exact two-task native study review scope required')
 need(isinstance(m.get('lead_verdict'),str) and m['lead_verdict'].strip() not in ('','NOT_REVIEWED') and task in m['allowed_tasks'],'Substantive explicit lead review required')
 return dict(path=str(paths['marker']),sha256=digest,lead_verdict=m['lead_verdict'],scope_sha256=semantic(expected),allowed_tasks=[0,1])
def identity(task,paths,reference,review):
 p,spec=task_params(task,reference)
 return dict(schema=1,task=int(task),physical_N=spec.n_periods_side,point=p.tolist(),spec=plain(dataclasses.asdict(spec)),duration_ps=150.,task_map=list(TASK_MAP),scope=scope(paths,reference),parent_review=review,overall_gradient_gate='FAILED_UNCHANGED')

def validate_case(case,reference,n):
 cp=reference['checkpoint']['result']['controls']['persisted'];stage=reference['profile_stage'];a,b=case['before'],case['reloaded']
 need(case['physical_N']==n and case['native_port_distance_periods']==(20 if n==98 else 38),'Physical length/gap family changed')
 need(case['original_saved_control_differences']==build.compare_saved(b,cp,stage,None) and not any(case['original_saved_control_differences'].values()),'Saved reference controls differ')
 need(not build.geometry_comparison(a['inner_geometry'],b['inner_geometry'])['blockers'],'Inner saved geometry changed')
 an,at=build.flatten_numeric_geometry(a['all_teeth_and_feeds']);bn,bt=build.flatten_numeric_geometry(b['all_teeth_and_feeds'])
 need(not build.geometry_comparison(an,bn)['blockers'] and at==bt,'Full saved geometry changed')
 need({k:v for k,v in a.items() if k not in ['inner_geometry','all_teeth_and_feeds']}=={k:v for k,v in b.items() if k not in ['inner_geometry','all_teeth_and_feeds']},'Saved controls changed')
 need(not build.geometry_comparison(canonical(cp['geometry']),canonical(b['inner_geometry']))['blockers'],'Original inner723 geometry differs')
 need(b['optimization']['optimization_dft']['enabled']==0 and b['field_profile']['enabled']==1 and b['global_monitor']['frequency points']==501 and b['field_profile']['frequency points']==501,'Recorder/501 controls changed')
 conversion=case['native_saved_format_construction'];need(conversion['qualification'].startswith('Native construction->saved-format conversion only'),'Preparatory saved-format provenance missing')
 need(conversion['reloaded']==case['serialized_native_geometry'],'Preparatory canonical baseline changed')
 if n==80:
  restoration=case['outer_geometry_restoration'];need(restoration['join_delta_m']=={'left':0.,'right':0.} and restoration['reloaded_endpoint_delta_m']=={'left':0.,'right':0.},'N80 feed joins differ')
def compare_pair(anchor,target):
 a,b=anchor['reloaded'],target['reloaded'];omit=['inner_geometry','all_teeth_and_feeds']
 need({k:v for k,v in a.items() if k not in omit}=={k:v for k,v in b.items() if k not in omit},'Pair source/domain/mesh/ports/PML/profile/timing differs')
 need(canonical(a['inner_geometry'])==canonical(b['inner_geometry']),'Pair inner723 geometry is not exact')
 common=set(a['all_teeth_and_feeds'])&set(b['all_teeth_and_feeds']);retained=[k for k in common if 'feed' not in k]
 need(all(a['all_teeth_and_feeds'][k]==b['all_teeth_and_feeds'][k] for k in retained),'Retained teeth differ')
 for key,sign in [('wg_left_feed',-1),('wg_right_feed',1)]:
  left,right=a['all_teeth_and_feeds'][key],b['all_teeth_and_feeds'][key]
  need(set(left)==set(right) and all(left[k]==right[k] for k in left if k not in ['x','x span']),'Feed transverse/material properties changed:'+key)
  need(left['x']+sign*left['x span']/2==right['x']+sign*right['x span']/2,'Feed outer domain endpoint changed:'+key)
 removed=sorted(set(a['all_teeth_and_feeds'])-set(b['all_teeth_and_feeds']));need(len(removed)==72 and all(re.fullmatch('[LR]_(narrow|wide)_(8[1-9]|9[0-8])',k) for k in removed),'Unexpected removed physical objects')
 return dict(retained_geometry_bitwise_equal=True,controls_bitwise_equal=True,removed_objects=removed,feed_changes={k:{'N98':a['all_teeth_and_feeds'][k],'N80':b['all_teeth_and_feeds'][k]} for k in common if 'feed' in k},qualification=QUALIFICATION)

def profile_metadata(f,reference):
 """E dataset stays inside CAD; transfer only axes/shape/availability/configuration."""
 data_query=build.scalar(f.havedata('field_profile'));result_query=build.scalar(f.haveresult('field_profile','E'))
 need(np.isfinite([data_query,result_query]).all() and data_query==result_query==1.,'Persisted field_profile raw data/E result unavailable')
 have_data=have_result=True
 f.eval('sf_meta_result=getresult("field_profile","E");sf_meta_x=sf_meta_result.x;sf_meta_y=sf_meta_result.y;sf_meta_z=sf_meta_result.z;sf_meta_f=sf_meta_result.f;sf_meta_lambda=sf_meta_result.lambda;sf_meta_shape=size(sf_meta_result.E);')
 axes={key:plain(np.asarray(f.getv(variable)).ravel()) for key,variable in [('x_m','sf_meta_x'),('y_m','sf_meta_y'),('z_m','sf_meta_z'),('f_Hz','sf_meta_f'),('lambda_m','sf_meta_lambda')]}
 shape=np.asarray(f.getv('sf_meta_shape')).ravel();need(np.isfinite(shape).all() and np.all(shape==shape.astype(int)),'Finite integral E shape required')
 metadata=dict(have_raw_data=have_data,have_E_result=have_result,axes=axes,E_shape=shape.astype(int).tolist(),monitor_configuration=build.property_read(f,'field_profile',build.PROFILE_KEYS),readback='COLD_LOADED_NEW_SOLVED_FSP_RESULT_DATASET_COORDINATES_NO_E_TRANSFER')
 validate_profile_metadata(metadata,reference);return metadata
def validate_profile_metadata(metadata,reference):
 need(metadata['have_raw_data'] is True and metadata['have_E_result'] is True and metadata['readback']=='COLD_LOADED_NEW_SOLVED_FSP_RESULT_DATASET_COORDINATES_NO_E_TRANSFER','Persisted profile availability/readback missing')
 expected=reference['profile_stage']['raw_profile_coordinate_inventory']
 need(metadata['E_shape']==expected['E_shape'],'Profile E axis shape changed')
 for key in ['x_m','y_m','z_m','lambda_m']:
  need(np.isfinite(metadata['axes'][key]).all() and np.array_equal(metadata['axes'][key],expected[key]),'Exact measured profile coordinate changed:'+key)
 need(np.array_equal(metadata['axes']['f_Hz'],reference['grids']['p2_f']['values']),'Exact profile501 frequency grid changed')
 actual=metadata['monitor_configuration'];need(actual['enabled']==1 and actual['frequency points']==501,'Profile recorder disabled/wrong grid')
 for key,item in reference['profile_stage']['monitor_geometry'].items():
  if item['available']:need(actual[key]==item['value'],'Profile monitor configuration changed:'+key)

def build_case(n,attempt,reference,anchor=None):
 from bragg_device import PiShiftBraggFDTD
 spec=task_spec(0 if n==98 else 1);p=np.asarray(reference['point']);cfg=eng.build_base_cfg(spec);cfg.mesh.n_periods_dist_to_port=20 if n==98 else 38
 sim=PiShiftBraggFDTD(**cfg.to_device_kwargs());f=build.PairCAD(sim.fdtd,attempt);sim.fdtd=f;notes=[]
 try:
  build.configure(sim,f,eng,spec,p,reference['checkpoint']['result']['controls']['persisted'],reference['profile_stage'],notes)
  native_before=build.all_teeth(f,eng,n);pre=attempt/f'N{n}_native_pre_correction.fsp';f.save(pre);f.load(pre);native_after=build.all_teeth(f,eng,n)
  an,at=build.flatten_numeric_geometry(native_before);bn,bt=build.flatten_numeric_geometry(native_after)
  conversion=dict(preparatory_fsp=str(pre),sha256=sha(pre),before=native_before,reloaded=native_after,raw_numeric_deltas=build.geometry_comparison(an,bn),text_differences=build.differences(at,bt),qualification='Native construction->saved-format conversion only; raw deltas retained, not approved by tolerance waiver')
  restoration=build.restore_retained_outer_and_feed_joins(f,eng,anchor['reloaded']['all_teeth_and_feeds']) if n==80 else None
  before=build.snapshot(f,eng,spec,p);source=attempt/'fwd_default.fsp';f.save(source);f.load(source);after=build.snapshot(f,eng,spec,p)
  case=dict(physical_N=n,native_port_distance_periods=cfg.mesh.n_periods_dist_to_port,source=str(source),setup_fsp_sha256=sha(source),before=before,reloaded=after,native_saved_format_construction=conversion,serialized_native_geometry=native_after,original_saved_control_differences=build.compare_saved(after,reference['checkpoint']['result']['controls']['persisted'],reference['profile_stage'],None),guard_audit=f.audit,version=plain(f.version()),portability_notes=notes)
  if restoration is not None:
   names=eng.tooth_names(80,80)[0];restoration['reloaded_endpoint_delta_m']=dict(left=build.scalar(f.getnamed('wg_left_inf_1','x max'))-build.scalar(f.getnamed(names[80][0],'x min')),right=build.scalar(f.getnamed('wg_right_inf_323','x min'))-build.scalar(f.getnamed(names[80][3],'x max')));case['outer_geometry_restoration']=restoration
  validate_case(case,reference,n);return case
 finally:sim.close()
def forward_only(task,paths,reference,review):
 need(os.environ.get('KEEP_H5')=='1' and os.environ.get('REQUIRE_GPU')=='1','KEEP_H5=1/REQUIRE_GPU=1 before fresh CAD')
 p,spec=task_params(task,reference);attempt=Path(tempfile.mkdtemp(prefix=f'task_{task:02d}_',dir=paths['root']))
 try:
  anchor_dir=attempt/'fresh_N98_anchor' if task==1 else attempt;anchor_dir.mkdir(exist_ok=True);anchor=build_case(98,anchor_dir,reference)
  target=build_case(80,attempt,reference,anchor) if task==1 else anchor;pair=compare_pair(anchor,target) if task==1 else None
  source=Path(target['source']);need(source.parent==attempt,'Target source escaped exclusive attempt')
  setup=attempt/'prepared_forward_before_solve.fsp'
  with setup.open('xb') as stream:stream.write(source.read_bytes())
  need(sha(setup)==target['setup_fsp_sha256'],'Immutable prepared source hash mismatch')
  target['immutable_setup_fsp']=str(setup)
  sys.path.insert(0,str(paths['api'].parent));import lumapi
  with lumapi.FDTD(hide=True) as f:
   f.load(str(source));threads=int(os.environ['SLURM_CPUS_PER_TASK']);need(threads>0,'Allocated threads required');f.setresource('FDTD',1,'device type','GPU');f.setresource('FDTD',1,'threads',str(threads))
   resource={'device_type':plain(f.getresource('FDTD',1,'device type')),'threads':plain(f.getresource('FDTD',1,'threads'))};need(str(resource['device_type']).strip().upper()=='GPU' and int(float(resource['threads']))==threads,'GPU/thread readback mismatch')
   need(build.snapshot(f,eng,spec,p)==target['reloaded'],'Prepared scene changed before solve')
   f.run('FDTD','GPU')
   term=base.termination(source,150.,target['reloaded']['timing']['dt'],f.getresult('FDTD','status'))
   # Explicitly prove the NEW solved FSP retains result access; no reference FSP is loaded.
   h5=source.with_suffix('')/(source.stem+'_output.h5');need(h5.is_file(),'Companion H5 required before cold readback');before_hashes={'fsp':sha(source),'h5':sha(h5)}
   f.load(str(source));need(base.termination(source,150.,target['reloaded']['timing']['dt'],f.getresult('FDTD','status'))==term,'Cold solved status/termination changed')
   profile=profile_metadata(f,reference);arrays=recovery.port_arrays(f);version=plain(f.version())
   need(before_hashes=={'fsp':sha(source),'h5':sha(h5)},'Cold result readback changed new solver files')
  h5=source.with_suffix('')/(source.stem+'_output.h5');need(h5.is_file(),'Companion H5 required');ports=attempt/'ports.npz'
  with ports.open('xb') as stream:np.savez(stream,**arrays)
  profile.update(source=str(source),h5=str(h5),fsp_sha256=sha(source),h5_sha256=sha(h5));profile['metadata_sha256']=semantic(profile)
  result=dict(task=task,physical_N=spec.n_periods_side,duration_ps=150.,params=p.tolist(),source=str(source),h5=str(h5),ports_npz=str(ports),fsp_sha256=sha(source),h5_sha256=sha(h5),ports_sha256=sha(ports),build_case=target,fresh_N98_anchor=anchor if task==1 else None,pair_comparison=pair,profile_metadata=profile,termination=term,resource=resource,FDTD_version=version,parent_review=review,vendor_api_sha256=base.vendor_sources(),qualification=QUALIFICATION,overall_gradient_gate='FAILED_UNCHANGED',**recovery.score(arrays))
  return result
 except Exception as exc:
  with (attempt/'failure_preserved.json').open('x',encoding='utf-8') as stream:json.dump(dict(task=int(task),physical_N=spec.n_periods_side,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),qualification='Attempt preserved; no completed checkpoint or scientific pass'),stream,indent=2)
  raise
def build_only(paths,output):
 """Explicit guarded native CAD qualification, separate from both solve tasks."""
 code();reference=load_reference(paths);output=Path(output)
 need(output.is_absolute() and not output.exists(),'NEW absolute build-only directory required')
 output=output.resolve();need(output.is_relative_to(paths['host']/'results') and not output.is_relative_to(paths['root']) and not paths['root'].is_relative_to(output),'Build-only output must be separate native results')
 sys.path.insert(0,str(paths['api'].parent));import lumapi
 klass=lumapi.FDTD;ctor=klass.__init__
 output.mkdir(parents=True,exist_ok=False);report=dict(scope='NATIVE_IGUM_GUARDED_CAD_BUILD_ONLY',no_FDTD_solve=True,reference=reference['binding'],code=code(),vendor_api_sha256=base.vendor_sources(),cases={})
 try:
  d98=output/'N98';d80=output/'N80';d98.mkdir();d80.mkdir()
  a=build_case(98,d98,reference);report['cases']['98']=a;b=build_case(80,d80,reference,a);report['cases']['80']=b;report['pair_comparison']=compare_pair(a,b)
  report['status']='NATIVE_CAD_PAIR_CONTROLS_PASS_NO_PHYSICS_OR_SOLVE_APPROVAL'
 except Exception as exc:
  report.update(status='NATIVE_BUILD_FAILED_CLOSED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());raise
 finally:
  report['FDTD_class_unchanged']=lumapi.FDTD is klass;report['FDTD_constructor_unchanged']=klass.__init__ is ctor
  changed=not report['FDTD_class_unchanged'] or not report['FDTD_constructor_unchanged']
  if changed:report.update(status='NATIVE_BUILD_FAILED_CLOSED',error='FDTD class/constructor identity changed')
  with (output/'native_build_only_report.json').open('x',encoding='utf-8') as stream:json.dump(plain(report),stream,indent=2,allow_nan=False)
  if changed:raise RuntimeError('FDTD class/constructor identity changed; report preserved')
 return report
def verify_result(r,task,paths,reference,review):
 p,spec=task_params(task,reference);need(r['task']==task and r['physical_N']==spec.n_periods_side and r['duration_ps']==150. and r['params']==p.tolist() and r['parent_review']==review and r['vendor_api_sha256']==base.vendor_sources() and r['qualification']==QUALIFICATION and r['overall_gradient_gate']=='FAILED_UNCHANGED','Completed result identity differs')
 validate_case(r['build_case'],reference,spec.n_periods_side)
 if task==1:validate_case(r['fresh_N98_anchor'],reference,98);need(r['pair_comparison']==compare_pair(r['fresh_N98_anchor'],r['build_case']),'Pair controls changed')
 else:need(r['fresh_N98_anchor'] is None and r['pair_comparison'] is None,'Nominal task acquired extra case')
 root=paths['root'];source=Path(r['source']).resolve();ports=Path(r['ports_npz']).resolve();h5=Path(r['h5']).resolve()
 need(ports.parent.is_relative_to(root) and ports.parent.name.startswith(f'task_{task:02d}_') and ports.name=='ports.npz' and source.parent==ports.parent and source.name=='fwd_default.fsp' and h5==source.with_suffix('')/(source.stem+'_output.h5'),'Artifacts escaped exclusive attempt')
 for key,digest in [('source','fsp_sha256'),('h5','h5_sha256'),('ports_npz','ports_sha256')]:need(Path(r[key]).is_file() and sha(r[key])==r[digest],'Completed artifact hash changed:'+key)
 metadata=r['profile_metadata'];validate_profile_metadata(metadata,reference)
 need(metadata['source']==r['source'] and metadata['h5']==r['h5'] and metadata['fsp_sha256']==r['fsp_sha256'] and metadata['h5_sha256']==r['h5_sha256'] and metadata['metadata_sha256']==semantic({k:v for k,v in metadata.items() if k!='metadata_sha256'}),'Profile metadata source/hash binding changed')
 case=r['build_case'];need(case['source']==r['source'],'Prepared source mismatch')
 setup=Path(case['immutable_setup_fsp']);need(setup.parent==ports.parent and setup.is_file() and sha(setup)==case['setup_fsp_sha256'],'Prepared pre-solve FSP missing/changed')
 if task==1:
  anchor=r['fresh_N98_anchor'];path=Path(anchor['source']);need(path.parent==ports.parent/'fresh_N98_anchor' and path.is_file() and sha(path)==anchor['setup_fsp_sha256'],'Fresh native N98 anchor FSP missing/changed')
 for c in [case]+([r['fresh_N98_anchor']] if task==1 else []):
  pre=Path(c['native_saved_format_construction']['preparatory_fsp']);need(pre.is_relative_to(ports.parent) and pre.is_file() and sha(pre)==c['native_saved_format_construction']['sha256'],'Preparatory FSP missing/changed')
 need(base.termination(source,150.,case['reloaded']['timing']['dt'],1)==r['termination'],'Full-time termination/log changed')
 with np.load(ports,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 need(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite port arrays');scores=recovery.score(arrays)
 need(all(np.isfinite(r[k]) and abs(float(scores[k])-float(r[k]))<=1e-12 for k in diag.METRICS),'Saved port scores do not reproduce')
 for key,expected in reference['grids'].items():
  need(list(arrays[key].shape)==expected['shape'] and np.array_equal(arrays[key].ravel(),np.asarray(expected['values'])),'Exact measured501 frequency/wavelength grid changed:'+key)
 need(r['resource']['device_type']=='GPU' and int(float(r['resource']['threads']))>0,'Completed GPU resource metadata missing')
def main(task_idx):
 task_spec(task_idx);task=int(task_idx);paths=native_paths();reference=load_reference(paths);review=reviewed(task,paths,reference);ident=identity(task,paths,reference,review);root=paths['root'];root.mkdir(parents=True,exist_ok=True);cp=root/f'task_{task:02d}.json'
 saved=recovery.read_checkpoint(cp,ident)
 if saved is not None:verify_result(saved,task,paths,reference,review);print('REUSED native fixed-domain length checkpoint',cp);return saved
 result=forward_only(task,paths,reference,review);verify_result(result,task,paths,reference,review);recovery.checkpoint(cp,ident,result);print('New length diagnostic; no convergence pass',cp);return result
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task',type=int);parser.add_argument('--tasklist',action='store_true');parser.add_argument('--review-template',action='store_true');parser.add_argument('--build-only-output');args=parser.parse_args()
 if args.tasklist:print(json.dumps(dict(family=LABEL,N_TASKS=2,task_map=list(TASK_MAP),duration_ps=150.,IS_HELPER=True)))
 elif args.review_template:
  paths=native_paths();print(json.dumps(review_template(paths,load_reference(paths)),indent=2))
 elif args.build_only_output:print(json.dumps({'status':build_only(native_paths(for_review=False),args.build_only_output)['status']}))
 elif args.task is not None:main(args.task)
 else:parser.error('--task/--tasklist/--review-template required')
