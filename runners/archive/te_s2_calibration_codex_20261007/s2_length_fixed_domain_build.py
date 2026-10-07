"""Private portability adapter for frozen builder1788f251; exact function bodies. No entrypoint/CAD on import."""
import hashlib,json
from pathlib import Path
import numpy as np
try:
 from .s2_length_no_solve_guard import NoSolveCAD
except ImportError:
 from s2_length_no_solve_guard import NoSolveCAD

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def need(x,msg):
 if not x:raise RuntimeError(msg)

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x

def scalar(x):return float(np.asarray(x).squeeze())

class PairCAD(NoSolveCAD):
 def __getattr__(self,name):
  if name.lower() in {'engine','fdtd_engine','launch','execute','addjobs','submit','submitjob'}:raise RuntimeError('No engine/job invocation allowed: '+name)
  return super().__getattr__(name)
 def load(self,path):
  need(Path(path).resolve().is_relative_to(self._root) and Path(path).is_file(),'Only owned new FSP may be reloaded')
  return super().load(path)

def differences(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  result={}
  for key in set(a)|set(b):
   q=path+'/'+key
   if key not in a or key not in b:result[q]={'left':a.get(key),'right':b.get(key)}
   else:result.update(differences(a[key],b[key],q))
  return result
 if a==b:return {}
 return {path:{'left':a,'right':b}}

def geometry_comparison(a,b):
 need(set(a)==set(b),'Semantic geometry keys differ');raw={};blocked={}
 for key in a:
  x,y=scalar(a[key]),scalar(b[key]);delta=y-x
  if delta:
   row=dict(left_m=x,right_m=y,delta_m=delta)
   okay=np.isfinite([x,y]).all() and abs(delta)<=1e-20 and abs(delta)<=64*np.finfo(float).eps*max(abs(x),abs(y))
   row['within_existing_serialization_allowance']=bool(okay);raw[key]=row
   if not okay:blocked[key]=row
 return dict(nonbitwise=raw,blockers=blocked,max_abs_delta_m=max([abs(v['delta_m']) for v in raw.values()]+[0.]),qualification='Existing reviewed geometry serialization bound only:absdelta<=1e-20m and64eps*scale;no physical tolerance')

def property_read(f,obj,keys):return {k:plain(f.getnamed(obj,k)) for k in keys}

FDTD_KEYS=['dimension','x','y','z','x span','y span','z span','mesh type','mesh accuracy','define x mesh by','define y mesh by','define z mesh by','dx','dy','dz','allow grading in x','allow grading in y','allow grading in z','grading factor','dt stability factor','force symmetric y mesh','force symmetric z mesh','background material','background index','x min bc','x max bc','y min bc','y max bc','z min bc','z max bc','pml layers','pml profile','mesh refinement','meshing refinement']

TIME_KEYS=['simulation time','use early shutoff','auto shutoff min','use divergence checking','auto shutoff max','down sample time','max source time signal length','dt']

SOURCE_KEYS=['pulselength','offset','frequency','bandwidth','source type','set time domain','set frequency','wavelength start','wavelength stop','optimize for short pulse']

PORT_KEYS=['x','y','z','x span','y span','z span','phase','injection axis','direction','enabled','mode selection','frequency dependent profile']

PROFILE_KEYS=['enabled','monitor type','x','y','z','x span','y span','z span','override global monitor settings','use source limits','frequency points','spatial interpolation','use wavelength spacing','down sample x','down sample y']

GEO_KEYS=['x','x span','y','y span','z','z span','material','index']

def optional_read(f,obj,keys):
 result={}
 for k in keys:
  try:result[k]={'available':True,'value':plain(f.getnamed(obj,k))}
  except Exception as e:result[k]={'available':False,'error':type(e).__name__+': '+str(e)}
 return result

def all_teeth(f,eng,n):
 # Names must be actual current objects,then canonical semantic names remove suffixes.
 names,cavity=eng.tooth_names(n,n);out={}
 for d,quad in names.items():
  for arm,kind,name in zip(['L','L','R','R'],['narrow','wide','narrow','wide'],quad):out[arm+'_'+kind+'_'+str(d)]=property_read(f,name,GEO_KEYS)
 out['cavity']=property_read(f,cavity,GEO_KEYS)
 for name in ['wg_left_inf_1','wg_right_inf_'+str(4*n+3)]:out[name.split('_inf_')[0]+'_feed']=property_read(f,name,GEO_KEYS)
 return out

def flatten_numeric_geometry(properties):
 numeric={};text={}
 for name,props in properties.items():
  for key,value in props.items():
   if isinstance(value,str):text[name+'::'+key]=value
   else:numeric[name+'::'+key]=scalar(value)
 return numeric,text

def restore_retained_outer_and_feed_joins(f,eng,baseline):
 """Hold retained N80 outer x at measured freshN98 values;not a tolerance waiver."""
 names,cavity=eng.tooth_names(80,80);changes={}
 for d in range(61,81):
  for arm,kind,name in zip(['L','L','R','R'],['narrow','wide','narrow','wide'],names[d]):
   semantic=arm+'_'+kind+'_'+str(d);anchor=baseline[semantic]
   before=property_read(f,name,GEO_KEYS)
   before_numeric,before_text=flatten_numeric_geometry({semantic:{k:v for k,v in before.items() if k!='x'}});anchor_numeric,anchor_text=flatten_numeric_geometry({semantic:{k:v for k,v in anchor.items() if k!='x'}})
   non_x_roundoff=geometry_comparison(before_numeric,anchor_numeric)
   need(not non_x_roundoff['blockers'] and before_text==anchor_text,'Unexpected retained outer non-x property differs beyond existing serialization allowance:'+semantic)
   f.setnamed(name,'x',anchor['x']);changes[semantic]=dict(native_x_m=before['x'],held_N98_x_m=anchor['x'],readback_x_m=plain(f.getnamed(name,'x')),non_x_pre_set_vs_serialized_anchor=non_x_roundoff)
 # Actual restored outer-edge readbacks define joins;no cell-alignment assertion.
 left_outer=names[80][0];right_outer=names[80][3]
 left_edge=scalar(f.getnamed(left_outer,'x min'));right_edge=scalar(f.getnamed(right_outer,'x max'))
 left_feed='wg_left_inf_1';right_feed='wg_right_inf_323'
 feed_before={o:property_read(f,o,['x min','x max','x','x span']) for o in [left_feed,right_feed]}
 f.setnamed(left_feed,'x max',left_edge);f.setnamed(right_feed,'x min',right_edge)
 feed_after={o:property_read(f,o,['x min','x max','x','x span']) for o in [left_feed,right_feed]}
 need(feed_after[left_feed]['x min']==feed_before[left_feed]['x min'] and feed_after[right_feed]['x max']==feed_before[right_feed]['x max'],'Feed outer domain endpoint changed')
 return dict(explicit_preserved_outer_x=changes,actual_outer_edge_m=dict(left=left_edge,right=right_edge),feed_before=feed_before,feed_after=feed_after,join_delta_m=dict(left=scalar(feed_after[left_feed]['x max'])-left_edge,right=scalar(feed_after[right_feed]['x min'])-right_edge),qualification='Explicit baseline-derived geometry restoration;rawendpointdeltas retained,no new tolerance or physical width claim')

def snapshot(f,eng,spec,p):
 mapped=eng.make_func(spec)(p)
 geo={k:plain(f.getnamed(*k.rsplit('::',1))) for k in mapped}
 return dict(inner_geometry=geo,all_teeth_and_feeds=all_teeth(f,eng,spec.n_periods_side),mesh={a:plain(f.getresult('FDTD',a)) for a in ['x','y','z']},user_mesh={a:plain(f.getnamed('FDTD','user specified mesh '+a)) for a in ['x','y','z']},FDTD=optional_read(f,'FDTD',FDTD_KEYS),timing=property_read(f,'FDTD',TIME_KEYS),source={k:plain(f.getglobalsource(k)) for k in SOURCE_KEYS},ports={o:property_read(f,o,PORT_KEYS) for o in ['FDTD::ports::Port_1','FDTD::ports::Port_2']},ports_monitor=property_read(f,'FDTD::ports',['monitor frequency points','source port','source mode']),field_profile=property_read(f,'field_profile',PROFILE_KEYS),time_monitors={o:property_read(f,o,['x','y','z','enabled','monitor type']) for o in ['time_input','time_cavity','time_output']},optimization={o:property_read(f,o,['enabled','x','y','z','x span','y span','z span','spatial interpolation']) for o in ['optimization_dft','optimization_index']},optimization_mesh=property_read(f,'optimization_mesh',['x','y','z','x span','y span','z span','dx','dy','dz']),mesh_override=property_read(f,'mesh_override',['x','y','z','x span','y span','z span','dx','dy','dz']),global_monitor={k:plain(f.getglobalmonitor(k)) for k in ['frequency points','use wavelength spacing','use source limits']})

def configure(sim,f,eng,spec,p,cp,profile,portability):
 cfg=eng.build_base_cfg(spec)
 sim.build();sim.update_scan(center_lambda_m=cfg.spectral.center_wavelength_m,width_nm=cfg.spectral.scan_width_nm,n_points=cfg.spectral.n_wl_points)
 eng.apply_monitor_overrides(sim,cfg)
 for key,value in eng.make_func(spec)(p).items():
  obj,prop=key.rsplit('::',1);need(f.getnamednumber(obj)==1,'Missing mapped object:'+obj);f.setnamed(obj,prop,value)
 # Pure supported vendor CAD helpers are called without their constructor/runner.
 from lumopt2.core.fdtd_session import FdtdSession
 from lumopt2.utils.common import Box
 box=Box(x_span=2*(spec.pitch_nm/4+spec.n_free*spec.pitch_nm+1000)*1e-9,y_span=2*spec.region_y_half_nm*1e-9,z_span=.8e-6,dx=spec.region_dx_nm*1e-9,dy=50e-9,dz=50e-9)
 from types import SimpleNamespace
 session=SimpleNamespace(fdtd=f)
 FdtdSession.set_mesh_refinement_region(session,box,box.dx,box.dy,box.dz);FdtdSession.add_field_monitors(session,box);FdtdSession.add_index_monitors(session,box)
 f.setnamed('FDTD','mesh refinement',cp['refinement']['mesh refinement'])
 # Vendor project.py relock path,with ACTUAL pinned mesh relative to solver centers.
 for axis,values in cp['mesh'].items():f.setnamed('FDTD','user specified mesh '+axis,np.asarray(values)-scalar(f.getnamed('FDTD',axis)))
 f.setnamed('FDTD','mesh type','user specified')
 # Restore the recorded vendor one-cell forward-source offset and phase. No mode/adjoint solver invoked.
 for obj,properties in cp['ports'].items():
  for prop in ['x','phase']:f.setnamed(obj,prop,properties[prop])
 # Measurement coverage is held at actual saved N98 settings;physical N80 production crop is separate.
 for key,item in profile['monitor_geometry'].items():
  if item['available'] and key in ['x','y','z','x span','y span','z span','override global monitor settings','use source limits','frequency points','spatial interpolation']:
   if plain(f.getnamed('field_profile',key))!=item['value']:f.setnamed('field_profile',key,item['value'])
 for obj,x in [('time_input',-(98*spec.pitch_nm*1e-9+spec.pitch_nm*1e-9/4)-.5e-6),('time_output',98*spec.pitch_nm*1e-9+spec.pitch_nm*1e-9/4+.5e-6)]:f.setnamed(obj,'x',x)
 f.setnamed('FDTD','simulation time',150e-12);f.setnamed('FDTD','use early shutoff',0);f.setnamed('optimization_dft','enabled',0)
 portability.append('Exact saved vendor port/source coordinates and phase restored;Windows vendor bytes differ from Athena and nativeIGUM requalification remains required')

def compare_saved(current,cp,profile,blue):
 result={}
 for key in ['mesh','timing','source']:
  result[key]=differences(cp[key],current[key])
 result['ports']=differences(cp['ports'],{o:{k:v for k,v in properties.items() if k in cp['ports'][o]} for o,properties in current['ports'].items()})
 result['ports_monitor']=differences(cp['monitor'],current['ports_monitor'])
 result['refinement']=differences(cp['refinement'],{k:current['FDTD'][k]['value'] for k in cp['refinement']})
 result['profile_inventory']=differences({k:item['value'] for k,item in profile['monitor_geometry'].items() if item['available']},{k:current['field_profile'][k] for k,item in profile['monitor_geometry'].items() if item['available']})
 result['PML_inventory']=differences({k:item['value'] for k,item in profile['FDTD_boundary_property_inventory'].items() if item['available']},{k:current['FDTD'][k].get('value') for k,item in profile['FDTD_boundary_property_inventory'].items() if item['available']})
 return result
