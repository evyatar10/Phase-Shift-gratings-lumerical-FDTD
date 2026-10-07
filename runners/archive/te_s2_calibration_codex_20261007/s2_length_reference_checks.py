"""Metadata-only immutable150ps reference binding; never accesses FSP/H5/CAD."""
import hashlib,json,math
from pathlib import Path
CENTER_SHA='a0ff55eacaa52e00a2aab17dcc43d0e5fff9a72566e15da371ff25279c172663'
PROFILE_SHA='01a6fe6540c7cf903de67cecc42a89c6318983301364b679b9da0870dede595a'
GRID_SHA='9aaaee8efe7d6017869b02705895ba1342c53175870dbcf46c8892b54241d6f4'
ACCESS_POLICY='METADATA_ONLY_NO_REFERENCE_FSP_H5_OR_SERVER_ACCESS'
def need(value,message):
 if not value:raise RuntimeError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(cp,profile):
 i,r=cp['identity'],cp['result'];point=i['point'];spec=i['spec']
 need(len(point)==296 and point==r['params'] and all(type(v) in (int,float) and math.isfinite(v) for v in point),'Finite exact296point required')
 need(i['task']==r['task']==3 and i['duration_ps']==r['duration_ps']==150. and i['sign']==r['sign']==0 and i['delta_nm']==r['delta_nm']==0. and i['index']==r['index']==29,'Original nominal150ps identity required')
 need(spec['n_periods_side']==98 and spec['n_free']==60 and spec['bare'] is True and spec['n_wl_points']==501 and spec['polarization']=='TE','Original N98/free60/bareTE501 family required')
 c=r['controls'];after=c['persisted'];t=r['termination']
 need(after['timing']['simulation time']==150e-12 and after['timing']['use early shutoff']==0 and after['timing']['use divergence checking']==1 and after['optimization_dft_enabled']==0 and after['monitor']['monitor frequency points']==501,'Exact150ps/501/DFToff/divergence controls required')
 need(t['status']==1 and t['requested_time_s']==150e-12 and abs(t['physical_time_s']-150e-12)<=c['dt_s'] and t['reason']=='completed_full_simulation_time','Original full150ps successful termination required')
 need(profile['status']=='SAVED_RAW_PROFILE_EXTRACTION_COMPLETE_NO_PHYSICS_PASS' and profile['no_solve'] is True and profile['all_inputs_preserved'] is True,'Actual immutable profile extraction required')
 inputs=profile['inputs']['150'];stage=profile['stages']['150']
 need(inputs[0]['sha256']==CENTER_SHA and inputs[1]['path']==r['source'] and inputs[1]['sha256']==r['fsp_sha256'] and inputs[2]['path']==r['h5'] and inputs[2]['sha256']==r['h5_sha256'] and inputs[3]['path']==r['ports_npz'] and inputs[3]['sha256']==r['ports_sha256'],'Profile inventory is not bound to exact center artifacts')
 need(stage['duration_ps']==150 and stage['raw_profile_E_shape'][-2:]==[501,3],'Recorded broadband profile dimensions required')
 need(all(v['available'] is True for v in stage['FDTD_boundary_property_inventory'].values()),'Actual PML/boundary inventory required')
 need(stage['monitor_geometry']['enabled']['value']==1 and stage['monitor_geometry']['frequency points']['value']==501,'Enabled501 field_profile required')
 return dict(point=list(point),checkpoint=cp,profile_stage=stage,reference_original_source=r['source'],reference_original_h5=r['h5'],access_policy=ACCESS_POLICY)
def load(cp_path,profile_path,grid_path):
 cp_path=Path(cp_path);profile_path=Path(profile_path);grid_path=Path(grid_path)
 need(cp_path.is_absolute() and profile_path.is_absolute() and grid_path.is_absolute(),'Absolute metadata paths required')
 need(sha(cp_path)==CENTER_SHA and sha(profile_path)==PROFILE_SHA and sha(grid_path)==GRID_SHA,'Exact unchanged metadata bytes required')
 result=validate(json.loads(cp_path.read_text()),json.loads(profile_path.read_text()))
 grid=json.loads(grid_path.read_text());need(grid['source_original_ports_sha256']==result['checkpoint']['result']['ports_sha256'] and grid['source_original_center_cp_sha256']==CENTER_SHA,'Grid metadata origin mismatch');result['grids']=grid['grids']
 result['binding']=dict(port_grid_metadata=str(grid_path),port_grid_sha256=GRID_SHA,checkpoint_metadata=str(cp_path),checkpoint_sha256=CENTER_SHA,profile_metadata=str(profile_path),profile_sha256=PROFILE_SHA,access_policy=ACCESS_POLICY)
 return result
