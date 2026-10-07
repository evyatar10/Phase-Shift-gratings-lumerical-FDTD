"""Six offline checks for the new forward-only diagnostic; no CAD/FDTD."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import diagnose_te_s2_fdstep as diag


class DiagnosticChecks(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='s2_fdstep_gate_'))
        center=diag.SPEC.scan_center_nm
        wl=np.linspace(center-1,center+1,501)
        line=1/(1+((wl-center)/.1)**2)
        self.arrays=dict(p2_lambda=wl*1e-9,p2_T_out=.93*line,p2_S=np.sqrt(.97*line),
                         p1_T_in=np.full(501,.94))
        metrics=diag.recovery.score(self.arrays)
        self.reference=self.root/'old_reference.json'
        self.reference.write_text(json.dumps(dict(identity=dict(task=0,
            point=diag.task_params(0)[0].tolist(),spec=dataclasses.asdict(diag.SPEC)),
            result=dict(s2_gate=dict(metrics,source='/old/preserved/fwd_default.fsp')))))
        self.digest=hashlib.sha256(self.reference.read_bytes()).hexdigest()
        self.env={'S2_FDSTEP_REFERENCE_CHECKPOINT':str(self.reference),
                  'S2_FDSTEP_REFERENCE_SHA256':self.digest,'KEEP_H5':'1'}
        self.builds=[]

    def make_project(self,spec,attempt,lmpt,mutate_mesh=False):
        source=Path(attempt)/'files/fwd_default.fsp'
        source.parent.mkdir()
        record=dict(forwards=0,closed=False,params=None,disabled=[])
        self.builds.append(record)
        class Fdtd:
            def getresult(inner,name,key):
                if name=='FDTD':
                    grid=np.arange(4.)
                    return grid+1 if mutate_mesh and record['disabled'] and key=='x' else grid
                port=2 if name.endswith('Port_2') else 1
                return {k.split('_',1)[1]:v for k,v in self.arrays.items() if k.startswith(f'p{port}_')}
            def setnamed(inner,name,key,value):
                record['disabled'].append((name,key,value))
            def save(inner,path):
                Path(path).write_bytes(b'fake FSP')
            def load(inner,path):
                pass
            def close(inner):
                record['closed']=True
        def set_fwd(p):
            record['params']=p.copy();source.write_bytes(b'fake FSP')
        def run_forward():
            record['forwards']+=1
            source.with_name(source.stem+'_p0.log').write_text('Simulation completed successfully')
            folder=source.with_suffix('');folder.mkdir()
            (folder/(source.stem+'_output.h5')).write_bytes(b'fake H5')
        project=types.SimpleNamespace(set_fwd=set_fwd,run_forward=run_forward,
            fdtd_session=types.SimpleNamespace(fdtd=Fdtd(),invalidate_loaded_state=lambda:None),
            fom=types.SimpleNamespace(config_map=types.SimpleNamespace(
                project_folder=str(source.parent),config_map={None:dict(fwd_filename=source.name)})))
        return project,None

    def test_exact_recovered_point_delta_bounds_identity_and_no_global_mutation(self):
        before=dataclasses.asdict(diag.recovery.SPEC)
        base,_=diag.task_params(0)
        np.testing.assert_equal(base,diag.val.te_point(diag.recovery.SPEC))
        for task,delta in ((1,.5),(2,-.5)):
            p,sign=diag.task_params(task)
            difference=p-base
            self.assertEqual(np.flatnonzero(difference).tolist(),[29])
            self.assertEqual(difference[29],delta)
            self.assertEqual(sign,1 if task==1 else -1)
        self.assertEqual(diag.INDEX,29)
        self.assertEqual(dataclasses.asdict(diag.recovery.SPEC),before)
        self.assertFalse(diag.SPEC.width_grad or diag.SPEC.wg_project)
        self.assertEqual((diag.SPEC.n_periods_side,diag.SPEC.n_free,diag.SPEC.polarization),(98,60,'TE'))
        self.assertTrue(diag.SPEC.bare)
        self.assertNotEqual(diag.SPEC.label,diag.recovery.SPEC.label)
        expected=hashlib.sha256(Path(diag.eng.__file__).read_bytes()).hexdigest()
        self.assertEqual(diag.identity(0)['code']['lumopt2_design.py'],expected)

    def test_plus_minus_block_without_center_before_any_project(self):
        with patch.dict(os.environ,self.env),patch.object(diag,'output_root',return_value=self.root), \
             patch.object(diag.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            for task in (1,2):
                with self.assertRaisesRegex(RuntimeError,'require task0'):
                    diag.main(task)

    def test_center_fake_forward_preserves_artifacts_and_resumes_without_new_solve(self):
        with patch.dict(os.environ,self.env),patch.object(diag,'output_root',return_value=self.root), \
             patch.object(diag.eng,'import_lumopt2',return_value=object()), \
             patch.object(diag.eng,'make_project',side_effect=self.make_project):
            result=diag.main(0)
            self.assertEqual(result['comparison_status'],'REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS')
            self.assertEqual(result['existing_overall_gradient_gate'],'FAILED_UNCHANGED')
            self.assertTrue(all(v['absolute_delta']==0 for v in result['old_center_comparison'].values()))
            for key in ('ports_npz','source','h5'):
                self.assertTrue(Path(result[key]).is_file())
            with patch.object(diag.eng,'import_lumopt2',side_effect=AssertionError('repeat solve')):
                self.assertEqual(diag.main(0),result)
        self.assertEqual(len(self.builds),1)
        self.assertEqual(self.builds[0]['forwards'],1)
        self.assertEqual(self.builds[0]['disabled'],[('optimization_dft','enabled',0)])
        self.assertTrue(self.builds[0]['closed'])

    def test_review_marker_exact_identity_and_checkpoint_no_overwrite(self):
        files={}
        for key in ('source','h5','ports_npz'):
            path=self.root/key;path.write_bytes(b'fixture');files[key]=str(path)
        with patch.dict(os.environ,self.env):
            expected=diag.identity(0)
            result=dict(task=0,sign=0,**files)
            target=self.root/'task_00.json'
            diag.recovery.checkpoint(target,expected,result)
            original=target.read_bytes()
            with self.assertRaises(FileExistsError):
                diag.recovery.checkpoint(target,expected,dict(result,sign=1))
            self.assertEqual(target.read_bytes(),original)
            marker=self.root/'center_reviewed.sha256'
            marker.write_text('0'*64)
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                diag.reviewed_center(self.root)
            marker.write_text(hashlib.sha256(original).hexdigest())
            self.assertEqual(diag.reviewed_center(self.root)[0],result)
            with patch.object(diag,'identity',return_value=dict(expected,task=9)):
                with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
                    diag.reviewed_center(self.root)

    def test_reference_hash_and_mesh_change_fail_closed(self):
        with patch.dict(os.environ,self.env):
            reference=diag.load_reference_center()
            self.reference.write_bytes(self.reference.read_bytes()+b' ')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                diag.load_reference_center()
        with patch.object(diag.eng,'make_project',side_effect=lambda *a:self.make_project(*a,mutate_mesh=True)):
            with self.assertRaisesRegex(RuntimeError,'changed x mesh'):
                diag.forward_only(0,self.root,object(),reference)
        self.assertEqual(self.builds[0]['forwards'],0)
        self.assertTrue(self.builds[0]['closed'])

    def test_helper_exact_task_membership_and_keep_h5_guard(self):
        self.assertTrue(diag.IS_HELPER)
        self.assertEqual(diag.N_TASKS,3)
        with self.assertRaises(ValueError):
            diag.main(3)
        env=dict(self.env,KEEP_H5='0')
        with patch.dict(os.environ,env),patch.object(diag,'output_root',return_value=self.root), \
             patch.object(diag.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            with self.assertRaisesRegex(RuntimeError,'KEEP_H5=1'):
                diag.main(0)


if __name__=='__main__':
    unittest.main()
