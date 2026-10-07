"""Offline staged-runner evidence/negative controls. Retain fixtures in temp."""
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runners.lumopt2_design import s2_fdhalf_classes as runner


class HalfChecks(unittest.TestCase):
    def setUp(self):
        self.temp=Path(tempfile.mkdtemp(prefix='fixture_',dir=None))
        self.root=self.temp/runner.LABEL
        self.centerpath=self.temp/'center'/'task_00.json';self.centerpath.parent.mkdir()
        wl=np.linspace(runner.SPEC.scan_center_nm-1,runner.SPEC.scan_center_nm+1,501)
        line=1/(1+((wl-runner.SPEC.scan_center_nm)/.1)**2)
        self.arrays=dict(p2_lambda=wl*1e-9,p2_T_out=.93*line,p2_S=np.sqrt(.97*line),p1_T_in=np.full(501,.94))
        metrics=runner.recovery.score(self.arrays)
        files={}
        for key in ('source','h5','ports_npz'):
            p=self.centerpath.parent/key;p.write_bytes(b'center fixture');files[key]=str(p)
        point=runner.val.te_point(runner.SPEC)
        self.centerrow=dict(identity=runner.diag.identity(0),result=dict(task=0,sign=0,params=point.tolist(),**files,**metrics))
        self.centerpath.write_text(json.dumps(self.centerrow))
        self.centerhash=runner.digest(self.centerpath)
        (self.centerpath.parent/'center_reviewed.sha256').write_text(self.centerhash)
        self.provenance=self.temp/'provenance.json'
        self.provenance.write_text(json.dumps(dict(critical_evidence_available=True,original_file_identity_unchanged=True)))
        self.first=self.temp/'first_review.json';self.later=self.temp/'later_review.json'
        self.env=dict(S2_HALF_CENTER_CHECKPOINT=str(self.centerpath),S2_HALF_OUTPUT_ROOT=str(self.root),
            S2_HALF_FIRST_REVIEW_MARKER=str(self.first),S2_HALF_LATER_REVIEW_MARKER=str(self.later),KEEP_H5='1')
        self.envpatch=patch.dict(os.environ,self.env);self.hashpatch=patch.object(runner,'CENTER_SHA256',self.centerhash)
        self.envpatch.start();self.hashpatch.start();self.addCleanup(self.envpatch.stop);self.addCleanup(self.hashpatch.stop)
        first=runner.review_template('first',self.root,runner.load_center(),self.provenance)
        self.assertEqual(first['status'],'NOT_REVIEWED');self.assertEqual(first['allowed_tasks'],[])
        first.update(status=runner.APPROVAL_STATUS['first'],allowed_tasks=[0,1,2,3])
        self.first.write_text(json.dumps(first))
        self.builds=[]

    def fake_project(self,spec,attempt,api,mesh_drift=False,missing_h5=False,fail=False):
        source=Path(attempt)/'files'/'fwd_default.fsp';source.parent.mkdir()
        record=dict(forwards=0,closed=False,disabled=False,params=None);self.builds.append(record)
        class Fdtd:
            def getresult(inner,name,key):
                if name=='FDTD':return np.arange(4.)+(1 if mesh_drift and record['disabled'] and key=='x' else 0)
                port=2 if name.endswith('Port_2') else 1
                return {k.split('_',1)[1]:v for k,v in self.arrays.items() if k.startswith(f'p{port}_')}
            def setnamed(inner,name,key,value):
                self.assertEqual((name,key,value),('optimization_dft','enabled',0));record['disabled']=True
            def save(inner,path):self.assertEqual(Path(path),source);source.write_bytes(b'new setup')
            def load(inner,path):self.assertEqual(Path(path),source)
            def close(inner):record['closed']=True
        def set_fwd(p):record['params']=p.copy();source.write_bytes(b'initial new setup')
        def run_forward():
            record['forwards']+=1
            if fail:raise RuntimeError('interrupted synthetic solve')
            source.with_name(source.stem+'_p0.log').write_text('Simulation completed successfully')
            if not missing_h5:
                folder=source.with_suffix('');folder.mkdir();(folder/(source.stem+'_output.h5')).write_bytes(b'H5 fixture')
        def forbidden(*a,**kw):raise AssertionError('adjoint/optimizer called')
        return types.SimpleNamespace(set_fwd=set_fwd,run_forward=run_forward,compute_gradient=forbidden,run_adjoint=forbidden,
            fdtd_session=types.SimpleNamespace(fdtd=Fdtd(),invalidate_loaded_state=lambda:None),
            fom=types.SimpleNamespace(config_map=types.SimpleNamespace(project_folder=str(source.parent),config_map={None:dict(fwd_filename=source.name)}))),None

    def wave(self,tasks):
        with patch.object(runner.eng,'import_lumopt2',return_value=object()),patch.object(runner.eng,'make_project',side_effect=self.fake_project):
            return [runner.main(task) for task in tasks]

    def approve_later(self):
        row=runner.review_template('later',self.root,runner.load_center(),self.provenance)
        self.assertEqual(row['status'],'NOT_REVIEWED');self.assertEqual(row['lead_verdict'],'NOT_REVIEWED')
        row.update(status=runner.APPROVAL_STATUS['later'],allowed_tasks=list(range(4,12)),
            lead_verdict='Parent reviewed four scalar/full-spectrum checkpoints; proceed with bounded remaining FD only. Calibration remains failed.')
        with self.later.open('x') as stream:json.dump(row,stream)
        return row

    def test_live_map_exact_point_deltas_and_cavity_not_comb(self):
        self.assertEqual(runner.INDICES,[0,29,60,120,149,295])
        expected=[29,29,149,149,0,0,60,60,120,120,295,295]
        base=runner.val.te_point(runner.SPEC)
        for task,index in enumerate(expected):
            p,i,h,sign,label=runner.task_params(task)
            self.assertEqual(i,index);self.assertEqual(h,.5)
            np.testing.assert_array_equal(np.flatnonzero(p-base),[index])
            self.assertEqual((p-base)[index],.5 if task%2==0 else -.5)
        for task in (10,11):
            p,*rest=runner.task_params(task);self.assertEqual(p[180],base[180]);self.assertNotEqual(p[295],base[295])
        for bad in (-1,12,True,1.2,'0'):
            with self.assertRaises(ValueError):runner.main(bad)

    def test_first_marker_scope_missing_approval_and_keep_h5_before_cad(self):
        row=json.loads(self.first.read_text())
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            for key,value in (('status','NOT_REVIEWED'),('allowed_tasks',list(range(12))),('runner_sha256','0'*64)):
                altered=dict(row);altered[key]=value;self.first.write_text(json.dumps(altered))
                with self.assertRaises(RuntimeError):runner.main(0)
            self.first.write_text(json.dumps(row))
            with self.assertRaises(FileNotFoundError):runner.main(4)
            with patch.dict(os.environ,{'KEEP_H5':'0'}):
                with self.assertRaisesRegex(RuntimeError,'KEEP_H5=1'):runner.main(0)
        self.assertEqual(self.builds,[])

    def test_first_four_forward_resume_and_exclusive_checkpoint(self):
        originalcenter=self.centerpath.read_bytes();originalmarker=self.first.read_bytes()
        results=self.wave(range(4))
        for task,result in enumerate(results):
            runner.verify_result(result,task,self.root)
            before=(self.root/f'task_{task:02d}.json').read_bytes()
            with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('repeat solve')):
                self.assertEqual(runner.main(task),result)
            with self.assertRaises(FileExistsError):runner.recovery.checkpoint(self.root/f'task_{task:02d}.json',json.loads(before)['identity'],result)
            self.assertEqual((self.root/f'task_{task:02d}.json').read_bytes(),before)
        self.assertEqual(self.centerpath.read_bytes(),originalcenter);self.assertEqual(self.first.read_bytes(),originalmarker)
        self.assertEqual(len(self.builds),4);self.assertTrue(all(b['forwards']==1 and b['closed'] for b in self.builds))

    def test_later_requires_all_authentic_first_wave_and_separate_new_marker(self):
        self.wave(range(3))
        with self.assertRaisesRegex(RuntimeError,'completed first-wave task 3'):
            runner.review_template('later',self.root,runner.load_center(),self.provenance)
        self.wave([3]);firstbytes=self.first.read_bytes();later=self.approve_later()
        self.assertEqual(len(later['first_wave']['checkpoints']),4)
        self.wave(range(4,12))
        self.assertEqual(self.first.read_bytes(),firstbytes)
        self.assertEqual(len(self.builds),12)
        self.assertEqual(self.builds[10]['params'][295]-runner.val.te_point(runner.SPEC)[295],.5)
        with patch.dict(os.environ,{'S2_HALF_LATER_REVIEW_MARKER':str(self.first)}),patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            with self.assertRaisesRegex(RuntimeError,'separate'):runner.main(4)

    def test_later_hash_lead_verdict_and_tampered_ports_refuse_without_new_solve(self):
        results=self.wave(range(4));later=self.approve_later()
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            altered=copy.deepcopy(later);altered['first_wave']['checkpoints']['2']['sha256']='0'*64;self.later.write_text(json.dumps(altered))
            with self.assertRaisesRegex(RuntimeError,'checkpoint identities'):runner.main(4)
            altered=copy.deepcopy(later);altered['lead_verdict']='NOT_REVIEWED';self.later.write_text(json.dumps(altered))
            with self.assertRaisesRegex(RuntimeError,'lead review verdict'):runner.main(4)
            self.later.write_text(json.dumps(later))
            Path(results[0]['ports_npz']).write_bytes(b'tamper fixture')
            with self.assertRaisesRegex(RuntimeError,'ports hash mismatch'):runner.main(4)
        self.assertEqual(len(self.builds),4)

    def test_mesh_h5_failure_cold_resume_and_source_identity(self):
        with patch.object(runner.eng,'import_lumopt2',return_value=object()):
            for kwargs,pattern in ((dict(mesh_drift=True),'changed x mesh'),(dict(missing_h5=True),'HDF5'),(dict(fail=True),'interrupted')):
                with patch.object(runner.eng,'make_project',side_effect=lambda *a,kw=kwargs:self.fake_project(*a,**kw)):
                    with self.assertRaisesRegex(RuntimeError,pattern):runner.main(2)
                self.assertFalse((self.root/'task_02.json').exists())
        self.wave([2]);self.assertEqual(len(list(self.root.glob('task_02_*'))),4)
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            changed=dict(runner.diag.identity(0)['code']);changed['lumopt2_design.py']='0'*64
            with patch.object(runner.diag,'identity',return_value={'code':changed}):
                with self.assertRaisesRegex(RuntimeError,'source hashes'):runner.main(2)


if __name__=='__main__':unittest.main()
