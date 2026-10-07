"""Offline surgical tests only. Retain fixtures in this unique temp package."""
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import s2_corr30_convergence as runner


class StepChecks(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='s2_convergence_fixture_'))
        self.root = self.temp / runner.LABEL
        self.center_path = self.temp/'center'/'task_00.json'
        self.center_path.parent.mkdir()
        wl = np.linspace(runner.SPEC.scan_center_nm-1, runner.SPEC.scan_center_nm+1, 501)
        line = 1/(1+((wl-runner.SPEC.scan_center_nm)/.1)**2)
        self.arrays = dict(p2_lambda=wl*1e-9, p2_T_out=.93*line, p2_S=np.sqrt(.97*line), p1_T_in=np.full(501,.94))
        metrics = runner.recovery.score(self.arrays)
        files = {}
        for key in ('source', 'h5', 'ports_npz'):
            path = self.center_path.parent/key
            path.write_bytes(b'preserved fixture')
            files[key] = str(path)
        point = runner.val.te_point(runner.SPEC).tolist()
        self.center = dict(identity=dict(runner.diag.identity(0)), result=dict(task=0, sign=0, params=point, **files, **metrics))
        self.center_path.write_text(json.dumps(self.center))
        self.center_hash = runner.digest(self.center_path)
        (self.center_path.parent/'center_reviewed.sha256').write_text(self.center_hash)
        self.provenance = self.temp/'provenance.json'
        self.provenance.write_text(json.dumps(dict(critical_evidence_available=True,original_file_identity_unchanged=True)))
        self.marker = self.temp/'parent_review.json'
        self.env = dict(S2_CONV_CENTER_CHECKPOINT=str(self.center_path), S2_CONV_OUTPUT_ROOT=str(self.root),
                        S2_CONV_REVIEW_MARKER=str(self.marker), KEEP_H5='1')
        self.calls = []
        self.envpatch = patch.dict(os.environ, self.env)
        self.hashpatch = patch.object(runner, 'CENTER_SHA256', self.center_hash)
        self.envpatch.start(); self.hashpatch.start()
        self.addCleanup(self.envpatch.stop); self.addCleanup(self.hashpatch.stop)
        review = runner.review_template(self.root, runner.load_center(), self.provenance)
        review.update(status=runner.APPROVAL_STATUS, allowed_tasks=[0,1,2,3])
        self.marker.write_text(json.dumps(review))

    def fake_project(self, spec, attempt, api, mutate_mesh=False, missing_h5=False, fail_solve=False, escape=False):
        source = (self.temp/'escape_fixture' if escape else Path(attempt)/'files')/'fwd_default.fsp'; source.parent.mkdir(exist_ok=True)
        record = dict(forwards=0, closed=False, disabled=False, params=None)
        self.calls.append(record)
        class Fdtd:
            def getresult(inner, name, key):
                if name == 'FDTD':
                    grid = np.arange(4.)
                    return grid+1 if mutate_mesh and record['disabled'] and key=='x' else grid
                port = 2 if name.endswith('Port_2') else 1
                return {k.split('_',1)[1]:v for k,v in self.arrays.items() if k.startswith(f'p{port}_')}
            def setnamed(inner, name, key, value):
                self.assertEqual((name,key,value), ('optimization_dft','enabled',0))
                record['disabled'] = True
            def save(inner, path):
                self.assertEqual(Path(path), source); source.write_bytes(b'new setup only')
            def load(inner, path):
                self.assertEqual(Path(path), source)
            def close(inner):
                record['closed'] = True
        def set_fwd(p):
            record['params'] = p.copy(); source.write_bytes(b'fake initial setup')
        def run_forward():
            record['forwards'] += 1
            if fail_solve:
                raise RuntimeError('fake interrupted solve')
            source.with_name(source.stem+'_p0.log').write_text('Simulation completed successfully')
            if not missing_h5:
                folder = source.with_suffix(''); folder.mkdir()
                (folder/(source.stem+'_output.h5')).write_bytes(b'preserved H5')
        def forbidden(*args, **kwargs):
            raise AssertionError('Adjoint/optimizer forbidden')
        project = types.SimpleNamespace(set_fwd=set_fwd, run_forward=run_forward,
            compute_gradient=forbidden, run_adjoint=forbidden,
            fdtd_session=types.SimpleNamespace(fdtd=Fdtd(), invalidate_loaded_state=lambda:None),
            fom=types.SimpleNamespace(config_map=types.SimpleNamespace(project_folder=str(source.parent),config_map={None:dict(fwd_filename=source.name)})))
        return project, None

    def test_taskmap_only_corr30_bounds_and_dependency_freeze(self):
        self.assertEqual(runner.N_TASKS,4)
        base = runner.val.te_point(runner.SPEC)
        for task, delta in enumerate((1.,-1.,.25,-.25)):
            p,h,sign = runner.task_params(task)
            np.testing.assert_array_equal(np.flatnonzero(p-base),[29])
            self.assertEqual((p-base)[29],delta)
            self.assertEqual(h*sign,delta)
        self.assertEqual(runner.dependency_hashes()[Path(runner.__file__).name],runner.digest(runner.__file__))
        self.assertEqual((runner.SPEC.polarization,runner.SPEC.n_periods_side,runner.SPEC.n_free,runner.SPEC.bare),('TE',98,60,True))
        self.assertFalse(runner.SPEC.width_grad or runner.SPEC.wg_project)
        self.assertEqual(dataclasses.asdict(runner.SPEC)['mesh_refinement'],'precise volume average')
        for task in (-1,4,True,1.2,'0'):
            with self.assertRaises(ValueError): runner.main(task)

    def test_missing_unapproved_wrongtask_and_incomplete_provenance_block_before_cad(self):
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            with patch.dict(os.environ,{'S2_CONV_REVIEW_MARKER':str(self.temp/'absent')}):
                with self.assertRaises(FileNotFoundError): runner.main(0)
            row = json.loads(self.marker.read_text())
            for field, value, pattern in (('status','NOT_REVIEWED','identity/status'),('runner_sha256','0'*64,'identity/status'),('allowed_tasks',[2,3],'not authorized'),('provenance_report_sha256','0'*64,'report hash')):
                altered = dict(row); altered[field]=value; self.marker.write_text(json.dumps(altered))
                with self.assertRaisesRegex(RuntimeError,pattern): runner.main(0)
            self.marker.write_text(json.dumps(row))
            self.provenance.write_text(json.dumps(dict(critical_evidence_available=False,original_file_identity_unchanged=True)))
            row['provenance_report_sha256']=runner.digest(self.provenance); self.marker.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError,'critical evidence'): runner.main(0)

    def test_center_hash_review_marker_and_frozen_source_family_gates(self):
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            (self.center_path.parent/'center_reviewed.sha256').write_text('0'*64)
            with self.assertRaisesRegex(RuntimeError,'reviewed-center'): runner.main(0)
            (self.center_path.parent/'center_reviewed.sha256').write_text(self.center_hash)
            self.center_path.write_bytes(self.center_path.read_bytes()+b' ')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'): runner.main(0)
            self.center_path.write_text(json.dumps(self.center))
            changed = dict(runner.diag.identity(0)['code']); changed['lumopt2_design.py']='0'*64
            with patch.object(runner.diag,'identity',return_value={'code':changed}):
                with self.assertRaisesRegex(RuntimeError,'source hashes'): runner.main(0)
            prior = runner.SPEC
            with patch.object(runner,'SPEC',dataclasses.replace(prior,region_dx_nm=prior.region_dx_nm*1.01)):
                with self.assertRaisesRegex(RuntimeError,'numerical family'): runner.main(0)

    def test_four_fake_forwards_preserve_and_resume_with_no_new_solve(self):
        original = self.center_path.read_bytes()
        with patch.object(runner.eng,'import_lumopt2',return_value=object()),patch.object(runner.eng,'make_project',side_effect=self.fake_project):
            for task in range(4):
                result = runner.main(task)
                self.assertEqual(result['existing_overall_gradient_gate'],'FAILED_UNCHANGED')
                self.assertEqual(result['comparison_status'],'REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS')
                runner.verify_result(result)
                checkpoint = self.root/f'task_{task:02d}.json'
                before = checkpoint.read_bytes()
                with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('repeat solve')):
                    self.assertEqual(runner.main(task),result)
                self.assertEqual(checkpoint.read_bytes(),before)
                with self.assertRaises(FileExistsError): runner.recovery.checkpoint(checkpoint,json.loads(before)['identity'],result)
        self.assertEqual(self.center_path.read_bytes(),original)
        self.assertEqual(len(self.calls),4)
        self.assertTrue(all(r['forwards']==1 and r['closed'] for r in self.calls))
        self.assertEqual([r['params'][29]-runner.val.te_point(runner.SPEC)[29] for r in self.calls],[1.,-1.,.25,-.25])

    def test_keep_h5_mesh_drift_missing_h5_and_cold_restart(self):
        with patch.dict(os.environ,{'KEEP_H5':'0'}),patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            with self.assertRaisesRegex(RuntimeError,'KEEP_H5=1'): runner.main(0)
        with patch.object(runner.eng,'import_lumopt2',return_value=object()):
            for kwargs, pattern in ((dict(mutate_mesh=True),'changed x mesh'),(dict(escape=True),'escaped unique'),(dict(missing_h5=True),'HDF5'),(dict(fail_solve=True),'interrupted')):
                with patch.object(runner.eng,'make_project',side_effect=lambda *args,kw=kwargs:self.fake_project(*args,**kw)):
                    with self.assertRaisesRegex(RuntimeError,pattern): runner.main(0)
                self.assertFalse((self.root/'task_00.json').exists())
            with patch.object(runner.eng,'make_project',side_effect=self.fake_project):
                runner.main(0)
        self.assertEqual(len(list(self.root.glob('task_00_*'))),5)
        self.assertTrue(all(r['closed'] for r in self.calls))

    def test_completed_result_tamper_does_not_relaunch(self):
        with patch.object(runner.eng,'import_lumopt2',return_value=object()),patch.object(runner.eng,'make_project',side_effect=self.fake_project):
            result=runner.main(0)
        Path(result['ports_npz']).write_bytes(b'tampered fixture')
        with patch.object(runner.eng,'import_lumopt2',side_effect=AssertionError('CAD touched')):
            with self.assertRaisesRegex(RuntimeError,'ports hash mismatch'): runner.main(0)


if __name__=='__main__':
    unittest.main()
