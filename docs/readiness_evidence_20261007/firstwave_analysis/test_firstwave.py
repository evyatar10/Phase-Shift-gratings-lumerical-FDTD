"""Offline analytic and tampered first-wave evidence controls; retain fixtures."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile
sys.dont_write_bytecode=True
import numpy as np
import analyze_firstwave as a

REPO=Path(__file__).resolve().parents[3]
EVIDENCE=REPO/'docs/readiness_evidence_20261007'
ARCHIVE=EVIDENCE/'s2_original_port_spectra_preserved.zip'
WATCH=REPO/'results_from_athena/recover_te_s2_20261006/watch/recover_te_s2'
sys.path.insert(0,str(REPO))
from runners.lumopt2_design import recover_te_s2 as recovery


class FirstWaveChecks(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='fixture_',dir=None))
        for name in list(a.EXISTING)+['s2_half_classes_first_review_original.json','s2_reference_task00_remote_original.json']:
            shutil.copyfile(EVIDENCE/name,self.root/name)
        self.pins=a.build_pins(self.root,ARCHIVE,WATCH)
        self.center=json.loads((self.root/'s2_fdstep_task00_original.json').read_text())
        self.marker=json.loads((self.root/'s2_half_classes_first_review_original.json').read_text())
        with zipfile.ZipFile(ARCHIVE) as z:
            base=a.h.load_npz(z.read('base_ports.npz'))
            oldshift=[a.h.load_npz(z.read(f'task_{t:02d}_ports.npz')) for t in (9,10)]
        review=dict(path=a.MARKER_REMOTE,sha256=a.MARKER_SHA,stage='first',
            provenance_report=self.marker['provenance_report'],provenance_report_sha256=self.marker['provenance_report_sha256'],allowed_tasks=[0,1,2,3])
        for task in range(4):
            label,index,step,sign=a.TASK_MAP[task]
            if task<2:
                arrays=a.h.load_npz((self.root/f's2_fdstep_task{task+1:02d}_ports.npz').read_bytes())
            else:
                # Synthetic first-wave spectra ONLY; no new solve is claimed.
                arrays={k:v.copy() for k,v in base.items()}
                endpoint=oldshift[task-2]
                for key in ('p2_T_out','p2_S'):
                    arrays[key]=base[key]+.25*(endpoint[key]-base[key])
            ports=self.root/f's2_half_classes_task{task:02d}_ports.npz'
            with ports.open('xb') as stream:np.savez(stream,**arrays)
            point=np.asarray(self.center['identity']['point']).copy();point[index]+=sign*step
            spec=copy.deepcopy(self.center['identity']['spec']);spec['label']=a.FAMILY
            attempt=self.marker['output_root']+f'/task_{task:02d}_synthetic_fixture'
            source=attempt+'/'+a.FAMILY+'_files/fwd_default.fsp'
            ident=dict(schema=1,task=task,spec=spec,point=point.tolist(),index=index,class_label=label,sign=sign,
                perturbation_nm=step,task_map=a.TASK_MAP,center_sha256=a.CENTER_SHA,center_checkpoint=a.CENTER_REMOTE,
                parent_review=review,code=self.marker['code'])
            result=dict(task=task,index=index,class_label=label,sign=sign,perturbation_nm=step,delta_nm=step*sign,
                params=point.tolist(),source=source,h5=source[:-4]+'/fwd_default_output.h5',ports_npz=attempt+'/ports.npz',
                ports_sha256=a.sha(ports),reviewed_center_sha256=a.CENTER_SHA,parent_review=review,
                existing_overall_gradient_gate='FAILED_UNCHANGED',comparison_status='REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS',
                **recovery.score(arrays))
            (self.root/f's2_half_classes_task{task:02d}_original.json').write_text(json.dumps(dict(identity=ident,result=result)))

    def row(self,task=0):return json.loads((self.root/f's2_half_classes_task{task:02d}_original.json').read_text())
    def put(self,row,task=0):(self.root/f's2_half_classes_task{task:02d}_original.json').write_text(json.dumps(row))
    def analyze(self,pins=None):return a.analyze(self.root,pins or self.pins,REPO)

    def test_complete_fixture_preserves_status_and_observed_repeat(self):
        report,spectra=self.analyze()
        self.assertEqual(report['diagnostic_status'],'REVIEW_REQUIRED_CALIBRATION_FAILED_UNCHANGED')
        self.assertTrue(report['no_automatic_later_approval'])
        self.assertTrue(report['provenance']['old_shift30']['available'])
        for metric in ('peak3_tout','peak3_modal'):
            self.assertEqual(report['observables'][metric]['corr30_repeat']['central_derivative_difference'],0.)
            self.assertEqual(report['observables'][metric]['corr30_repeat']['plus_spectral_difference']['max_abs'],0.)
            self.assertEqual(len(spectra[metric]['shift30_half_spectral_fd']),501)
        output=self.root/'report.json';a.write(output,report)
        with self.assertRaises(FileExistsError):a.write(output,report)

    def test_task_sign_point_spec_source_marker_identity_tamper(self):
        original=self.row()
        alterations=[('identity','task',1),('identity','sign',-1),('identity','point',[0.]*296),
            ('result','params',[0.]*296),('result','source','/old/fwd_default.fsp'),('result','ports_sha256','0'*64)]
        for outer,key,value in alterations:
            row=copy.deepcopy(original);row[outer][key]=value;self.put(row)
            with self.assertRaises(a.h.EvidenceError):self.analyze()
        for inner,key,value in [('spec','dp_tooth_nm',.25),('code','lumopt2_design.py','0'*64),('parent_review','sha256','0'*64)]:
            row=copy.deepcopy(original);row['identity'][inner][key]=value;self.put(row)
            with self.assertRaises(a.h.EvidenceError):self.analyze()

    def test_grid_tamper_even_when_npz_digest_updated(self):
        path=self.root/'s2_half_classes_task02_ports.npz'
        arrays=a.h.load_npz(path.read_bytes());arrays['p2_lambda']=arrays['p2_lambda']+1e-15
        with path.open('wb') as stream:np.savez(stream,**arrays)
        row=self.row(2);row['result']['ports_sha256']=a.sha(path);self.put(row,2)
        with self.assertRaisesRegex(a.h.EvidenceError,'grid mismatch'):self.analyze()

    def test_scalar_tamper_and_exact_required_inputs(self):
        row=self.row(3);row['result']['peak3_tout']+=1e-8;self.put(row,3)
        with self.assertRaisesRegex(a.h.EvidenceError,'scalar not reproduced'):self.analyze()
        self.assertFalse((self.root/'not_automatically_written.json').exists())
        with self.assertRaises(FileNotFoundError):a.analyze(self.root/'absent',self.pins,REPO)

    def test_optional_old_shift_unavailable_and_raw_pin_tamper(self):
        pins=copy.deepcopy(self.pins);pins['old_shift30']=dict(available=False,reason='Explicitly not provided')
        report,spectra=self.analyze(pins)
        self.assertFalse(report['observables']['peak3_tout']['shift30_old_h2']['available'])
        self.assertNotIn('shift30_h2_spectral_fd',spectra['peak3_tout'])
        pins=copy.deepcopy(self.pins);pins['old_shift30']['archive_sha256']='0'*64
        with self.assertRaisesRegex(a.h.EvidenceError,'archive identity'):self.analyze(pins)

    def test_analytic_odd_even_and_peak_jacobian(self):
        x=np.arange(9)-4;center=2-.05*x*x
        # Multiplicative amplitude has exact peak FD and fixed Jacobian projection.
        step=.5;plus=center*(1+.2*step+.03*step*step);minus=center*(1-.2*step+.03*step*step)
        result,spectral=a.pair_report(plus,minus,center,step)
        self.assertAlmostEqual(result['central_derivative'],.4,places=13)
        self.assertAlmostEqual(result['even'],.015,places=13)
        self.assertAlmostEqual(result['exact_center_peak3_jacobian_projection'],.4,places=13)


if __name__=='__main__':unittest.main()
