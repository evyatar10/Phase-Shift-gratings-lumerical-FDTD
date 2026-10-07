import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import compare_four_steps as calc


class Fixture:
    def __init__(self,repeat=0.,omit=None):
        self.root=Path(tempfile.mkdtemp(prefix='four_step_fixture_'))
        self.archive=self.root/'original.zip'
        point=np.zeros(296);point[29]=580.5
        self.point=point.tolist()
        self.spec={'label':'nominal','n_wl_points':31,'polarization':'TE','n_periods_side':98,'n_free':60}
        self.code={'lumopt2_design.py':calc.sha(b'new'),'diagnose_te_s2_fdstep.py':calc.sha(b'diag')}
        old_code={'lumopt2_design.py':calc.sha(b'old-fd'),
                  'recover_te_s2.py':calc.sha(b'fd-runner')}
        export_code={'lumopt2_design.py':calc.sha(b'old-export'),
                     'recover_te_s2.py':calc.sha(b'exporter')}
        self.marker_path='/work/data/review.json';self.center_path='/work/nominal/task_00.json'
        self.out='/work/convergence'
        self.center=dict(identity=dict(task=0,index=29,sign=0,perturbation_nm=.5,
            code=self.code,spec=self.spec,point=self.point),result=dict(task=0,index=29,
            sign=0,delta_nm=0,params=self.point,**self.scores(self.arrays(0,repeat))))
        self.write_json('s2_fdstep_task00_original.json',self.center)
        self.center_sha=calc.sha((self.root/'s2_fdstep_task00_original.json').read_bytes())
        self.write_arrays('s2_fdstep_task00_ports.npz',self.arrays(0,repeat))
        for i,sign in ((1,1),(2,-1)):
            p=point.copy();p[29]+=sign*.5
            ident=copy.deepcopy(self.center['identity']);ident.update(task=i,sign=sign,point=p.tolist(),reviewed_center_sha256=self.center_sha)
            row=dict(identity=ident,result=dict(task=i,index=29,sign=sign,delta_nm=sign*.5,
                params=p.tolist(),reviewed_center_sha256=self.center_sha,**self.scores(self.arrays(sign*.5,repeat))))
            self.write_json(f's2_fdstep_task{i:02d}_original.json',row)
            self.write_arrays(f's2_fdstep_task{i:02d}_ports.npz',self.arrays(sign*.5,repeat))
        code=dict(self.code,**{'s2_corr30_convergence.py':calc.sha(b'conv'),'config.py':calc.sha(b'cfg')})
        self.marker=dict(schema=1,family='convergence',status='APPROVED_FOR_BOUNDED_FORWARD_STEP_DIAGNOSTIC',
            center_sha256=self.center_sha,runner_sha256=code['s2_corr30_convergence.py'],code=code,
            task_map=calc.TASK_MAP,allowed_tasks=[0,1,2,3],output_root=self.out,
            provenance_report='/work/provenance.json',provenance_report_sha256=calc.sha(b'provenance'))
        self.write_json('corr30_convergence_parent_review.json',self.marker)
        marker_sha=calc.sha((self.root/'corr30_convergence_parent_review.json').read_bytes())
        review=dict(path=self.marker_path,sha256=marker_sha,allowed_tasks=[0,1,2,3],
            provenance_report=self.marker['provenance_report'],provenance_report_sha256=self.marker['provenance_report_sha256'])
        for i,(h,sign) in enumerate(calc.TASK_MAP):
            if i==omit:
                continue
            p=point.copy();p[29]+=h*sign
            raw=self.write_arrays(f's2_corr30_convergence_task{i:02d}_ports.npz',self.arrays(h*sign,repeat))
            spec=dict(self.spec,label='convergence')
            row=dict(identity=dict(task=i,index=29,sign=sign,perturbation_nm=h,
                task_map=calc.TASK_MAP,center_sha256=self.center_sha,center_checkpoint=self.center_path,
                spec=spec,code=code,parent_review=review,point=p.tolist()),
                result=dict(task=i,index=29,sign=sign,perturbation_nm=h,delta_nm=h*sign,
                reviewed_center_sha256=self.center_sha,parent_review=review,params=p.tolist(),
                ports_sha256=calc.sha(raw),source=self.out+f'/task_{i:02d}/fwd.fsp',
                h5=self.out+f'/task_{i:02d}/out.h5',ports_npz=self.out+f'/task_{i:02d}/ports.npz',
                **self.scores(self.arrays(h*sign,repeat))))
            self.write_json(f's2_corr30_convergence_task{i:02d}_original.json',row)
        members={}
        with zipfile.ZipFile(self.archive,'w') as z:
            for name,x in (('base_ports.npz',0.),('task_03_ports.npz',2.),('task_04_ports.npz',-2.)):
                raw=self.npz(self.arrays(x));z.writestr(name,raw);members[name]=calc.sha(raw)
        old_center=dict(identity=dict(task=0,point=self.point,spec=dict(self.spec,label='recovery'),code=export_code),
            result=self.scores(self.arrays(0)))
        old=[old_center]
        for task,sign in ((3,1),(4,-1)):
            p=point.copy();p[29]+=sign*2.
            old.append(dict(identity=dict(task=task,perturbation=2.,point=self.point,
                spec=dict(self.spec,label='recovery'),code=old_code),
                result=dict(index=29,sign=sign,params=p.tolist(),**self.scores(self.arrays(sign*2.)))))
        existing={name:calc.sha((self.root/name).read_bytes()) for name in
            [f's2_fdstep_task{i:02d}_{suffix}' for i in range(3) for suffix in ('original.json','ports.npz')]}
        self.pins=dict(schema=1,existing_files=existing,center_sha256=self.center_sha,
            center_code=self.code,old_archive_sha256=calc.sha(self.archive.read_bytes()),old_members=members,
            old_checkpoint_rows=old,review_marker_sha256=marker_sha,
            center_remote_checkpoint=self.center_path,review_remote_path=self.marker_path,
            old_source_bindings=dict(
                center_export=dict(role='stored_forward_export_no_solve',
                    code=export_code,semantic_sha256=calc.semantic_sha(old_center)),
                fd_endpoints={str(row['identity']['task']):dict(role='new_forward_fd_leg',
                    code=old_code,semantic_sha256=calc.semantic_sha(row)) for row in old[1:]}))

    @staticmethod
    def arrays(p,repeat=0.):
        x=np.arange(-15,16.)
        amp=.7+.01*p+.02*p*p+.003*p**3+.005*p**4+repeat
        shift=.1*p+.03*p*p
        t=amp-.002*(x-shift)**2
        f=(1.92e14+x*1e9)[:,None];lam=299792458./f
        return dict(p1_f=f,p2_f=f,p1_lambda=lam,p2_lambda=lam,
                    p2_T_out=t[:,None],p2_S=np.sqrt(t+.03)[:,None])

    @staticmethod
    def scores(a):
        return {name:calc.peak3(transform(a))[1] for name,transform in calc.METRICS.items()}

    @staticmethod
    def npz(a):
        stream=io.BytesIO();np.savez(stream,**a);return stream.getvalue()

    def write_arrays(self,name,a):
        raw=self.npz(a);(self.root/name).write_bytes(raw);return raw

    def write_json(self,name,row):
        (self.root/name).write_text(json.dumps(row),encoding='utf8')

    def mutate_row(self,task,fn):
        name=f's2_corr30_convergence_task{task:02d}_original.json'
        row=json.loads((self.root/name).read_text());fn(row);self.write_json(name,row)

    def analyze(self):
        return calc.analyze(self.root,self.archive,self.pins)


class FourStepChecks(unittest.TestCase):
    def test_analytic_cubic_quartic_halving_richardson_and_exact_jacobian(self):
        f=Fixture();report=f.analyze()['observables']['peak3_tout']
        for h in (2.,1.,.5,.25):
            row=report['steps'][str(h)]
            self.assertAlmostEqual(row['scalar_with_new_center']['central_derivative'],.01+.003*h*h,places=11)
            self.assertAlmostEqual(row['scalar_with_new_center']['second_difference'],.04+.01*h*h,places=10)
            self.assertAlmostEqual(row['center_projections']['new']['moving_peak_minus_projection'],
                                   2*.002*.1*.03*h*h,places=11)
            self.assertEqual(len(row['full_spectral']['central_derivative']),31)
        for ratio in report['derivative_halving']['observed_ratios']:
            self.assertAlmostEqual(ratio['absolute_ratio'],4.,places=7)
        for value in report['derivative_halving']['conditional_Richardson_assuming_O_h2']:
            self.assertAlmostEqual(value['value'],.01,places=11)

    def test_repeat_center_dependence_is_reported_without_noise_or_pass(self):
        f=Fixture(repeat=.00004);report=f.analyze()
        self.assertEqual(report['verdict'],'DIAGNOSTIC_ONLY_OVERALL_CALIBRATION_REMAINS_FAILED')
        row=report['observables']['peak3_tout']['steps']['0.25']
        difference=row['scalar_with_old_center']['second_difference']-row['scalar_with_new_center']['second_difference']
        self.assertAlmostEqual(difference,2*.00004/.25**2,places=10)
        self.assertTrue(any('No noise bounds' in s for s in report['limitations']))

    def test_swapped_endpoint_checkpoints_fail(self):
        f=Fixture();a=f.root/'s2_corr30_convergence_task00_original.json';b=f.root/'s2_corr30_convergence_task01_original.json'
        ra,rb=a.read_bytes(),b.read_bytes();a.write_bytes(rb);b.write_bytes(ra)
        with self.assertRaisesRegex(calc.EvidenceError,'task/step'):
            f.analyze()

    def test_negative_code_point_marker_and_saved_score_controls(self):
        controls=[('source',lambda r:r['identity']['code'].update({'lumopt2_design.py':calc.sha(b'wrong')})),
                  ('point',lambda r:r['identity']['point'].__setitem__(29,580.8)),
                  ('score',lambda r:r['result'].__setitem__('peak3_tout',.1)),
                  ('center',lambda r:r['identity'].__setitem__('center_sha256','0'*64))]
        for label,mutate in controls:
            with self.subTest(label=label):
                f=Fixture();f.mutate_row(0,mutate)
                with self.assertRaises(calc.EvidenceError):
                    f.analyze()
        f=Fixture();p=f.root/'corr30_convergence_parent_review.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(calc.EvidenceError,'hash mismatch'):
            f.analyze()
        f=Fixture()
        f.pins['old_checkpoint_rows'][1]['identity']['code']['lumopt2_design.py']=calc.sha(b'unexpected-old-fd')
        with self.assertRaisesRegex(calc.EvidenceError,'FD source/content binding'):
            f.analyze()

    def test_modified_npz_archive_and_full_grid_fail(self):
        for kind in ('new','half','grid','archive'):
            with self.subTest(kind=kind):
                f=Fixture()
                if kind=='archive':
                    f.archive.write_bytes(f.archive.read_bytes()+b'changed')
                else:
                    name='s2_corr30_convergence_task00_ports.npz' if kind!='half' else 's2_fdstep_task01_ports.npz'
                    a=calc.load_npz((f.root/name).read_bytes())
                    a['p2_f'][0,0]+=1 if kind=='grid' else 0
                    if kind!='grid':
                        a['p2_T_out'][0,0]+=1e-6  # leaves the selected peak score unchanged
                    raw=f.write_arrays(name,a)
                    if kind=='grid':
                        f.mutate_row(0,lambda r:r['result'].__setitem__('ports_sha256',calc.sha(raw)))
                with self.assertRaises(calc.EvidenceError):
                    f.analyze()

    def test_missing_data_and_exclusive_output(self):
        f=Fixture(omit=3)
        with self.assertRaises(FileNotFoundError):
            f.analyze()
        out=f.root/'answer.json';calc.write_exclusive(out,{'first':True});raw=out.read_bytes()
        with self.assertRaises(FileExistsError):
            calc.write_exclusive(out,{'first':False})
        self.assertEqual(out.read_bytes(),raw)


if __name__=='__main__':
    unittest.main()
