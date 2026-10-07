"""Offline failure/restart checks for one-solve S2 calibration tasks."""
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import recover_te_s2 as r
from runners.lumopt2_design import recover_te_s2_peak as peak


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='te_recovery_gate_'))

    def test_atomic_checkpoint_and_identity(self):
        path = self.root/'done.json'
        expected = r.identity(1)
        self.assertIsNone(r.read_checkpoint(path, expected))
        r.checkpoint(path, expected, {'fom': .4})
        self.assertEqual(r.read_checkpoint(path, expected), {'fom': .4})
        with self.assertRaises(FileExistsError):
            r.checkpoint(path, expected, {'fom': .5})
        with self.assertRaises(RuntimeError):
            r.read_checkpoint(path, r.identity(2))
        with self.assertRaises(ValueError):
            r.checkpoint(self.root/'nan.json', expected, {'fom': float('nan')})
        self.assertFalse((self.root/'nan.json').exists())

    def test_partial_file_is_not_a_checkpoint(self):
        (self.root/'done.json.partial').write_text('{')
        self.assertIsNone(r.read_checkpoint(self.root/'done.json', r.identity(1)))
        (self.root/'done.json').write_text('{')
        with self.assertRaises(json.JSONDecodeError):
            r.read_checkpoint(self.root/'done.json', r.identity(1))

    def test_all_legs_within_bounds(self):
        r.eng._assert_fd_legs_in_bounds(r.SPEC, r.val.te_point(r.SPEC),
                                      r.INDICES, r.PERTURBATION)

    def test_resume_skips_solver_import(self):
        root = self.root/r.SPEC.label
        root.mkdir()
        r.checkpoint(root/'task_01.json', r.identity(1), {'fom': .4})
        with patch.object(r.config, 'RESULTS_DIR', str(self.root)), \
             patch.object(r.eng, 'import_lumopt2', side_effect=AssertionError('must not start CAD')):
            self.assertEqual(r.main(1)['fom'], .4)

    def test_fd_one_solve_and_failed_attempt_preserved(self):
        calls = []
        def make_project(spec, folder, lmpt):
            files = Path(folder)/'files'
            files.mkdir()
            target = files/'fwd_default.fsp'
            def save(path):
                self.assertEqual(Path(path), target)
                target.write_text('new attempt')
            f = SimpleNamespace(getresult=lambda *a: np.arange(4),
                                setnamed=lambda *a: calls.append(('disable', a)),
                                save=save, load=lambda path: None)
            def run():
                calls.append(('solve', str(target)))
                if len([c for c in calls if c[0]=='solve']) == 1:
                    raise RuntimeError('simulated preemption')
                target.with_name(target.stem+'_p0.log').write_text('Simulation completed successfully')
            return SimpleNamespace(set_fwd=lambda p: calls.append(('params', p.copy())),
                fdtd_session=SimpleNamespace(fdtd=f, invalidate_loaded_state=lambda: None),
                fom=SimpleNamespace(config_map=SimpleNamespace(project_folder=str(files),
                      config_map={None: {'fwd_filename': 'fwd_default.fsp'}})),
                run_forward=run), None
        with patch.object(r.eng, 'make_project', side_effect=make_project), \
             patch.object(r, 'port_arrays', return_value={'a': np.array([1])}), \
             patch.object(r, 'score', return_value={'fom': .4}):
            with self.assertRaisesRegex(RuntimeError, 'preemption'):
                r.fd_leg(1, self.root, None)
            result = r.fd_leg(1, self.root, None)
        solves = [c[1] for c in calls if c[0]=='solve']
        self.assertEqual(len(solves), 2)
        self.assertNotEqual(*solves)
        self.assertTrue(all(Path(p).exists() for p in solves))
        self.assertEqual(result['sign'], 1)
        self.assertEqual(result['index'], 0)

    def test_fit_requires_all_legs(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete calibration'):
            r.fit(self.root)

    def test_solver_success_is_required(self):
        fsp = self.root/'bad.fsp'
        fsp.with_name('bad_p0.log').write_text('2.8% complete')
        with self.assertRaisesRegex(RuntimeError, 'No successful solve'):
            r.successful_log(fsp)

    def test_peak_fit_uses_peak_values_and_rejects_wrong_gradient(self):
        legacy = self.root/'recover_te_s2/results'/r.SPEC.label
        legacy.mkdir(parents=True)
        root = self.root/'peak'
        root.mkdir()
        re = np.array([1., 3., 2., 5., 4., 2.])*1e-3
        im = np.array([2., -1., 1., -2., 3., 4.])*1e-3
        fd = .94*re+.12*im
        for i in range(1, 13):
            sign = 1 if i % 2 else -1
            r.checkpoint(legacy/f'task_{i:02d}.json', r.identity(i),
                         dict(fom=100., peak3_tout=.7+sign*r.PERTURBATION*fd[(i-1)//2]))
        for i, g in enumerate((re, im)):
            r.checkpoint(root/f'task_{i:02d}.json', peak.identity(i), dict(gradient=g.tolist()))
        with patch.object(r.config, 'BASE_SAVE_DIR', str(self.root)):
            result = peak.fit(root)
            self.assertTrue(result['passed'])
            np.testing.assert_allclose(result['coefficients'], [.94, .12], atol=1e-12)
            # A separate root preserves the good fixture and exposes a bad class.
            bad = self.root/'bad_peak'; bad.mkdir()
            re_bad = re.copy(); re_bad[0] *= -5
            for i, g in enumerate((re_bad, im)):
                r.checkpoint(bad/f'task_{i:02d}.json', peak.identity(i), dict(gradient=g.tolist()))
            self.assertFalse(peak.fit(bad)['passed'])

    def test_peak_reweights_saved_fields_without_solving_or_saving(self):
        created = []
        wl = np.linspace(1559.9, 1561.9, 501)
        def make_project(spec, folder, lmpt):
            cm = SimpleNamespace(config_map={None: {'adjoints': [{}]}},
                                 _sim_result_info={1: {}})
            def fields(**kw):
                self.assertIn('cport_', cm.config_map[None]['fwd_filename'])
                self.assertEqual(cm._sim_result_info[1]['adj_filename'],
                                 cm.config_map[None]['adjoints'][0]['adj_filename'])
                return np.array([123.])
            par = SimpleNamespace(update_structure=lambda *a: None,
                    compute_gradient_from_fields=lambda fields, session, p: np.arange(len(p)))
            project = SimpleNamespace(generate=lambda: None, fom_symmetry_factors=[1],
                parametrization=par,
                fdtd_session=SimpleNamespace(fdtd=SimpleNamespace(switchtolayout=lambda: None),
                                             invalidate_loaded_state=lambda: None),
                fom=SimpleNamespace(config_map=cm, calculate_gradient_fields=fields))
            # There are deliberately no runner, set_fwd, save or compute_gradient APIs.
            created.append(project)
            return project, wl
        with patch.object(r.eng, 'make_project', side_effect=make_project), \
             patch.object(r.eng, 'import_lumopt2', return_value=None), \
             patch.object(r, 'successful_log'):
            result = peak.gradient(0, self.root)
        self.assertEqual(result['gradient'], r.INDICES)
        t = .9/(1+((wl-1560.9)/.1)**2)
        self.assertAlmostEqual(created[0].fom.fct(t), .9, places=8)
        self.assertLess(r.eng.make_fct(wl)(t), .85)


if __name__ == '__main__':
    unittest.main()
