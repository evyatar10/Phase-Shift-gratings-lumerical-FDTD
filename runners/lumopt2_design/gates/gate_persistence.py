"""Failure-injection checks for campaign persistence; no CAD or solver."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import persistence as io
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_te_s1 import SPEC_V3


class PersistenceChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='te_persistence_gate_'))
        self.spec = dataclasses.replace(SPEC_V3, label='persistence_gate')
        self.p = eng.seed_params(self.spec)
        self.row = dict(params=self.p.tolist(), fom=.8, lam_pk_nm=1561.0,
                        fwhm_env_um=19.12, softw_um=18.75)

    def test_strict_width_selector_rejects_wider_historical_s1_rows(self):
        root = Path(__file__).resolve()
        # Candidate bootstrap runs from the production repository cwd.
        historical = Path.cwd()/'results_from_athena/campaign_te_s1/lumopt2_te_s1_v3_evals.jsonl'
        rows = io.read_rows(historical)
        self.assertEqual(len(rows),8)
        path = self.root/(self.spec.label+'_evals.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        old = eng._best_from_log(self.spec,str(self.root),{'fom':-np.inf,'params':self.p},return_row=True)[2]
        self.assertEqual(old,rows[-1])
        strict = dataclasses.replace(self.spec,fwhm_bounds_um=(eng.RHO_DN*19.121,19.121))
        selected = eng._best_from_log(strict,str(self.root),{'fom':-np.inf,'params':self.p},return_row=True)[2]
        self.assertEqual(selected,rows[0])
        self.assertLessEqual(selected['fwhm_env_um'],19.121)

    def test_explicit_hard_width_without_anchor_rejects_missing_and_wide(self):
        spec = dataclasses.replace(self.spec,fwhm0_um=None,fwhm_bounds_um=(19.,19.121))
        rows = [dict(self.row,fom=.7,fwhm_env_um=19.12),
                dict(self.row,fom=.8,fwhm_env_um=19.3),dict(self.row,fom=.9,fwhm_env_um=None)]
        path = self.root/(spec.label+'_evals.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        selected = eng._best_from_log(spec,str(self.root),{'fom':-np.inf,'params':self.p},return_row=True)[2]
        self.assertEqual(selected,rows[0])

    def test_interrupted_tail_read_append_archive_and_restart(self):
        path = self.root/(self.spec.label+'_evals.jsonl')
        raw = (json.dumps(self.row)+'\n').encode()+b'{"params": [1, 2,'
        path.write_bytes(raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            best, lam = eng._best_from_log(self.spec, str(self.root),
                                            {'fom': -np.inf, 'params': self.p})
            np.testing.assert_equal(best, self.p)
            self.assertEqual(lam, 1561.)
            self.assertEqual(path.read_bytes(), raw)
            newer = dict(self.row, fom=.81, lam_pk_nm=1561.1)
            io.append_row(path, newer)
        self.assertEqual(list(self.root.glob('*.interrupted_*'))[0].read_bytes(), raw)
        self.assertEqual(io.read_rows(path), [self.row, newer])
        self.assertEqual(eng._row_of_params(self.spec, str(self.root), self.p), newer)

    def test_completed_line_corruption_is_fatal_and_preserved(self):
        for raw in (b'{"fom":\n', b'{bad}\n'+json.dumps(self.row).encode()):
            path = self.root/'corrupt.jsonl'
            path.write_bytes(raw)
            with self.assertRaises(RuntimeError):
                io.read_rows(path)
            with self.assertRaises(RuntimeError):
                io.append_row(path, self.row)
            self.assertEqual(path.read_bytes(), raw)

    def test_complete_final_record_without_newline_is_retained(self):
        path = self.root/'journal.jsonl'
        path.write_bytes(json.dumps(self.row).encode())
        io.append_row(path, dict(self.row, fom=.82))
        self.assertEqual([r['fom'] for r in io.read_rows(path)], [.8, .82])

    def test_state_corruption_never_resets_rejected_history(self):
        self.assertEqual(eng._load_opt_state(str(self.root), self.spec.label), {})
        path = Path(eng._optstate_path(str(self.root), self.spec.label))
        for raw in ('{"ineligible": [', '[]', '{"cap_nm": NaN}',
                    '{"ineligible": "bad"}', '{"n_acc": -1}'):
            path.write_text(raw, encoding='utf8')
            with self.assertRaises(RuntimeError):
                eng._load_opt_state(str(self.root), self.spec.label)
            with self.assertRaises(RuntimeError):
                eng._save_opt_state(str(self.root), self.spec.label, {'cap_nm': 10})
            self.assertEqual(path.read_text(), raw)

    def test_interrupted_state_publication_preserves_prior_state(self):
        old = dict(cap_nm=10., n_acc=2, ineligible=['rejected'])
        new = dict(cap_nm=15., n_acc=3, ineligible=['rejected', 'also rejected'])
        eng._save_opt_state(str(self.root), self.spec.label, old)
        with patch.object(io.os, 'replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                eng._save_opt_state(str(self.root), self.spec.label, new)
        self.assertEqual(eng._load_opt_state(str(self.root), self.spec.label), old)
        self.assertIn(new, [json.loads(p.read_text()) for p in (self.root/'state_history').glob('*.json')])
        eng._save_opt_state(str(self.root), self.spec.label, new)
        self.assertEqual(eng._load_opt_state(str(self.root), self.spec.label), new)

    def test_interrupted_journal_repair_preserves_original(self):
        path = self.root/'evals.jsonl'
        raw = json.dumps(self.row).encode()+b'\n{'
        path.write_bytes(raw)
        with patch.object(io.os, 'replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError):
                io.append_row(path, self.row)
        self.assertEqual(path.read_bytes(), raw)
        self.assertTrue(any(p.read_bytes() == raw for p in self.root.glob('*.interrupted_*')))

    def test_existing_rejection_history_still_filters_best(self):
        path = self.root/(self.spec.label+'_evals.jsonl')
        p_bad = self.p.copy(); p_bad[0] += 1
        io.append_row(path, self.row)
        io.append_row(path, dict(self.row, params=p_bad.tolist(), fom=.99))
        eng._save_opt_state(str(self.root), self.spec.label,
                            dict(ineligible=[eng._param_key(p_bad)], cap_nm=10.))
        selected, _ = eng._best_from_log(self.spec, str(self.root),
                                        {'fom': -np.inf, 'params': self.p})
        np.testing.assert_equal(selected, self.p)

    def test_interruption_before_acceptance_does_not_promote_trial(self):
        path = self.root/(self.spec.label+'_evals.jsonl')
        p_trial = self.p.copy(); p_trial[0] += 1
        io.append_row(path, dict(self.row, eval_id='accepted'))
        io.append_row(path, dict(self.row, params=p_trial.tolist(), fom=.99,
                                eval_id='unclassified'))
        state = dict(cap_nm=10., ineligible=[], accepted_eval_ids=['accepted'])
        eng._save_opt_state(str(self.root), self.spec.label, state)
        selected, _ = eng._best_from_log(self.spec, str(self.root),
                                        {'fom': -np.inf, 'params': self.p})
        np.testing.assert_equal(selected, self.p)
        state['accepted_eval_ids'].append('unclassified')
        eng._save_opt_state(str(self.root), self.spec.label, state)
        selected, _ = eng._best_from_log(self.spec, str(self.root),
                                        {'fom': -np.inf, 'params': self.p})
        np.testing.assert_equal(selected, p_trial)

    def test_newest_failed_measurement_cannot_borrow_older_width(self):
        path = self.root/(self.spec.label+'_evals.jsonl')
        io.append_row(path, dict(self.row, eval_id='old'))
        fresh = dict(self.row, eval_id='fresh', fwhm_env_um=None,
                     diag_error='simulated failed profile extraction')
        io.append_row(path, fresh)
        self.assertIsNone(eng._row_of_params(self.spec, str(self.root), self.p,
                                            need=('fwhm_env_um',)))
        self.assertEqual(eng._row_of_params(self.spec, str(self.root), self.p), fresh)

    def test_old_profile_paths_remain_readable_and_new_missing_is_not_aliased(self):
        old = io.profile_path(self.root, self.spec.label, {'eval': 0}, 7)
        self.assertTrue(old.endswith('_ev0007.npz'))
        self.assertIsNone(io.profile_path(self.root, self.spec.label, {'eval_id': 'new'}, 7))
        self.assertIsNone(eng.profile_mac(None, old))
        with self.assertRaises(ValueError):
            io.profile_path(self.root, self.spec.label, {'profile_file': '../foreign.npz'}, 7)

    def test_reported_best_fom_belongs_to_reported_vector(self):
        p_last = self.p.copy(); p_last[0] += 1
        path = self.root/(self.spec.label+'_evals.jsonl')
        io.append_row(path, dict(self.row, fom=.9))
        io.append_row(path, dict(self.row, fom=.8, params=p_last.tolist()))
        with patch.object(eng, 'import_lumopt2', return_value=object()), \
             patch.object(eng, 'make_project', return_value=(
                 types.SimpleNamespace(_queue_forward_jobs=lambda: None), None)), \
             patch.object(eng, 'make_log_callback', return_value=object()), \
             patch.object(eng, 'run_projected', return_value=(p_last, .8)):
            result = eng.run_campaign(self.spec, str(self.root))
        self.assertEqual(result['fom'], .9)
        np.testing.assert_equal(result['params'], self.p)
        saved = json.loads((self.root/(self.spec.label+'_best.json')).read_text())
        self.assertEqual(saved['best_fom'], .9)
        np.testing.assert_equal(saved['best_params_nm'], self.p)

    def test_real_callback_repeated_eval_zero_retains_both_profiles(self):
        callbacks = types.ModuleType('lumopt2.utils.callbacks'); callbacks.BaseCallback = object
        modules = {'lumopt2': types.ModuleType('lumopt2'),
                   'lumopt2.utils': types.ModuleType('lumopt2.utils'),
                   'lumopt2.utils.callbacks': callbacks}
        wl = np.linspace(1556., 1566., 101)
        T = .9/(1+((wl-1561.)/.5)**2)
        x = np.linspace(-49, 49, 501)
        I = np.exp(-4*np.log(2)*x*x/19.12**2)

        class Fdtd:
            def getresult(self, name, key):
                if name.endswith('Port_2'):
                    return {'S': np.sqrt(T), 'T_out': .95*T, 'lambda': wl*1e-9}
                if name.endswith('Port_1'):
                    return {'S': np.full(len(wl), .02), 'lambda': wl*1e-9}
                raise KeyError(name)

        spec = dataclasses.replace(self.spec, width_grad=False, wg_lam_chain=False,
                                   two_kl_floor=0.)
        proj = types.SimpleNamespace(load_forward_results=lambda: None,
                                     fdtd_session=types.SimpleNamespace(fdtd=Fdtd()))
        with patch.dict(sys.modules, modules), patch.object(eng, 'profile_line', return_value=(x, I)):
            eng.make_log_callback(spec, str(self.root), lmpt=object()).on_function_eval(
                proj, 0, self.p, .8)
        path = self.root/(spec.label+'_evals.jsonl')
        first = io.read_rows(path)[0]
        a = Path(io.profile_path(self.root, spec.label, first, 0)); raw = a.read_bytes()
        with patch.dict(sys.modules, modules), patch.object(eng, 'profile_line', return_value=(x, I[::-1]*2)):
            eng.make_log_callback(spec, str(self.root), lmpt=object()).on_function_eval(
                proj, 0, self.p, .81)
        second = io.read_rows(path)[-1]
        b = Path(io.profile_path(self.root, spec.label, second, 0))
        self.assertNotEqual(a, b)
        self.assertEqual(a.read_bytes(), raw)
        self.assertEqual(len(list((self.root/'profiles').glob('*.npz'))), 2)
        self.assertAlmostEqual(eng.profile_mac(str(a), str(b)), 1.)
        self.assertEqual(eng._row_of_params(spec, str(self.root), self.p)['eval_id'], second['eval_id'])


if __name__ == '__main__':
    unittest.main()
