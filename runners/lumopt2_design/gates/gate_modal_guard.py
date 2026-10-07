"""Offline pilot acceptance/selection regressions using the real driver."""
import dataclasses
import json
from pathlib import Path
import tempfile
import types
import unittest
import sys
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import persistence
from runners.lumopt2_design.campaign_te_s1 import SPEC_V3


class Project:
    def __init__(self, script, p):
        self.script, self.n, self.gradients = script, 0, 0
        self.gT = np.zeros_like(p); self.gT[0] = 1e-3
        gW = np.zeros_like(p); gW[1] = 1e-3
        gLam = np.zeros_like(p); gLam[2] = 1e-3
        self.fom = types.SimpleNamespace(gfields_W='W', gfields_Tlo='lo', gfields_Thi='hi')
        self.fdtd_session = None
        gs = {'W': gW, 'lo': np.zeros_like(p), 'hi': gLam}
        self.parametrization = types.SimpleNamespace(
            compute_gradient_from_fields=lambda f, sess, params: gs[f].copy())

    def compute_fom(self, p):
        row = self.script[self.n]; self.n += 1
        return row['fom']

    def compute_gradient(self, p):
        self.gradients += 1
        return self.gT.copy()


class Log:
    def __init__(self, spec, folder, script):
        self.spec, self.folder, self.script = spec, folder, script

    def on_function_eval(self, project, it, p, fom):
        self.spec._wg_dTp, self.spec._wg_lam_idx = -1., (0, 1)
        e = self.script[it]
        row = dict(e, eval=it, eval_id=persistence.new_id(), fom=fom,
                   params=p.tolist(), lam_pk_nm=self.spec.scan_center_nm,
                   softw_um=e['fwhm_env_um']-.4, softw_adj_um=e['fwhm_env_um']-.3,
                   cw_um_per_nm=0., cw_curved=False, twin_lam_nm=self.spec.scan_center_nm)
        # Deliberately permit nonfinite legacy values in this fixture so the
        # driver check is exercised beyond the writer's independent guard.
        with (self.folder/(self.spec.label+'_evals.jsonl')).open('a') as stream:
            stream.write(json.dumps(row)+'\n')


def measurement(fom, modal, width=19.12):
    row = dict(fom=fom, fwhm_env_um=width)
    if modal != 'missing':
        row['modal_peak3'] = modal
    return row


class ModalGuardChecks(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='te_modal_guard_gate_'))
        self.spec = dataclasses.replace(SPEC_V3, label='modal_gate',
            wgp_modal_guard=True, best_metric='modal_peak3', wgp_mode_mac=None,
            wgp_noise_freeze=False, wgp_cap_adapt=True,
            fwhm_bounds_um=(18.73858, 19.121), wgp_v3_band_um=(18.9, 19.12))
        self.p = eng.seed_params(self.spec)

    def drive(self, script, p=None):
        self.spec.max_iter = len(script)
        project = Project(script, self.p)
        eng.run_projected(self.spec, project, Log(self.spec, self.folder, script),
                          str(self.folder), self.p if p is None else p)
        rows = persistence.read_rows(self.folder/(self.spec.label+'_proj.jsonl'))
        state = eng._load_opt_state(str(self.folder), self.spec.label)
        return rows, state, project

    def journal(self, rows, accepted):
        (self.folder/(self.spec.label+'_evals.jsonl')).write_text(
            ''.join(json.dumps(row)+'\n' for row in rows),encoding='utf8')
        eng._save_opt_state(str(self.folder), self.spec.label,
                           dict(accepted_eval_ids=accepted, ineligible=[]))

    def row(self, identifier, fom, modal, width=19.12, delta=0):
        p = self.p.copy(); p[0] += delta
        return dict(params=p.tolist(), fom=fom, modal_peak3=modal,
                    eval_id=identifier, fwhm_env_um=width,
                    softw_um=18.75, lam_pk_nm=self.spec.scan_center_nm)

    def selected(self, fallback_fom=-np.inf):
        return eng._best_from_log(self.spec, str(self.folder),
            {'fom':fallback_fom, 'params':self.p}, return_row=True)[2]

    def test_tout_ascent_with_modal_loss_rejects(self):
        rows, state, project = self.drive([
            measurement(.7,.9), measurement(.8,.89), measurement(.75,.905)])
        self.assertTrue(rows[1]['modal_reject'])
        self.assertFalse(rows[2]['modal_reject'])
        self.assertEqual(project.gradients, 2)
        self.assertEqual(len(state['accepted_eval_ids']), 2)
        self.assertEqual(state['modal_best'], .905)

    def test_rejected_high_modal_does_not_poison(self):
        rows, state, _ = self.drive([measurement(.7,.9),
            measurement(.9,.99,19.3), measurement(.75,.91)])
        self.assertIn('v3_width_reject',rows[1])
        self.assertFalse(rows[2]['modal_reject'])
        self.assertEqual(state['modal_best'], .91)
        self.assertEqual(self.selected()['modal_peak3'],.91)

    def test_successive_dips_never_ratchet(self):
        rows, state, project = self.drive([measurement(.7,.9),
            measurement(.8,.8999), measurement(.81,.8998), measurement(.82,.8997)])
        self.assertTrue(all(r['modal_reject'] for r in rows[1:]))
        self.assertEqual(project.gradients,1)
        self.assertEqual(state['modal_best'],.9)
        self.assertEqual(len(state['accepted_eval_ids']),1)

    def test_cold_restart_reconstructs_committed_feasible_reference(self):
        self.drive([measurement(.7,.9)])
        state = eng._load_opt_state(str(self.folder),self.spec.label)
        state['modal_best'] = .999  # stale scalar must not override measured rows
        eng._save_opt_state(str(self.folder),self.spec.label,state)
        rows,state,project = self.drive([measurement(.72,.9),measurement(.8,.89)])
        self.assertEqual(rows[-1]['modal_reference'],.9)
        self.assertTrue(rows[-1]['modal_reject'])
        self.assertEqual(state['modal_best'],.9)
        self.assertEqual(project.gradients,1)

    def test_infeasible_restoration_can_lower_modal_without_poisoning(self):
        rows,state,_ = self.drive([measurement(.9,.99,19.3),
                                   measurement(.8,.9,19.12),measurement(.85,.895)])
        self.assertIsNone(rows[1]['modal_reference'])
        self.assertTrue(rows[1]['modal_restoration'])
        self.assertTrue(rows[2]['modal_reject'])
        self.assertEqual(state['modal_best'],.9)

    def test_strict_width_committed_ids_and_modal_ranking(self):
        rows=[self.row('tout',.95,.90,delta=1), self.row('modal',.8,.94,delta=2),
              self.row('wide',.99,.98,width=19.2,delta=3),
              self.row('unclassified',.99,.99,delta=4),
              self.row('hop',.99,.995,delta=5),
              self.row('legacy',.99,.999,delta=6)]
        rows[-1].pop('eval_id')
        self.journal(rows,['tout','modal','wide','hop'])
        state=eng._load_opt_state(str(self.folder),self.spec.label)
        state['ineligible']=[eng._param_key(rows[4]['params'])]
        eng._save_opt_state(str(self.folder),self.spec.label,state)
        self.assertEqual(self.selected(fallback_fom=100.),rows[1])

    def test_missing_nonfinite_live_modal_fail_closed_before_gradient(self):
        for modal in ('missing',None,np.nan,np.inf):
            folder=Path(tempfile.mkdtemp(prefix='te_modal_missing_'))
            script=[measurement(.7,modal)]
            project=Project(script,self.p)
            with self.assertRaisesRegex(RuntimeError,'modal_peak3'):
                eng.run_projected(self.spec,project,Log(self.spec,folder,script),
                                  str(folder),self.p)
            self.assertEqual(project.gradients,0)
            self.assertEqual(eng._load_opt_state(str(folder),self.spec.label),{})

    def test_missing_historical_modal_is_not_selected(self):
        rows=[self.row('good',.7,.9),self.row('missing',.99,None,delta=1),
              self.row('nan',.99,np.nan,delta=2)]
        self.journal(rows,['good','missing','nan'])
        self.assertEqual(self.selected(),rows[0])

    def test_paired_final_report_keeps_carrier_and_modal_from_same_row(self):
        rows=[self.row('tout',.95,.9,delta=1),self.row('modal',.8,.94,delta=2)]
        self.journal(rows,['tout','modal'])
        with patch.object(eng,'import_lumopt2',return_value=object()), \
             patch.object(eng,'make_project',return_value=(
                 types.SimpleNamespace(_queue_forward_jobs=lambda:None),None)), \
             patch.object(eng,'make_log_callback',return_value=object()), \
             patch.object(eng,'run_projected',return_value=(self.p,.95)):
            result=eng.run_campaign(self.spec,str(self.folder))
        self.assertEqual(result['fom'],.8)
        self.assertEqual(result['modal_peak3'],.94)
        np.testing.assert_equal(result['params'],rows[1]['params'])
        out=json.loads((self.folder/(self.spec.label+'_best.json')).read_text())
        self.assertEqual((out['best_fom'],out['best_modal_peak3']),(.8,.94))
        self.assertEqual(out['best_eval_id'],'modal')

    def test_defaults_retain_carrier_ranking_and_do_not_require_modal(self):
        self.spec.wgp_modal_guard=False; self.spec.best_metric='fom'
        rows=[self.row('tout',.95,None,delta=1),self.row('modal',.8,.94,delta=2)]
        self.journal(rows,['tout','modal'])
        self.assertEqual(self.selected(),rows[0])


if __name__=='__main__':
    unittest.main()
