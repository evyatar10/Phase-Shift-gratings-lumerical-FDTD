"""Drive real run_campaign budget paths offline; no CAD, solver or fake ledger.

Requires evaluation_budget.py and its queue hook integrated into this candidate.
Physics/gradient math has separate gates; the scripted projected loop here
controls forward/recenter/failure ordering while run_campaign remains real.
"""
import contextlib
import dataclasses
import io as streams
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import lumopt2_design as eng, persistence as persistence
from runners.lumopt2_design.campaign_te_s1 import SPEC_V3
from runners.lumopt2_design.evaluation_budget import BudgetExhausted


class FakeProject:
    """Model the vendor's forward queue and consumed gradient-cache semantics."""
    def __init__(self, harness):
        self.harness = harness
        self.cached = None

    def _queue_forward_jobs(self):
        h = self.harness
        # This is the solver-launch boundary: count must already be durable.
        state = json.loads(h.budget_path.read_text())
        h.test.assertEqual(state['used'], h.observed_used+1)
        h.test.assertTrue(state['last_reservation_id'])
        h.observed_used = state['used']
        h.forward_launches += 1
        if h.fail_forward:
            raise RuntimeError('simulated solver failure after reservation')

    def compute_fom(self, params):
        try:
            self._queue_forward_jobs()
        except Exception as error:
            if not self.harness.wrap_errors:
                raise
            # Vendor wraps with both context and cause, not just one chain.
            inner = RuntimeError('vendor internal wrapper')
            inner.__context__ = error
            outer = RuntimeError('vendor optimization wrapper')
            raise outer from inner
        self.cached = np.asarray(params).copy()
        return .8+.01*self.harness.forward_launches

    def compute_gradient(self, params):
        cached = self.cached is not None and np.array_equal(self.cached, params)
        self.cached = None
        if not cached:
            # Concurrent-adjoint branch calls the queue directly, not run_forward.
            self._queue_forward_jobs()
        self.harness.adjoint_launches += 1
        return np.zeros_like(params)


class Harness:
    def __init__(self, test, root, spec):
        self.test, self.root, self.spec = test, root, spec
        self.budget_path = root/(spec.label+'_forward_budget.json')
        self.forward_launches = self.adjoint_launches = self.observed_used = 0
        self.fail_forward = self.wrap_errors = False
        self.projects = []

    def make_project(self, spec, out_dir, lmpt):
        project = FakeProject(self)
        self.projects.append(project)
        return project, None

    def make_callback(self, spec, out_dir, *args):
        harness = self

        class Callback:
            def on_function_eval(self, project, it, params, fom):
                eval_id = persistence.new_id()
                row = dict(eval=it, eval_id=eval_id, params=np.asarray(params).tolist(),
                           fom=float(fom), fwhm_env_um=spec.fwhm0_um,
                           lam_pk_nm=spec.scan_center_nm+.02, softw_um=spec.fwhm0_um)
                persistence.append_row(harness.root/(spec.label+'_evals.jsonl'), row)
                state = eng._load_opt_state(out_dir, spec.label)
                state.setdefault('accepted_eval_ids', []).append(eval_id)
                eng._save_opt_state(out_dir, spec.label, state)

        return Callback()


class CampaignBudgetChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='te_campaign_budget_gate_'))
        self.spec = dataclasses.replace(SPEC_V3, label='campaign_budget_gate',
                                        width_grad=False, max_feval=2, max_iter=2)
        self.h = Harness(self, self.root, self.spec)
        self.output = streams.StringIO()

    def run_driver(self, driver, limit=None):
        spec = dataclasses.replace(self.spec,
                                   max_feval=self.spec.max_feval if limit is None else limit)
        with patch.object(eng, 'import_lumopt2', return_value=object()), \
                patch.object(eng, 'make_project', side_effect=self.h.make_project), \
                patch.object(eng, 'make_log_callback', side_effect=self.h.make_callback), \
                patch.object(eng, 'run_projected', side_effect=driver), \
                contextlib.redirect_stdout(self.output):
            return eng.run_campaign(spec, str(self.root))

    @staticmethod
    def one_forward(spec, project, callback, out_dir, params):
        fom = project.compute_fom(params)
        callback.on_function_eval(project, 0, params, fom)
        return params, fom

    def test_automatic_recenter_does_not_refill_budget(self):
        def recenter(spec, project, callback, out_dir, params):
            self.one_forward(spec, project, callback, out_dir, params)
            raise eng.RecenterNeeded('simulated accepted-point recenter')
        with self.assertRaises(BudgetExhausted):
            self.run_driver(recenter)
        self.assertEqual(self.h.forward_launches, 2)
        self.assertEqual(json.loads(self.h.budget_path.read_text())['used'], 2)
        self.assertNotIn('CONVERGED', self.output.getvalue())
        self.assertNotIn('[done]', self.output.getvalue())

    def test_cold_resume_preserves_consumed_total_and_best_result(self):
        first = self.run_driver(self.one_forward, limit=1)
        best_path = self.root/(self.spec.label+'_best.json')
        best_raw = best_path.read_bytes()
        rows_raw = (self.root/(self.spec.label+'_evals.jsonl')).read_bytes()
        self.output = streams.StringIO()
        with self.assertRaises(BudgetExhausted):
            self.run_driver(self.one_forward, limit=1)
        self.assertEqual(self.h.forward_launches, 1)
        self.assertEqual(best_path.read_bytes(), best_raw)
        self.assertEqual((self.root/(self.spec.label+'_evals.jsonl')).read_bytes(), rows_raw)
        self.assertEqual(first['fom'], .81)
        self.assertNotIn('CONVERGED', self.output.getvalue())
        self.assertNotIn('[done]', self.output.getvalue())

    def test_explicit_extension_after_cold_resume_adds_only_remaining_units(self):
        self.run_driver(self.one_forward, limit=1)
        self.run_driver(self.one_forward, limit=2)
        with self.assertRaises(BudgetExhausted):
            self.run_driver(self.one_forward, limit=2)
        self.assertEqual(self.h.forward_launches, 2)
        self.assertEqual(json.loads(self.h.budget_path.read_text())['used'], 2)

    def test_cached_adjoint_free_uncached_gradient_forward_counted(self):
        def gradients(spec, project, callback, out_dir, params):
            result = self.one_forward(spec, project, callback, out_dir, params)
            project.compute_gradient(params)  # consumes the cached forward
            project.compute_gradient(params)  # no cache: spends one forward
            return result
        self.run_driver(gradients)
        self.assertEqual(self.h.forward_launches, 2)
        self.assertEqual(self.h.adjoint_launches, 2)
        self.assertEqual(json.loads(self.h.budget_path.read_text())['used'], 2)
        with self.assertRaises(BudgetExhausted):
            self.run_driver(self.one_forward)
        self.assertEqual(self.h.forward_launches, 2)

    def test_wrapped_exhaustion_is_terminal_and_does_not_recenter(self):
        self.h.wrap_errors = True
        def excessive(spec, project, callback, out_dir, params):
            self.one_forward(spec, project, callback, out_dir, params)
            self.one_forward(spec, project, callback, out_dir, params)
            return self.one_forward(spec, project, callback, out_dir, params)
        with self.assertRaises(BudgetExhausted):
            self.run_driver(excessive)
        self.assertEqual(len(self.h.projects), 1)
        self.assertEqual(self.h.forward_launches, 2)
        self.assertNotIn('[recenter', self.output.getvalue())
        self.assertNotIn('[done]', self.output.getvalue())

    def test_failed_unlogged_forward_is_counted_before_launch(self):
        self.h.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, 'simulated solver failure'):
            self.run_driver(self.one_forward, limit=1)
        self.assertFalse((self.root/(self.spec.label+'_evals.jsonl')).exists())
        self.h.fail_forward = False
        with self.assertRaises(BudgetExhausted):
            self.run_driver(self.one_forward, limit=1)
        self.assertEqual(self.h.forward_launches, 1)
        self.assertEqual(json.loads(self.h.budget_path.read_text())['used'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
