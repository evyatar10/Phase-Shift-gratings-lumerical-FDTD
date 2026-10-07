"""No-solver checks that the bounded smoke cannot bypass calibration."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import smoke_te_s2_restart as smoke


class SmokeChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='s2_smoke_guard_'))
        self.path = self.root/'calibration.json'
        re = np.arange(1., 7.)
        im = np.array([2., -1., 3., 1., -2., 4.])
        self.coeff = np.array([.93, .12])
        self.row = dict(identity=dict(objective='peak3_abs_T_out'),
                        result=dict(re=re.tolist(), im=im.tolist(),
                                    fd=(self.coeff[0]*re+self.coeff[1]*im).tolist(),
                                    coefficients=self.coeff.tolist(), passed=True))

    def write(self):
        raw = json.dumps(self.row).encode()
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_missing_review_stops_before_cad(self):
        with patch.object(smoke, 'CALIBRATION_SHA256', None), \
             patch.object(smoke.eng, 'run_campaign', side_effect=AssertionError('CAD touched')):
            with self.assertRaisesRegex(RuntimeError, 'not passed review'):
                smoke.main(0)

    def test_valid_pinned_fit_and_bounded_spec(self):
        coeff = smoke.load_calibration(self.path, self.write())
        np.testing.assert_allclose(coeff, self.coeff, rtol=0, atol=0)
        spec = smoke.smoke_spec(coeff)
        self.assertEqual((spec.n_periods_side, spec.n_free, spec.max_iter), (70, 60, 2))
        self.assertTrue(spec.bare and spec.wgp_v3 and spec.wgp_v3_peak)
        self.assertEqual((spec.adj_fix_re, spec.adj_fix_im), coeff)
        self.assertIsNone(spec.fwhm0_um)

    def test_hash_and_wrong_objective_rejected(self):
        good_hash = self.write()
        self.path.write_bytes(self.path.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            smoke.load_calibration(self.path, good_hash)
        self.row['identity']['objective'] = 'softmax'
        with self.assertRaisesRegex(RuntimeError, 'different objective'):
            smoke.load_calibration(self.path, self.write())

    def test_pass_boolean_does_not_override_failed_fit(self):
        self.row['result']['fd'][0] *= 2
        with self.assertRaisesRegex(RuntimeError, '10% gates'):
            smoke.load_calibration(self.path, self.write())

    def test_bad_vectors_rejected(self):
        self.row['result']['fd'][0] = 0
        with self.assertRaisesRegex(RuntimeError, 'Invalid calibration'):
            smoke.load_calibration(self.path, self.write())

    def test_smoke_requires_budget_evidence_and_counts_attempts(self):
        spec = smoke.smoke_spec((.93, .12))
        with self.assertRaisesRegex(RuntimeError, 'no persistent forward budget'):
            smoke.audit_budget(self.root, spec, 2)
        ledger = smoke.evaluation_budget.ForwardEvaluationBudget(
            self.root, spec.label, spec.max_feval)
        ledger.reserve()
        with self.assertRaisesRegex(RuntimeError, 'does not cover'):
            smoke.audit_budget(self.root, spec, 2)
        ledger.reserve()
        self.assertEqual(smoke.audit_budget(self.root, spec, 2), 2)


if __name__ == '__main__':
    unittest.main()
