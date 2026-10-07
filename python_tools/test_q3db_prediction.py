"""Offline regressions for inference domain, false roots and honest error bands."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import predict_q3db as p
import calibrate_q3db as c


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.base = p.load_calib()["tm_bare_c325"]
        self.row = dict(N=100, T=0.91, Q_L=3000., lam_nm=1560., width_um=19.)

    def test_anchor_reproduces_observables_and_records_condition(self):
        q, _, _ = p.anchor_on_row(self.base, self.row)
        observed = p.observables(q, 100)
        for k in ["T", "Q_L", "width_um", "lam_nm"]:
            self.assertAlmostEqual(observed[k], self.row[k], places=9)
        self.assertGreater(q["anchor_T_condition"], 5)
        short, _, _ = p.anchor_on_row(self.base, {**self.row, "N": 1})
        self.assertTrue(any("ANCHOR WARNING" in s for s in p.validity_notes(short, 200)))

    def test_invalid_anchor_refused(self):
        for bad in [dict(T=1.), dict(T=1.01), dict(T=0.), dict(Q_L=-1.),
                    dict(N=0), dict(lam_nm=np.nan)]:
            with self.assertRaises(ValueError):
                p.anchor_on_row(self.base, {**self.row, **bad})

    def test_discontinuous_feasibility_boundary_is_not_width_root(self):
        with patch.object(p, "load_calib", return_value={"knob_tm": {"iw_a": 0., "iw_b": 1.}}), \
                patch.object(p, "CORR_BOUNDS_NM", (1., 2.)), \
                patch.object(p, "rescale_family_to_corr", side_effect=lambda base, old, new: {"corr": new}), \
                patch.object(p, "design_N", side_effect=lambda q, target: 100. if q["corr"] < 1.5 else None), \
                patch.object(p, "width_of_N", return_value=21.):
            q, _, note = p.retune_corr({"w_Finf": 20.}, 1., "TM", 20., .5)
            self.assertIsNone(q)
            self.assertTrue("boundary" in note or "did not solve width" in note)

    def test_real_width_design_satisfies_both_targets(self):
        q, _, note = p.retune_corr(self.base, 325., "TM", 20., 10**(-3/10))
        self.assertIsNotNone(q, note)
        n = p.design_N(q, 10**(-3/10))
        self.assertAlmostEqual(p.observables(q, n)["T"], 10**(-3/10), places=5)
        self.assertLess(abs(p.width_of_N(q, n)-20.), .002)

    def test_single_anchor_does_not_inherit_multirow_error_band(self):
        q, _, _ = p.anchor_on_row(self.base, self.row)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            p.print_prediction("test", q, 165, bands=p.load_calib()["errband"])
        self.assertIn("no validated single-anchor error band", output.getvalue())
        self.assertNotIn("expected deviation (from", output.getvalue())

    def test_selective_mat_reader_and_invalid_tcmt_input(self):
        folder = Path(tempfile.mkdtemp(prefix="q3db_reader_test_"))
        path = folder / "result_N100_TM_C325_test.mat"
        values = dict(resonance_transmission=.5, resonance_wavelength_nm=1560.,
                      spectral_fwhm_nm=.1, fwhm_m=20e-6, pitch_m=500e-9,
                      corrugation_depth_m=325e-9, wl_nm=np.array([1559.,1560.,1561.]),
                      T=np.array([.1,.5,.1]), R=np.zeros(3), unused_field=np.ones((100,100)))
        savemat(path, values)
        with patch.object(c, "loadmat", wraps=c.loadmat) as loader:
            row = c.load_mat_row(path)
        self.assertEqual(row["T"], .5)
        self.assertEqual(row["QL"], 15600.)
        self.assertNotIn("unused_field", loader.call_args.kwargs["variable_names"])
        invalid = folder / "result_N100_TM_C325_invalid.mat"
        savemat(invalid, {**values, "resonance_transmission": 1.01})
        with self.assertRaises(ValueError):
            c.load_mat_row(invalid)
        width_only = c.load_mat_row(invalid, require_tcmt=False)
        self.assertTrue(np.isnan(width_only["Qi"]))
        self.assertAlmostEqual(width_only["fwhm_um"], 20.)


if __name__ == "__main__":
    unittest.main()
