"""Offline tests of the observable contract; no CAD or solves."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import width_contract as wc


class WidthContractTests(unittest.TestCase):
    def spec(self, **kw):
        return SimpleNamespace(**dict(dict(fwhm0_um=19.121, fwhm_bounds_um=None,
            wgp_margin_um=.1, wgp_target_um=19.121, wgp_v3_band_um=None), **kw))

    def test_retained_anchor_band_and_smoke(self):
        s = self.spec()
        self.assertEqual(wc.hard_band(s, .98, 1.02), (.98*19.121, 1.02*19.121))
        self.assertEqual(wc.qp_band(s, .98, 1.02), (.98*19.121+.1, 1.02*19.121-.1))
        smoke = self.spec(fwhm0_um=None)
        self.assertIsNone(wc.hard_band(smoke, .98, 1.02))
        self.assertEqual(wc.qp_band(smoke, .98, 1.02), wc.qp_band(s, .98, 1.02))

    def test_no_growth_is_an_explicit_hard_upper(self):
        s = self.spec(fwhm_bounds_um=(.98*19.121, 19.121))
        band = wc.hard_band(s, .98, 1.02)
        self.assertAlmostEqual(wc.qp_band(s, .98, 1.02)[1], 19.021)
        self.assertEqual(wc.violation(19.121, band), 0)
        self.assertGreater(wc.violation(19.121001, band), 0)
        self.assertGreater(wc.violation(19.385006, band), .264)

    def test_invalid_or_inconsistent_config_fails_before_solver(self):
        for band in [(20, 19), (0, 20), (19, float('nan')), (19,), '19,20']:
            with self.subTest(band=band), self.assertRaises(ValueError):
                wc.hard_band(self.spec(fwhm_bounds_um=band), .98, 1.02)
        for changes in [dict(wgp_margin_um=-.1), dict(wgp_margin_um=1),
                        dict(wgp_v3_band_um=(19, 21))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                wc.qp_band(self.spec(**changes), .98, 1.02)

    def test_missing_nonfinite_never_counts_as_measured_compliance(self):
        for value in [None, float('nan'), float('inf'), -float('inf')]:
            self.assertEqual(wc.violation(value, (19,20)), float('inf'))


if __name__ == '__main__':
    unittest.main()
