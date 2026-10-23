"""farfield_sph_15um_comb_overshoot VALIDATION: the single winning comb (row 3: +12 um, 51 deg, r 63) at the converged mesh dy 28.8 nm.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Control = Athena 267605 task 0 (dy29_box6p8, farfield_sph_15um_budget_overshoot): Q_i 5.67e5 at identical numerics.
"""
import os
os.environ["FF15_COMB_DY29"] = "1"
from runners.sweeps.farfield_sph_15um_comb_overshoot import SPEC, BASE  # noqa: E402,F401
