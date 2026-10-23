"""farfield_sph_15um_comb_overshoot TWO-ROW: the single winning comb (row 1) + a second row at y +-1.9 um aimed at |u_x| 0.85, phase circle.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Reference = the DY29 validation of the single winner (same dy 28.8 nm, box 6.8/6.81, short-run numerics); control = Athena 267605 task 0.
"""
import os
os.environ["FF15_COMB_2ROW"] = "1"
from runners.sweeps.farfield_sph_15um_comb_overshoot import SPEC, BASE  # noqa: E402,F401
