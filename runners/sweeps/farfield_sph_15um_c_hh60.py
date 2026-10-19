"""Phase C of farfield_sph_15um for the HH60 overshoot (own module = own results dir and file names).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): see memory project_farfield_sph_15um.md
Purpose: locate (FF15_C_STEP=LOC) then measure (PROD) the 15.5-um corr-500 overshoot at pitch 514, N=96.
"""
import os
os.environ["FF15_ROUND"] = "C_HH60"
from runners.sweeps.farfield_sph_15um import SPEC, BASE  # noqa: E402,F401
