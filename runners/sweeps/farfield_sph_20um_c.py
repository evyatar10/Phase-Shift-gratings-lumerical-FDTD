"""Round C of farfield_sph_20um under its own module name (own results/ dir).

Study dir: runners/sweeps/ | Created 2026-10-13 | Job(s): see memory project_te_inverse_design_lane.md
Purpose: S2 seed vs TE_S2_N98_T9739 far field at conformal / 150 ps for the alpha/beta table.
"""
import os
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um import SPEC, BASE  # noqa: E402,F401
