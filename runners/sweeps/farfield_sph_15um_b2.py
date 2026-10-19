"""Phase B2 of farfield_sph_15um (GPU-compatible decay gate) under its own module name (own results/ dir).

Study dir: runners/sweeps/ | Created 2026-10-19 | Job(s): see memory project_farfield_sph_15um.md
Purpose: short-run correction + two-length difference window on the stored 20-um overshoot (ROUND B2).
"""
import os
os.environ["FF15_ROUND"] = "B2"
from runners.sweeps.farfield_sph_15um import SPEC, BASE  # noqa: E402,F401
