"""Phase B of farfield_sph_15um (decay-mode gate) under its own module name (own results/ dir).

Study dir: runners/sweeps/ | Created 2026-10-19 | Job(s): see memory project_farfield_sph_15um.md
Purpose: gate the ring-down far-field mode on the stored 20-um overshoot (see farfield_sph_15um ROUND B).
"""
import os
os.environ["FF15_ROUND"] = "B"
from runners.sweeps.farfield_sph_15um import SPEC, BASE  # noqa: E402,F401
