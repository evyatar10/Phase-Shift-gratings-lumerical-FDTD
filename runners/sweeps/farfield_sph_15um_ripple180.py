"""Mean-width ripple probe on the 15.5-um Nt80x overshoot, phase 180 deg (see farfield_sph_15um_ripple.py).
Study dir: runners/sweeps/ | Created 2026-10-20
"""
import os
os.environ["FF15_RIPPLE_DEG"] = "180"
from runners.sweeps.farfield_sph_15um_ripple import SPEC, BASE  # noqa: E402,F401
