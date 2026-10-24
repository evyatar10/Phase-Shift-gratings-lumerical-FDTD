"""Mean-width ripple probe, phase 90 deg (see farfield_sph_20um_ripple.py; one phase per module = own results dir).
Study dir: runners/sweeps/ | Created 2026-10-20
"""
import os
os.environ["FF_RIPPLE_DEG"] = "90"
from runners.sweeps.farfield_sph_20um_ripple import SPEC, BASE  # noqa: E402,F401
