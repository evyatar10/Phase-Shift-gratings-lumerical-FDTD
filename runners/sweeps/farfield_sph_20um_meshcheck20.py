"""Staircase-moire kill test, y_mesh_cells 20 (dy ~38 nm); see farfield_sph_20um_meshcheck.py.
Study dir: runners/sweeps/ | Created 2026-10-20
"""
import os
os.environ["FF_MESHCHECK_YCELLS"] = "20"
from runners.sweeps.farfield_sph_20um_meshcheck import CARDS, BASE  # noqa: E402,F401
