"""Refinement sequence, 20-um seed at y_mesh_cells 37 (dy ~20 nm); see farfield_sph_20um_meshcheck.py (dy 50/37/29 -> leak 100/95/78 %, unconverged).
Study dir: runners/sweeps/ | Created 2026-10-20 | GPT review 01a157fa: extend the sequence before any physical/numerical attribution.
"""
import os
os.environ["FF_MESHCHECK_YCELLS"] = "37"
from runners.sweeps.farfield_sph_20um_meshcheck import CARDS, BASE  # noqa: E402,F401
