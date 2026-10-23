"""farfield_sph_15um_widecore_plain, row 1500 nm mean width (own module: y_mesh_cells 21 so dy matches the plain control 49.4 nm).

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
"""
import os
os.environ["FF15_WC"] = "1500"
from runners.sweeps.farfield_sph_15um_widecore_plain import SPEC, BASE  # noqa: E402,F401
