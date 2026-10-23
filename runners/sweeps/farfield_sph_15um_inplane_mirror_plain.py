"""farfield_sph_15um_inplane_mirror on the PLAIN TE pi-shift device (own module = own results dir and file tags).

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Purpose: fast (150 ps steady state) read of the in-plane mirror gap scan; control = the stored round-A plain far field.
"""
import os
os.environ["FF15_MIRROR_DEV"] = "PLAIN"
from runners.sweeps.farfield_sph_15um_inplane_mirror import SPEC, BASE  # noqa: E402,F401
