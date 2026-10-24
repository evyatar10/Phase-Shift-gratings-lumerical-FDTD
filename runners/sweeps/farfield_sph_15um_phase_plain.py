"""farfield_sph_15um_phase on the PLAIN TE pi-shift device (own module = own results dir and file tags).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Purpose: the registration (phase) test of the off-lattice tab set on the plain corr-316 / pitch-500 / W800 device
(round A of farfield_sph_15um: 150 ps steady state; control = the stored round-A far field, Lorentzian normalisation).
"""
import os
os.environ["FF15_PHASE_DEV"] = "PLAIN"
from runners.sweeps.farfield_sph_15um_phase import SPEC, BASE  # noqa: E402,F401
