"""farfield_sph_15um_comb_overshoot V3: one-line comb (single lattice, period 539 nm) and the pair + a second row per arm (Born-designed).

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Control = stored Nt80x T340 (production mesh); reference = the measured pair 268986 (Q_i 6.25e5).
"""
import os
os.environ["FF15_COMB_V3"] = "1"
from runners.sweeps.farfield_sph_15um_comb_overshoot import SPEC, BASE  # noqa: E402,F401
