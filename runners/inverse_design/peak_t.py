"""
Inverse-design study: maximize peak transmission of the pi-shift Bragg cavity
by tuning the two innermost teeth (DW + shift on each) and the cavity width.

Invocation patterns:
  Local (one driver, sequential):
    python -m runners.inverse_design.peak_t_adjoint --spec runners.inverse_design.peak_t --start 0
  Athena (SLURM array, multiple drivers in parallel):
    bash athena/deploy_athena.sh --inverse-design   # picks runners.inverse_design.peak_t
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.spec import InverseDesignSpec, regular_grating_start
from simulation_config import SimulationConfig


# ── Base simulation config (matches the inverse-design plan) ─────────────────

BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"   # dx=50 nm
BASE.spectral.scan_width_nm      = 10.0             # narrow window per the plan
# Apodization stays at its default (off); freed inner teeth carry their own DW.


# ── Inverse-design spec ──────────────────────────────────────────────────────
# Single deterministic start at the regular-grating geometry. Peak T at this
# start must be reproduced by the optimizer's first iteration; any drop is a
# wiring bug. Multi-start LHS sampling is available via spec.get_starts() but
# disabled here since the regular grating is a known-good convergent baseline.

N_FREE = 2
INITIAL_P = regular_grating_start(BASE, n_free_inner_teeth=N_FREE, cavity_width_nm=800.0)

SPEC = InverseDesignSpec(
    n_free_inner_teeth = N_FREE,
    # Bounds widened from the plan defaults to keep the regular-grating
    # baseline (full corrugation depth = 300 nm) strictly inside dw_bounds.
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,
    max_iter           = 30,
    optimizer_method   = "L-BFGS-B",
    use_concurrent_adjoint_solves = True,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "peak_t",
)


if __name__ == "__main__":
    print(SPEC.describe())
