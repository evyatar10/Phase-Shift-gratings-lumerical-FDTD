"""
Inverse-design study: maximize peak transmission of the pi-shift Bragg cavity
by tuning the two innermost teeth (DW + shift on each) and the cavity width.

Invocation:
  Local (one driver, sequential):
    python -m runners.inverse_design.inverse_design --spec runners.inverse_design.optimize_transmission
  Athena (SLURM, parallel-ready via concurrent_adjoint_solves=True):
    bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission
  DGX:
    bash dgx/deploy_dgx.sh --inverse-design=runners.inverse_design.optimize_transmission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.inverse_design import InverseDesignSpec, regular_grating_start
from simulation_config import SimulationConfig


# ── Base simulation config (matches the inverse-design plan) ─────────────────

BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True             # NOT optimizing cavity length;
                                                    # this absorbs Σ(shifts) so total
                                                    # device length stays constant.
BASE.mesh.simulation_mode        = "optimization"   # dx=50 nm
BASE.spectral.scan_width_nm      = 10.0             # narrow window per the plan
# Apodization stays at its default (off); freed inner teeth carry their own DW.

# ── Optional field profile monitors (default OFF for inverse design) ────────
# Optimization only needs T(λ) at Port_2 (PortTransmission FOM); profile
# monitors are pure overhead during the 60+ FDTD runs.
# Flip any of these to True if you want field profiles recorded during the run
# (e.g. for post-mortem visualization of the optimized geometry's mode shape).
BASE.monitors.record_2d_fields = False   # XY top, YZ cross, XZ side profiles
BASE.monitors.record_3d_fields = False   # full 3D field volume
BASE.farfield.enabled          = False   # side + top far-field monitors


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
    label              = "transmission",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
