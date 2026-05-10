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
BASE.mesh.simulation_mode        = "optimization"   # device-wide dx=50 nm; the
                                                    # freed region gets a finer
                                                    # 10 nm override (Phase-2 fix
                                                    # #2) wired in inverse_design.
BASE.spectral.scan_width_nm      = 10.0             # full bandgap window; FOM
                                                    # weight (σ=1 nm Gaussian)
                                                    # restricts the integral.
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

# Empirically-informed starting point. The regular grating ([300,300,0,0,800])
# is a known LOCAL plateau of the FOM landscape — gradient nearly zero, L-BFGS-B
# stalls. Per the user's prior sweep work (innermost_shift, apod_and_shift),
# improvement comes from APODIZING the inner teeth (smaller DW) and adding
# a TOOTH SHIFT. Starting near a known-good empirical region gives the
# optimizer a meaningful initial gradient and avoids the regular-grating
# saddle.
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]   # apodized + slightly shifted

SPEC = InverseDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    # half_pitch = 250 nm, so 200 nm shift leaves 50 nm narrow-tooth min.
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # Single L-BFGS-B with 16 iterations. Each iter ≈ 7 FDTDs × ~3 min = 21
    # min, plus line search → ~5 hours total. ARRAY_TIME=23:30 has plenty of
    # headroom.
    max_iter           = 16,
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,

    # FOM: Gaussian-weighted T over the 10 nm bandgap window. σ=1 nm is wide
    # enough to keep the resonance in the FOM band even with ±2 nm drift.
    fom_window_nm        = 10.0,
    fom_n_points         = 201,
    fom_weight_sigma_nm  = 1.0,

    # No fine override mesh — device-wide periodic-aligned 50 nm mesh is the
    # right thing for periodic structures. dx_param=50 nm matches the mesh
    # so finite-difference perturbations actually move at least one cell
    # boundary.
    mesh_override_dxyz_nm= 0,
    param_dx_nm          = 50.0,

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
