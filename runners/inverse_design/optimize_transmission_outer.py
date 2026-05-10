"""
Lumopt adjoint inverse design with outer-loop active-set re-centering.

Why this exists: the standard `optimize_transmission.py` runs lumopt with
fom_n_points=201 (Gaussian-weighted multi-λ FOM). On GPU FDTD this is
broken — the broadband port profile mismatch contaminates the adjoint
(check_gradient vec_error 11.40 vs healthy <0.1).

This spec uses fom_n_points=1 (single-λ FOM) + outer-loop re-centering.
At single-λ the GPU adjoint is mathematically correct; the outer loop
handles resonance drift between inner-loop iterations by re-measuring
the resonance and rebuilding the inner FOM.

Invocation:
  bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission_outer
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.inverse_design import InverseDesignSpec
from simulation_config import SimulationConfig


# ── Base simulation config ───────────────────────────────────────────────────
BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0     # broadband scan for baseline measurement
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


# ── Outer-loop spec ──────────────────────────────────────────────────────────
# Start from the uniform pi-shift Bragg grating (matches FD-gradient path).

INITIAL_P = [300.0, 300.0, 0.0, 0.0, 800.0]   # uniform pi-shift Bragg grating

SPEC = InverseDesignSpec(
    n_free_inner_teeth = 2,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # Inner = lumopt L-BFGS-B with single-λ FOM. GPU stays valid.
    # Outer = re-center λ_target between inner runs.
    # Total inner iters = max_iter * n_outer_iters = 4 * 4 = 16.
    max_iter           = 4,
    n_outer_iters      = 4,
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,

    # Single-λ FOM → adjoint correct on GPU. Outer loop re-centers between
    # inner iters to handle resonance drift (~0.5-1 nm per inner block).
    fom_window_nm        = 0.0,           # ignored when fom_n_points=1
    fom_n_points         = 1,
    fom_weight_sigma_nm  = 1.0,           # ignored when fom_n_points=1

    mesh_override_dxyz_nm= 0,
    param_dx_nm          = 50.0,

    use_concurrent_adjoint_solves = True,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission_outer",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design_outer_loop
    print(SPEC.describe())
    print(f"  n_outer_iters    = {SPEC.n_outer_iters}")
    print()
    run_inverse_design_outer_loop(cfg=BASE, spec=SPEC, start_idx=0)
