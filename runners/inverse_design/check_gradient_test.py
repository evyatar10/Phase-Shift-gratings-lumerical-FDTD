"""
Diagnostic: verify lumopt's adjoint gradient against finite differences at the
apodized starting point.

If FD ≈ adjoint within ~5% relative error, the gradient pipeline is correct
and any near-zero gradient at this point is a real local optimum (not a
wiring bug). If FD and adjoint disagree, we have a code issue to fix.

Same device size as smoke_test (n_periods=20, fom_n_points=51) for fast
turnaround. Single GPU, ≈30 min wall.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.inverse_design import InverseDesignSpec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 20
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


# Same starting point as smoke_test / optimize_transmission (apodized).
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]

SPEC = InverseDesignSpec(
    n_free_inner_teeth = 2,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    mode                  = "check_gradient",
    check_gradient_dx_nm  = 50.0,           # central diff: ±25 nm

    max_iter           = 1,                  # unused in check_gradient mode
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,

    # σ=2 nm matches production. With patched adjoint (target_T_fwd_weights
    # now correctly applied to the kernel), FD and adjoint should agree to
    # vec_error < 0.1. Pre-patch: ~12-44 across all tests.
    fom_window_nm        = 10.0,
    fom_n_points         = 51,
    fom_weight_sigma_nm  = 2.0,

    mesh_override_dxyz_nm= 0,
    param_dx_nm          = 50.0,

    use_concurrent_adjoint_solves = False,    # serial: 1 license seat at a time
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "check_grad",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
