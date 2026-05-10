"""
Diagnostic variant: check_gradient at the REGULAR-GRATING starting point
(no apodization, no shifts) instead of the apodized empirical optimum.

Hypothesis: at the apodized point, the true gradient is small and FD is
dominated by FDTD noise. At the regular-grating point, peak T ~0.86 and
the FOM has a clear slope — both FD and adjoint should give large values
and agreement should be cleaner.

If FD ≈ adjoint here but not at apodized → starting point was the issue.
If both disagree the same way → adjoint pipeline has a real bug.
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


# Regular grating: full DW, zero shifts, default cavity.
INITIAL_P = [300.0, 300.0, 0.0, 0.0, 800.0]

SPEC = InverseDesignSpec(
    n_free_inner_teeth = 2,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    mode                  = "check_gradient",
    check_gradient_dx_nm  = 50.0,

    max_iter           = 1,
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,

    fom_window_nm        = 10.0,
    fom_n_points         = 51,
    fom_weight_sigma_nm  = 1.0,            # Gaussian (matches production)

    mesh_override_dxyz_nm= 0,
    param_dx_nm          = 50.0,

    use_concurrent_adjoint_solves = False,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "check_grad_regular",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
