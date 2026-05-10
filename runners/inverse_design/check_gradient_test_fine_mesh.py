"""
Diagnostic variant: check_gradient with 10 nm override mesh in freed region.

Hypothesis: 50 nm device mesh with 25 nm FD perturbation samples eps changes
at coarse resolution; finer mesh would let the FD-on-eps integration give
adjoint-correct values.

If FD ≈ adjoint here but not at 50 nm mesh → mesh resolution was the issue.
If still disagreeing → mesh isn't the bottleneck.
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


INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]

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
    fom_weight_sigma_nm  = 1.0,

    mesh_override_dxyz_nm= 10.0,            # ← 10 nm override in freed region
    param_dx_nm          = 50.0,

    use_concurrent_adjoint_solves = False,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "check_grad_fine",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
