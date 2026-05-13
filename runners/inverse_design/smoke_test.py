"""
Smoke-test variant of optimize_transmission.py.

Same spec but:
  - n_periods_each_side = 20 (vs 80) → ~10x faster FDTD
  - max_iter = 2 → 2 adjoint iters
  - fom_n_points = 51 (vs 201)

Goal: catch wiring bugs in <15 minutes wall time. After this passes, run
optimize_transmission.py for the production run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.inverse_design.inverse_design import InverseDesignSpec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 20      # smaller device for fast smoke test
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]

SPEC = InverseDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    max_iter           = 4,                   # need >= 2 successful iters to verify
    optimizer_method   = "L-BFGS-B",
    optimizer_pgtol    = 1e-6,
    optimizer_ftol     = 1e-6,
    # Force first L-BFGS-B step to be 25% of bound range — see
    # InverseDesignSpec docstring. Without this the raw gradient (~1e-4 in
    # scaled [0,1] space) translates to sub-Angstrom physical steps and the
    # optimizer terminates after one iteration with no real movement.
    scale_initial_gradient_to = 0.25,

    fom_window_nm        = 10.0,
    fom_n_points         = 51,
    fom_weight_sigma_nm  = 2.0,                # match production / check_gradient

    mesh_override_dxyz_nm= 15,                 # user-requested 15 nm freed-region mesh
    param_dx_nm          = 50.0,

    use_concurrent_adjoint_solves = True,
    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "smoke",
)


if __name__ == "__main__":
    from runners.inverse_design.inverse_design import run_inverse_design
    print(SPEC.describe())
    print()
    run_inverse_design(cfg=BASE, spec=SPEC, start_idx=0)
