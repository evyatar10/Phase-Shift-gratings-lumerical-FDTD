"""
Smoke test for the FD-gradient runner — n_periods=20, max_iter=1.

Verifies end-to-end wiring on Athena without burning a 7-hour production slot:
  - Build .fsp via _build_parametric_fsp (PSO infra)
  - One L-BFGS-B step → 11 FDTD evaluations (1 base + 2*5 perturbed)
  - Post-opt verification at accurate mesh
  - JSON summary

Expected runtime: ~30 min on a single A100/H200/L40S.

Acceptance:
  - fdgrad_incremental.json lists exactly 11 evaluations after one iter
  - At p0 = uniform pi-shift Bragg grating [300,300,0,0,800], by mirror
    symmetry, ∂T/∂shift_i ≈ 0 (within FDTD noise); ∂T/∂dw_i and
    ∂T/∂cavity_width should be nonzero
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.fd_gradient_design.fd_gradient_design import FDGradientSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=20)   # ← smoke: small device


SPEC = FDGradientSpec(
    n_free_inner_teeth = 2,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = None,             # → uniform pi-shift Bragg grating
    cavity_width_start_nm = 800.0,
    n_starts           = 1,

    max_iter           = 1,                 # ← smoke: one L-BFGS-B step
    ftol               = 1e-5,
    gtol               = 1e-5,

    fd_step_nm         = 50.0,
    fom_window_nm      = 10.0,
    fom_n_points       = 51,                # smoke: coarse spectral grid
    mesh_override_dxyz_nm = 0.0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "smoke",
)


if __name__ == "__main__":
    from runners.fd_gradient_design.fd_gradient_design import run_fd_gradient_design
    print(SPEC.describe())
    print()
    run_fd_gradient_design(cfg=BASE, spec=SPEC, start_idx=0)
