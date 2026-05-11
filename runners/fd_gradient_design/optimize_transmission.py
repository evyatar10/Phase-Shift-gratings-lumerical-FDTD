"""
FD-gradient study: maximize peak T(λ) of the pi-shift Bragg cavity from the
uniform pi-shift Bragg grating starting point, using scipy L-BFGS-B with
central-difference gradients.

Cost function: peak T(λ) = max(T) over the bandgap window, evaluated by the
existing PSO FOM evaluator (one FDTD per call).

Starting point: regular_grating_start = [300, 300, 0, 0, 800] (uniform pi-shift
Bragg grating with default cavity width). Per the user's stated preference and
the published practice (Akahane 2003, Lalanne) of starting shape-optimization
from a known-good baseline, this is the canonical starting condition.

Invocation:
  Athena (single-GPU job, ~6-7 hours):
    bash athena/deploy_athena.sh --fd-gradient-design=runners.fd_gradient_design.optimize_transmission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.fd_gradient_design.fd_gradient_design import FDGradientSpec
from simulation_config import SimulationConfig


# ── Base simulation config ───────────────────────────────────────────────────

BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True             # cavity absorbs Σ(shifts)
BASE.mesh.simulation_mode        = "optimization"   # device-wide dx=50 nm
BASE.spectral.scan_width_nm      = 10.0             # bandgap window for T(λ)

BASE.monitors.record_2d_fields = False              # FD-gradient runs 100+ FDTDs;
BASE.monitors.record_3d_fields = False              # field profiles are pure overhead.
BASE.farfield.enabled          = False


# ── FD-gradient spec ─────────────────────────────────────────────────────────
# Default initial_points=None → spec.get_starts(cfg) returns
# regular_grating_start(cfg, n_free=2, cavity_width_nm=800.0) =
# [300, 300, 0, 0, 800] — uniform pi-shift Bragg grating with default cavity.

SPEC = FDGradientSpec(
    n_free_inner_teeth = 2,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    # Warm-start from job 79145's best partial-run point (peak_T = 0.9474 at
    # eval 24/137 before cancellation). The cancelled run had only swept dw_1
    # and dw_2 — shift_1, shift_2, cavity_width still at defaults. Restarting
    # here lets Powell skip the dw exploration and go straight to probing
    # shift+cavity dimensions, where most remaining T headroom lives.
    initial_points     = [[189.87, 176.22, 25.0, 25.0, 800.0]],
    cavity_width_start_nm = 800.0,
    n_starts           = 1,

    # Powell budget. With max_iter=8 (rounds of N=5 directional searches),
    # eval_budget = max_iter * N * 15 = 600 evals = ~10 hours at 1 min/eval.
    # Fits Athena ARRAY_TIME=23:30 with margin.
    max_iter           = 8,
    ftol               = 1e-5,
    gtol               = 1e-5,

    # Central-difference half-step. 50 nm matches the device-wide mesh, so
    # each probe moves at least one Yee-cell boundary (avoids the "ε didn't
    # change → zero gradient" failure mode that broke the lumopt path).
    fd_step_nm         = 50.0,

    # Wide bandgap window so the resonance never escapes during optimization.
    # peak T evaluator picks max(T) over the recorded band; window width does
    # not affect FOM value once the peak sits inside.
    fom_window_nm      = 10.0,
    fom_n_points       = 201,         # ~5 samples per FWHM at production Q.

    # No fine override mesh — at n_periods=80 the override makes FDTD ~8x
    # slower and doesn't fit walltime. Device-wide 50 nm + fd_step=50 nm
    # gives mesh-aligned probes.
    mesh_override_dxyz_nm = 0.0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission_fdgrad",
)


if __name__ == "__main__":
    from runners.fd_gradient_design.fd_gradient_design import run_fd_gradient_design
    print(SPEC.describe())
    print()
    run_fd_gradient_design(cfg=BASE, spec=SPEC, start_idx=0)
