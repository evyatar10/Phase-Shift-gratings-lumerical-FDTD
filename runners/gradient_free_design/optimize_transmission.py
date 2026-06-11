"""
Gradient-free study: maximize peak T(λ) of the pi-shift Bragg cavity using
Lumerical's built-in Particle Swarm Optimization. Same 5 parameters as
Option A; FOM is max(T) over the bandgap (non-differentiable, fine for PSO).

Invocation:
  Athena (4-GPU job, ~2-3 hours):
    bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=80)


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]   # apodized + shifted

SPEC = GradientFreeDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # PSO budget for 23:30 ARRAY_TIME at n_periods=80, single GPU:
    # pop=15 × (gens=10 + 1 init) ≈ 165 evals × ~3 min FDTD = ~8 hours.
    # Incremental save after each gen so walltime-cut runs still produce
    # the global-best particle to date.
    algorithm          = "Particle Swarm",
    population_size    = 15,
    max_generations    = 10,
    tolerance          = 1e-4,
    n_concurrent       = 1,         # single-GPU sequential

    fom_window_nm      = 10.0,
    fom_n_points       = 201,        # 6 samples per FWHM is enough; reduces
                                      # transmission monitor memory.
    mesh_override_dxyz_nm = 0,       # NO fine override — at n_periods=80 the
                                      # fine override makes FDTD intractable.
                                      # Device-wide 50 nm mesh is fine for PSO
                                      # since each particle just runs FDTD once.

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission_gf",
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0)
