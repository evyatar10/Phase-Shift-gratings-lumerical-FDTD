"""
Smoke-test variant: small device, small population, few generations, single GPU.
Goal: verify the .fsp build, parametric structure group, and Lumerical PSO
all wire together correctly. ~15-30 minutes wall-time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=20)   # smaller device


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]

SPEC = GradientFreeDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    algorithm          = "Particle Swarm",
    population_size    = 8,           # small swarm
    max_generations    = 2,           # short run = 16 evals
    tolerance          = 1e-3,
    n_concurrent       = 1,           # single-GPU smoke (Athena --gpus=1)

    fom_window_nm      = 10.0,
    fom_n_points       = 201,         # reduced for speed
    mesh_override_dxyz_nm = 10.0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "smoke_gf",
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0)
