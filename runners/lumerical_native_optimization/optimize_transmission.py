"""
Lumerical-native study: maximize peak T(lambda) of the pi-shift Bragg cavity
using Lumerical's BUILT-IN addsweep("Optimization") with the Particle Swarm
algorithm. Same 5 parameters as the gradient_free_design path; FOM is
max(T) over the bandgap (non-differentiable, fine for PSO/GA).

Invocation:
  Athena (single-GPU job, walltime budget like gradient_free_design):
    bash athena/deploy_athena.sh --lumerical-native=runners.lumerical_native_optimization.optimize_transmission

  DGX:
    bash dgx/deploy_dgx.sh --lumerical-native=runners.lumerical_native_optimization.optimize_transmission
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.lumerical_native_optimization.lumerical_native_optimization import LumericalNativeSpec
from runners.optimization_common import make_optimization_base


BASE = make_optimization_base(n_periods=80)


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]   # apodized + shifted, same as gradient_free

SPEC = LumericalNativeSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    # Same budget as the Python-PSO path so timings are comparable head-to-head.
    # Lumerical's addsweep('Optimization') will dispatch generation_size jobs
    # per generation; n_concurrent caps how many run in parallel.
    algorithm          = "Particle Swarm",
    population_size    = 15,
    max_generations    = 10,
    tolerance          = 1e-4,
    n_concurrent       = 1,         # single-GPU sequential (Athena --gpus=1)

    fom_window_nm      = 10.0,
    fom_n_points       = 201,
    mesh_override_dxyz_nm = 0,       # NO fine override at n_periods=80 — same
                                      # rationale as gradient_free_design.

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission_lumerical_native",
)


if __name__ == "__main__":
    from runners.lumerical_native_optimization.lumerical_native_optimization import run_lumerical_native
    print(SPEC.describe())
    print()
    run_lumerical_native(cfg=BASE, spec=SPEC, start_idx=0)
