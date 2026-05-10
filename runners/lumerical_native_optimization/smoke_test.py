"""
Smoke-test variant for the Lumerical-native (addsweep('Optimization')) path:
small device, small population, few generations, single GPU. Goal: verify that
the .fsp build, parametric structure group, optimization-FOM analysis group,
and Lumerical PSO sweep all wire together correctly. Faster than the full
optimize_transmission study so you can iterate on debugging.

Invocation:
  bash athena/deploy_athena.sh --lumerical-native=runners.lumerical_native_optimization.smoke_test
  bash dgx/deploy_dgx.sh        --lumerical-native=runners.lumerical_native_optimization.smoke_test
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.lumerical_native_optimization.lumerical_native_optimization import LumericalNativeSpec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 20      # smaller device
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]

SPEC = LumericalNativeSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    algorithm          = "Particle Swarm",
    population_size    = 5,           # very small swarm for smoke test
    max_generations    = 2,           # 2 generations -> ~10 evals
    tolerance          = 1e-3,
    n_concurrent       = 1,           # single-GPU smoke (Athena --gpus=1)

    fom_window_nm      = 10.0,
    fom_n_points       = 201,
    mesh_override_dxyz_nm = 10.0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "smoke_lumerical_native",
)


if __name__ == "__main__":
    from runners.lumerical_native_optimization.lumerical_native_optimization import run_lumerical_native
    print(SPEC.describe())
    print()
    run_lumerical_native(cfg=BASE, spec=SPEC, start_idx=0)
