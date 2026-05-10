"""
Tiny LOCAL smoke test for Option B (Python PSO + lumapi). Runs on Windows
with the Lumerical license. ~5-8 minutes total.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 10      # tiny
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


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
    population_size    = 3,
    max_generations    = 1,
    tolerance          = 1e-3,
    n_concurrent       = 1,

    fom_window_nm      = 10.0,
    fom_n_points       = 21,
    mesh_override_dxyz_nm = 20.0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "local_smoke_gf",
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0,
                             output_root=os.path.join(os.path.dirname(__file__), "_local_results"))
