"""
Short corrected PSO run with the modal-T FOM fix.

Previous PSO (DGX 470634) used port's flux-based "T" output. Independent verify
showed it climbed a hill of cladding radiation: gen-5 best gave port-T=0.9916
but modal |S21|² = 0.972, BELOW the regular-grating baseline of 0.9847.
This run uses |S21|² (modal) → find_bragg_resonance as the FOM.

Budget: 3 gens × 15 particles = 45 evals (~2 h on a single DGX GPU at n=80).
Seeded at the apodized point so we can directly test whether apodization +
inner-tooth shifts can exceed the regular-grating baseline (0.9847 modal).

Invocation:
  bash dgx/deploy_dgx.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission_corrected
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.gradient_free_design.gradient_free_design import GradientFreeDesignSpec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.grating.lengthen_cavity     = True
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 10.0
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled          = False


N_FREE = 2
INITIAL_P = [250.0, 280.0, 50.0, 30.0, 800.0]   # apodized + shifted; modal T to be measured

SPEC = GradientFreeDesignSpec(
    n_free_inner_teeth = N_FREE,
    dw_bounds_nm       = [(60.0, 400.0), (60.0, 400.0)],
    shift_bounds_nm    = [(0.0, 200.0),  (0.0, 200.0)],
    cavity_width_bounds_nm = (500.0, 1100.0),
    initial_points     = [INITIAL_P],
    n_starts           = 1,

    algorithm          = "Particle Swarm",
    population_size    = 15,
    max_generations    = 3,
    tolerance          = 1e-4,
    n_concurrent       = 1,

    fom_window_nm      = 10.0,
    fom_n_points       = 201,
    mesh_override_dxyz_nm = 0,

    enforce_mirror_symmetry = True,
    lengthen_cavity    = True,
    label              = "transmission_gf_modal",
)


if __name__ == "__main__":
    from runners.gradient_free_design.gradient_free_design import run_gradient_free_design
    print(SPEC.describe())
    print()
    run_gradient_free_design(cfg=BASE, spec=SPEC, start_idx=0)
