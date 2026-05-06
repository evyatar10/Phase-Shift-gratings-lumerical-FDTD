"""
Sweep: innermost tooth corrugation depth × tooth shift (cartesian product).

The innermost-tooth depth is set via 1-period apodization
(n_apod_periods_each_side=1, center_mod_depth_nm=<size>).

Run locally (sequential):
    python -m runners.sweeps.inner_tooth_size

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash dgx/deploy_dgx.sh --option2   # choose sweep (legacy DGX cluster)
    # bash dgx/deploy_dgx.sh --option2   # (or use the new Athena cluster) → inner_tooth_size
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.grating.n_periods_each_side = 80
BASE.mesh.simulation_mode        = "optimization"
BASE.spectral.scan_width_nm      = 16.0


SPEC = SweepSpec(
    innermost_tooth_shift_nm = [90, 130],
    n_apod_periods_each_side = [1],
    center_mod_depth_nm      = [100, 125, 150, 200],
    apod_method              = ["linear"],
    cavity_neg_detuning_nm   = [5.76],   # phase-matched detuning
    lengthen_cavity          = [True],
    label = "inner_tooth_size",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
