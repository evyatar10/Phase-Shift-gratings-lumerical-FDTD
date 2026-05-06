"""
Example study: vary apodization periods AND innermost-tooth shift together.

This file is the new pattern: keep one .py per investigation (so you can still
"open this file and remember what you wanted"), but the body is just a
SweepSpec listing the fields that vary. Everything else uses SimulationConfig
defaults. Cartesian product by default — change mode='zipped' for parallel
arrays of values.

Run locally (sequential):
    python -m runners.sweeps.apod_and_shift

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash dgx/deploy_dgx.sh --option2   # choose sweep (legacy DGX cluster)
    # bash dgx/deploy_dgx.sh --option2   # (or use the new Athena cluster) → apod_and_shift
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.mesh.simulation_mode    = "optimization"
BASE.grating.n_periods_each_side = 80
BASE.spectral.scan_width_nm  = 20.0


SPEC = SweepSpec(
    n_apod_periods_each_side  = [2, 3],
    innermost_tooth_shift_nm  = [0],
    apod_method               = ["linear"],
    cavity_neg_detuning_nm    = [5.76],   # phase-matched detuning
    label = "apod_and_shift",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
