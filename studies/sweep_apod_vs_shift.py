"""
Example study: vary apodization periods AND innermost-tooth shift together.

This file is the new pattern: keep one .py per investigation (so you can still
"open this file and remember what you wanted"), but the body is just a
SweepSpec listing the fields that vary. Everything else uses SimulationConfig
defaults. Cartesian product by default — change mode='zipped' for parallel
arrays of values.

Run locally (sequential):
    python studies/sweep_apod_vs_shift.py

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash athena/deploy_athena.sh --option3 --spec=studies.sweep_apod_vs_shift
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


SPEC = SweepSpec(
    n_apod_periods_each_side  = [0, 3, 5],
    innermost_tooth_shift_nm  = [0, 50, 100, 150],
    apod_method               = ["linear"],
    cavity_neg_detuning_nm    = [5.76],   # phase-matched detuning
    label = "apod_vs_shift",
)


if __name__ == "__main__":
    base = SimulationConfig()
    base.mesh.simulation_mode    = "optimization"
    base.grating.n_periods_each_side = 80
    base.spectral.scan_width_nm  = 20.0

    run_sweep_spec(SPEC, target="local", base=base)
