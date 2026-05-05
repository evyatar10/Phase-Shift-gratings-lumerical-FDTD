"""
Sweep: innermost grating tooth shift.

Run locally (sequential):
    python -m runners.sweeps.innermost_shift

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash dgx/deploy_dgx.sh --option2   # choose sweep (legacy DGX cluster)
    # bash dgx/deploy_dgx.sh --option2   # (or use the new Athena cluster) → innermost_shift
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


SPEC = SweepSpec(
    innermost_tooth_shift_nm = [100],
    cavity_neg_detuning_nm   = [5.76],   # phase-matched detuning
    lengthen_cavity          = [True],
    label = "innermost_shift",
)


if __name__ == "__main__":
    base = SimulationConfig()
    base.grating.n_periods_each_side = 80
    base.mesh.simulation_mode        = "optimization"
    base.spectral.scan_width_nm      = 20.0

    run_sweep_spec(SPEC, target="local", base=base)
