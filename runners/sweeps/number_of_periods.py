"""
Sweep: number of grating periods on each side of the cavity.

Run locally (sequential):
    python -m runners.sweeps.number_of_periods

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash dgx/deploy_dgx.sh --option2   # choose sweep (legacy DGX cluster)
    # bash dgx/deploy_dgx.sh --option2   # (or use the new Athena cluster) → number_of_periods
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.mesh.simulation_mode = "optimization"


SPEC = SweepSpec(
    n_periods_each_side = [80, 100, 120],
    label = "number_of_periods",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
