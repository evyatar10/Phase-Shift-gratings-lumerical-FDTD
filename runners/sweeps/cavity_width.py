"""
Sweep: cavity-segment width W_cavity.

Probes whether the heuristic "average width" (= avg_corrugation_width_m, 800 nm by default)
is actually optimal for peak on-resonance transmission and Q. Cavity length is left at the
plain half-pitch (cavity_neg_detuning_nm = 0): cavity length only shifts the resonance
wavelength, not peak T or Q.

Sweep values: 700, 750, 790, 800, 810, 850, 900 nm
  - 800 nm           = current default (avg)
  - 790, 810 nm      = +/-10 nm fine pair (local curvature near presumed optimum)
  - 750, 850 nm      = +/-50 nm linear step
  - 700, 900 nm      = +/-100 nm linear step
  All inside [W_narrow=650, W_wide=950] for the default corrugation_depth=300 nm.

Run locally (sequential):
    python -m runners.sweeps.cavity_width

Run on Athena as a parallel SLURM array (one task per width):
    bash athena/deploy_athena.sh --option2   # choose sweep -> cavity_width
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


SPEC = SweepSpec(
    cavity_width_nm        = [700, 750, 790, 800, 810, 850, 900],
    cavity_neg_detuning_nm = [0.0],
    label = "cavity_width",
)


if __name__ == "__main__":
    base = SimulationConfig()
    base.grating.n_periods_each_side = 80
    base.mesh.simulation_mode        = "optimization"
    base.spectral.scan_width_nm      = 16.0

    run_sweep_spec(SPEC, target="local", base=base)
