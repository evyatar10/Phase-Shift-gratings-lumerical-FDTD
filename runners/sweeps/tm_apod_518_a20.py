"""
Recovery sweep: the A20 (20 apodized teeth/side) TM points at pitch 518.3, both
profiles — the two configs that failed in tm_apod_pitch518 (jobs 97304/97336)
when a concurrent --option3 deploy clobbered the shared remote sweep_list.txt.

Identical baseline/window/monitors to runners/sweeps/tm_apod_pitch518.py; only the
swept set is narrowed to n_apod=20 so this is a small 2-task array that can be
resubmitted/retried quickly until it lands in a clean (un-clobbered) window.

  apodized teeth/side : {20}
  apod_method         : {linear, tanh}     (tanh_steepness = 2.0)
  polarization        : {TM}
  => 1 × 2 = 2 cartesian points.

Results land in the SAME folder as tm_apod_pitch518 is NOT automatic — this module's
RUN_NAME is tm_apod_518_a20, so outputs go to results_from_athena/tm_apod_518_a20/
results/ (filenames result_N80_A20_M4_TM_avg.mat / result_N80_A20_th_M4_TM_avg.mat).
Combine with the tm_apod_pitch518 folder (A2/A5/A10) + the 518.3 0-teeth baseline
when plotting.

Run on Athena as a parallel SLURM array:
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_apod_518_a20
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


BASE = build_base_cfg(SimulationConfig())
BASE.grating.pitch_m = 516.14e-9   # TM pitch at n 1.97 (was 518.3 @ 1.9963)
BASE.spectral.center_wavelength_m = 1.550e-6
BASE.spectral.scan_width_nm       = 150.0
BASE.spectral.n_wl_points         = 6001
BASE.monitors.record_2d_fields    = True
BASE.farfield.enabled             = False


SPEC = SweepSpec(
    n_apod_periods_each_side = [20],
    polarization             = ["TM"],
    apod_method              = ["linear", "tanh"],
    tanh_steepness           = [2.0],
    center_mod_depth_nm      = [4.0],
    label = "tm_apod_518_a20",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
