"""
N=1300/side pi-shift Bragg grating — the user's target device, exact parameters:
h = 200 nm, avg width 1800 nm, corrugation 400 nm (teeth 1600/2000), pitch 504 nm,
TM, accurate mesh, span_mult 5.0 box (box-converged per locator study).

Study dir: results/tm_h200_w1800_p504_n1300/      Job ID(s): probe + main TBD
Date: 2026-07-31
Purpose: full-size device (2600 periods, 1.31 mm grating). Locator study
(tm_h200_w1800_p504_locator, array 127309) measured: resonance 1492.122 nm
(N-independent, box-independent to 1 pm), stopband ~1488.4-1495.9 nm, radiative
loss 1.32% at N=300 => radiation-limited Q ~ 2.8e5 => full ring-down ~2 ns =
multi-GPU-day. Athena kills jobs at 23:30 h and an unfinished FDTD job writes
NOTHING, so the sim time is explicitly budgeted (TM_SIM_TIME_PS below):

  Phase 1 (probe):  SIM_TIME_PS = 20  — measures wall-seconds per simulated ps
    and node build time at the true 1.3 mm grid; spectrum already locates the
    N=1300 resonance to ~0.4 nm (truncation-limited resolution).
  Phase 2 (main):   SIM_TIME_PS raised to the largest value that provably
    finishes in <= ~19 h solve, computed from the probe. Gives exact lambda_res,
    stopband, T; Q as a LOWER BOUND (truncation-broadened) unless the budget
    reaches the full ring-down. Full Q ~ 2.8e5 needs ~2 ns => only via chained
    jobs (infra that does not exist yet) — decision deferred to the user.

NOTE: probe and main share the same file tag (sim time is not in
generate_file_tag), so the main run OVERWRITES the probe's .mat/.fsp — intended;
the two phases are strictly sequential (never queued together).

Scan window: phase 1 (probe) used 16 nm / 3001 pts (5.3 pm/pt) — survey.
Phase 2 (main): 2 nm, center 1492.12 (user 2026-07-31: same 3001 points, just a
narrow range) -> 0.67 pm/pt: even a fully-resolved Q~2.8e5 line (~5 pm) gets
~8 points. The stopband context is deliberately dropped — it is N-independent
and already measured at N=300 (locator study). NOTE: sampling density does not
beat the record-length limit — the measured linewidth is still floored at
~lambda^2/(c*T_sim) by truncation; the narrow window just guarantees the line is
never sampling-limited.

Dispatch (both phases; edit SIM_TIME_PS between them):
  SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --gpu=a100 \
      --spec=runners.sweeps.tm_h200_w1800_p504_n1300
Phase 2 additionally needs athena.conf temporarily set to ARRAY_QOS=4d_1g +
ARRAY_TIME=95:00:00 for the submit (revert right after — do not leave routine
sweeps on the low-priority 4-day QOS).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig

# Simulated-time budget in ps (read by bragg_device at build time on the node).
# Phase-1 probe ran at 20 (job 127321): MEASURED 600 wall-s per simulated ps on
# the a100 => phase 2 = 500 ps ~ 83 h solve, fits the 4d_1g QOS (95 h) with
# ~11 h margin. User picked 4d_1g on a100-public 2026-07-31. Linewidth floor
# lambda^2/(c*500ps) = 14.8 pm => spectrum resolves Q up to ~1.0e5; beyond that
# the log's Auto-Shutoff decay slope gives the time-domain Q estimate.
SIM_TIME_PS = 500
os.environ["TM_SIM_TIME_PS"] = str(SIM_TIME_PS)


BASE = SimulationConfig()
BASE.geometry.core_height_m = 200e-9
BASE.geometry.avg_corrugation_width_m = 1800e-9
BASE.geometry.corrugation_depth_m = 400e-9
BASE.geometry.width_port_m = 1800e-9
BASE.grating.pitch_m = 504e-9
BASE.source.polarization = "TM"
BASE.spectral.center_wavelength_m = 1.4921e-6
# 6 nm (1489.1-1495.1): probe showed the 20-ps spectrum peaking at the upper
# band-edge lobe (1494.9) — 6 nm keeps BOTH the defect (1492.12) and that edge
# in-window at 2 pm/pt (>=7 pts across the 14.8 pm truncation floor).
BASE.spectral.scan_width_nm = 6.0
BASE.monitors.record_2d_fields = False
# Probe 127321 segfaulted in getresult("field_profile") — 501 pts x 1.31 mm is
# ~28 GB. 3 points keeps the spatial-envelope FWHM deliverable and is ~170 MB.
BASE.monitors.field_profile_freq_points = 3
BASE.farfield.enabled = False


SPEC = SweepSpec(
    n_periods_each_side = [1300],
    simulation_mode     = ["accurate"],
    span_mult           = [5.0],
    mode = "zipped",
    label = "tm_h200_w1800_p504_n1300",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
