"""
Study: innermost-tooth-shift sweep for TM at the pitch-matched pitch 518.3 nm.

The first shift study (runners/sweeps/tm_te_shift.py) ran BOTH polarizations at
pitch 500 nm, which leaves the TM resonance at ~1524 nm (uncorrected). This rerun
redoes ONLY the TM polarization at pitch 518.3 nm — the FDE-calibrated pitch that
re-centers TM on the TE wavelength (~1571 nm), matching the rest of the project
(see results_from_athena/*/result_N80_TM_avg_tm_P518p3.mat). The TE points are
unchanged (pitch 500) and reused.

  innermost-tooth shift : {51.83, 103.66, 155.49, 207.32} nm   (0 reused)
                          = {50, 100, 150, 200} × (518.3/500), i.e. the SAME
                          RELATIVE shift (20/40/60/80% of the half-pitch) as the
                          pitch-500 TE points. The shift is scaled with the pitch
                          so TE and TM are compared at equal relative shifts.
  polarization          : {TM}
  pitch                 : 518.3 nm (pitch-matched)
  => 4 cartesian points → one SLURM array (4 tasks).

deploy_athena.sh seeds the corrected shift=0 TM baseline (result_N80_TM_avg_tm_P518p3.mat)
AND the existing pitch-500 TE points (shift 0/50/100/150/200) into this study's
results folder, then runs plot_tm_te_shift.py with a RELATIVE x-axis (shift as %
of half-pitch) — so the summary images are the corrected TE-vs-TM comparison
(TE@500 + TM@518.3) with both curves on the same 0/20/40/60/80% grid.

Run on Athena as a parallel SLURM array (one task per shift):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_shift_p518
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


# Baseline = the TM/TE comparison device, but at the pitch-matched pitch so TM
# re-centers on the TE wavelength. Everything else identical to tm_te_shift.
BASE = build_base_cfg(SimulationConfig())
BASE.grating.pitch_m              = 516.14e-9     # pitch-matched TM at n 1.97 (re-centers ~1558.7 nm)
# 150 nm window (1475–1625 nm) still holds the pitch-516.14 TM resonance (~1558.7).
BASE.spectral.center_wavelength_m = 1.550e-6
BASE.spectral.scan_width_nm       = 150.0
BASE.spectral.n_wl_points         = 6001
BASE.monitors.record_2d_fields    = True          # needed for fwhm_m (mode width)
BASE.farfield.enabled             = False


SPEC = SweepSpec(
    # Pitch-scaled shifts = {50,100,150,200}×(516.14/500) → same relative shift
    # (20/40/60/80% of half-pitch) as the pitch-500 TE points.
    innermost_tooth_shift_nm = [51.61, 103.23, 154.84, 206.46],   # 0 reused; 250 excluded
    polarization             = ["TM"],
    label = "tm_shift_p518",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
