"""
Study: TM apodization sweep at the TM-corrected pitch (518.3 nm) — linear AND tanh.

Re-run of the TM half of the apodization study, but at the pitch that re-centers
the TM resonance on ~1571 nm (518.3 nm) instead of the 500 nm pitch used by
runners/sweeps/tm_te_apod.py + tm_te_apod_tanh.py (where TM sits at ~1524 nm).
This makes the TM apodization curves directly comparable to TE at the same
wavelength. TE is unchanged — keep using the existing pitch-500 TE results.

Sweeps BOTH apodization profiles in one job array:
  apodized teeth/side : {2, 5, 10, 20}
  apod_method         : {linear, tanh}     (tanh_steepness = 2.0)
  polarization        : {TM}
  => 4 × 2 = 8 cartesian points (one SLURM array task each).

The 0-teeth (no-apodization) point is REUSED from the existing 518.3 TM baseline
results_from_athena/run_tm/results/result_N80_TM_avg_tm_P518p3_fields_smp.mat
(pitch 518.3, λ≈1570.5 nm, T≈0.958, fwhm_m≈19.0 µm) — same as the original study
reused a 0-teeth baseline rather than re-running it.

Same baseline device as the apodization study (N=80, constant materials n_core
1.9963 / n_clad 1.444, optimization mesh) and the same 150 nm window / 6001 pts /
record_2d_fields=True / far-field OFF (→ transverse box 1.8·λ); only the pitch
differs. Result filenames do NOT encode pitch, so the distinct RUN_NAME folder
(tm_apod_pitch518) keeps them from colliding with the pitch-500 study files.
Filenames: linear → result_N80_A{2,5,10,20}_M4_TM_avg.mat;
           tanh   → result_N80_A{2,5,10,20}_th_M4_TM_avg.mat.

Run locally (sequential):
    python -m runners.sweeps.tm_apod_pitch518

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_apod_pitch518

Outputs land in <BASE_SAVE_DIR>/tm_apod_pitch518/results/ on Athena and sync to
results_from_athena/tm_apod_pitch518/results/ via `--results-no-fsp`.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


# Baseline = the TM/TE comparison device, but pinned to the TM-calibrated pitch
# (518.3 nm) so the TM resonance sits at ~1571 nm (mirrors tm_periods_match_te.py).
BASE = build_base_cfg(SimulationConfig())
BASE.grating.pitch_m = 518.3e-9
# 150 nm window (1475–1625 nm) — same as the apodization study; comfortably
# holds the TM resonance (~1570.5 nm at pitch 518.3).
BASE.spectral.center_wavelength_m = 1.550e-6
BASE.spectral.scan_width_nm       = 150.0
BASE.spectral.n_wl_points         = 6001
# Record the 2D cross-section field profiles (XY top, YZ cross, XZ side);
# no far-field. Far-field OFF keeps the transverse box at 1.8·λ.
BASE.monitors.record_2d_fields    = True
BASE.farfield.enabled             = False
# span_multiplier_override stays None → _span_multiplier == 1.8 (far-field off).


SPEC = SweepSpec(
    n_apod_periods_each_side = [2, 5, 10, 20],
    polarization             = ["TM"],
    apod_method              = ["linear", "tanh"],
    tanh_steepness           = [2.0],     # tanh S-curve steepness (ignored for linear)
    center_mod_depth_nm      = [4.0],     # apodization floor at the cavity (nm)
    label = "tm_apod_pitch518",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
