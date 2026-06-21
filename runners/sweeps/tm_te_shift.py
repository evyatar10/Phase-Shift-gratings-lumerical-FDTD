"""
Study: innermost-tooth-shift sweep for the pi-shift Bragg grating — TE vs TM.

Sweep the innermost-tooth shift (the X-shift of the tooth nearest the cavity,
which shortens the narrow-tooth segments) for both polarizations, on the same
baseline device used by the TM/TE comparison and the apodization study
(runners/tm/_tm_vs_te_common.build_base_cfg). This is the shift analog of
runners/sweeps/tm_te_apod.py.

  innermost-tooth shift : {50, 100, 150, 200} nm
                          (250 is excluded — at half_pitch=250 nm a 250 nm shift
                           makes the narrow tooth vanish / teeth touch)
  polarization          : {TE, TM}
  => 4 × 2 = 8 cartesian points (one SLURM array task each).

Shift=0 is NOT re-run — we already have that result. deploy_athena.sh copies the
existing baseline files (run_tm_vs_te result_N80_avg_te.mat / result_N80_TM_avg_tm.mat)
into this study's results folder before plotting, where they read as the shift=0
point (no `_S` tag). So the summary plots include 0 without recomputing it.

Both polarizations run at pitch 500 nm (no pitch correction): TE resonates
≈1570.7 nm, TM ≈1523.6 nm, both inside the 150 nm scan window. Far-field is OFF
and the 2D cross-section field profiles (XY/YZ/XZ) are ON, so the transverse
domain stays at 1.8·λ and fwhm_m (spatial mode width) is recorded.

Run locally (sequential):
    python -m runners.sweeps.tm_te_shift

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_te_shift

Outputs land in <BASE_SAVE_DIR>/tm_te_shift/results/ on Athena (athena_run_one.py
sets RUN_NAME from the module short name) and sync to
results_from_athena/tm_te_shift/results/ via `--results-no-fsp`. After the array
finishes, deploy_athena.sh queues an afterok job (run_shift_summary.sh) that runs
plot_tm_te_shift.py to write the transmission/mode-width summary PNGs into the
same folder, so they download alongside the .mat files.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


# Baseline = the TM/TE comparison device (pitch 500, N=80, constant materials,
# optimization mesh). Reused verbatim so the 0-shift point reproduces the
# existing baseline and the shifted points are directly comparable.
BASE = build_base_cfg(SimulationConfig())
# 150 nm window (1475–1625 nm) holds both the TE (~1571) and TM (~1524)
# resonances at pitch 500 — identical to the apodization study scan.
BASE.spectral.center_wavelength_m = 1.550e-6
BASE.spectral.scan_width_nm       = 150.0
BASE.spectral.n_wl_points         = 6001
# Record the 2D cross-section field profiles (XY top, YZ cross, XZ side);
# no far-field. Far-field OFF keeps the transverse box at 1.8·λ.
BASE.monitors.record_2d_fields    = True
BASE.farfield.enabled             = False
# span_multiplier_override stays None → _span_multiplier == 1.8 (far-field off).


SPEC = SweepSpec(
    innermost_tooth_shift_nm = [50, 100, 150, 200],   # 0 reused from baseline; 250 excluded (teeth touch)
    polarization             = ["TE", "TM"],
    label = "tm_te_shift",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
