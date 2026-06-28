"""
Study: increase the TM period count until TM peak transmission matches TE@80.

Keep the TE reference fixed (80 periods, pitch 500 nm — run separately via
runners/tm/run_te.py). Keep every TM parameter as in the existing TM/TE
comparison (pitch 518.3 nm, constant materials, optimization mesh, same cavity)
EXCEPT the number of grating periods per side, which is swept upward. TM couples
much more weakly to the sidewall corrugation than TE, so it needs a longer
grating (more periods) to reach the same grating strength κ·L — and therefore the
same resonance peak transmission — as TE at 80 periods.

  n_periods_each_side : {80, 100, 120, 140, 160, 180, 200, 240}
  polarization        : {TM}
  => 8 cartesian points (one SLURM array task each).

N=80 reproduces the TM baseline under this study's clean spectra-only window so
the whole peak-T-vs-N curve is self-consistent. The range brackets the ~2×
grating-strength gap implied by the measured TM Q≈793 vs TE Q≈1640; if the TM
peak-T curve has not yet crossed the TE@80 value by N=240, extend the list upward.

Spectra-only (no 2D field monitors, no far-field): peak T and the spectral
Q = λ_res / spectral_fwhm_nm both come from the T(λ) spectrum, so the runs stay
light (~3.5 GB) and the transverse box stays at 1.8·λ. Pitch 518.3 nm re-centers
the TM resonance on ~1571 nm; the 12 nm window (1565–1577 nm) at ~5 pm spacing
resolves the narrowing TM FWHM at high N.

Run on Athena as a parallel SLURM array (one task per period count):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_periods_match_te

Outputs land in <BASE_SAVE_DIR>/tm_periods_match_te/results/ on Athena
(athena_run_one.py sets RUN_NAME from the module short name) and sync to
results_from_athena/tm_periods_match_te/results/ via `--results-no-fsp`.
Analyze locally with runners/sweeps/plot_tm_periods_match_te.py.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


# Baseline = the TM/TE comparison device, but pinned to the TM-calibrated pitch
# (516.14 nm, n 1.97) so the TM resonance sits at ~1558.7 nm regardless of TM_PITCH_NM.
BASE = build_base_cfg(SimulationConfig())
BASE.grating.pitch_m = 516.14e-9
# Clean, narrow spectra-only window centered on the TM resonance, fine enough to
# resolve the FWHM (hence Q) as the grating narrows it at high N.
BASE.spectral.center_wavelength_m = 1.5587e-6
BASE.spectral.scan_width_nm       = 12.0
BASE.spectral.n_wl_points         = 2401          # ~5 pm spacing across 12 nm
# Spectra only — no 2D field profiles, no far-field (keeps the box at 1.8·λ).
BASE.monitors.record_2d_fields    = False
BASE.farfield.enabled             = False


SPEC = SweepSpec(
    n_periods_each_side = [80, 100, 120, 140, 160, 180, 200, 240],
    polarization        = ["TM"],
    label = "tm_periods_match_te",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
