"""
Study: apodization sweep for the pi-shift Bragg grating — TE vs TM (tanh profile).

Identical to runners/sweeps/tm_te_apod.py but with a hyperbolic-tangent (tanh)
corrugation-depth taper instead of the linear one, so the two apodization profiles
can be compared. Sweep the number of apodized teeth per side (each side of the
pi-shift cavity) for both polarizations, on the same baseline device used by the
TM/TE comparison (runners/tm/_tm_vs_te_common.build_base_cfg). Apodization ramps
the corrugation depth from a small floor (center_mod_depth_nm) at the cavity up to
the full corrugation depth (300 nm) over n_apod_periods_each_side teeth, following
the tanh S-curve tanh(a*2*frac)/tanh(2*a) with a = tanh_steepness.

  apodized teeth/side : {2, 5, 10, 20}     (0 = no apodization → reuse the
                                            existing run_tm_vs_te baseline files)
  polarization        : {TE, TM}
  tanh_steepness      : 2.0  (ApodizationConfig default)
  => 4 × 2 = 8 cartesian points (one SLURM array task each).

Both polarizations run at pitch 500 nm (no pitch correction): TE resonates
≈1570.7 nm, TM ≈1523.6 nm, both inside the 150 nm scan window. Far-field is OFF
and the 2D cross-section field profiles (XY/YZ/XZ) are ON, so the transverse
domain stays at 1.8·λ (SimulationConfig._span_multiplier == 1.8 when far-field
is off and no span_multiplier_override is set).

Run locally (sequential):
    python -m runners.sweeps.tm_te_apod_tanh

Run on Athena as a parallel SLURM array (one task per cartesian point):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_te_apod_tanh

Outputs land in <BASE_SAVE_DIR>/tm_te_apod_tanh/results/ on Athena (athena_run_one.py
sets RUN_NAME from the module short name) and sync to
results_from_athena/tm_te_apod_tanh/results/ via `--results-no-fsp`.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig


# Baseline = the TM/TE comparison device (pitch 500, N=80, constant materials,
# optimization mesh). Reused verbatim so the 0-teeth point (existing
# run_tm_vs_te results) is directly comparable to the apodized points.
BASE = build_base_cfg(SimulationConfig())
# 150 nm window (1475–1625 nm) holds both the TE (~1571) and TM (~1524)
# resonances at pitch 500 — identical to the existing baseline scan.
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
    polarization             = ["TE", "TM"],
    apod_method              = ["tanh"],  # tanh taper (vs the linear study)
    tanh_steepness           = [2.0],     # tanh S-curve steepness (ApodizationConfig default)
    center_mod_depth_nm      = [4.0],     # apodization floor at the cavity (nm)
    label = "tm_te_apod_tanh",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
