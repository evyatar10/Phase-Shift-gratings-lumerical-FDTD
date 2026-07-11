"""
TM corrugation-DETUNING sweep at the 400 nm / pitch 516.83 / n=1.97 device.

Companion to side_by_side_tm_400nm.py (the gap x stagger distance grid). There BOTH
cavities carried the same 400 nm corrugation. Here device 1 stays at its 400 nm
TE-matched value and only device 2's corrugation depth is detuned AROUND 400 nm, to
see how the coupled-cavity response (device-1 drain, supermode splitting, power
coupled into device 2) reacts to a small mismatch of the second cavity. This is the
direct TM analog of side_by_side_te_detune_300nm.py, extended from one gap to three.

Why Dx = 0, three gaps:
  We hold the longitudinal stagger at ZERO (maximum overlap of the two cavities) and
  sweep the detuning at three lateral gaps Dy = 1, 2, 3 um. TM couples longer-range
  than TE (the distance grid still showed split supermodes out to ~2 um), so the three
  gaps span strongly-coupled (1 um) to nearly-decoupled (3 um), letting us watch how
  corrugation detuning loses its leverage as the cavities separate.

  device_gap_nm          : {1000, 2000, 3000}             (Dy = 1, 2, 3 um)
  device_stagger_nm      : {0}                            (fixed, max overlap)
  corrugation_depth_2_nm : {360, 380, 400, 420, 440}      (device-2 depth, nm)
  polarization           : {TM}
  => 3 x 1 x 5 = 15 cartesian points (one SLURM array task each). Device 1 stays at
     400 nm (build_base); 400 nm is the MATCHED reference (also reproduces the prior
     gap x stagger run at stagger 0, gaps 1/2/3 um), bracketed +-40 nm in 20 nm steps.

Inherits the device + window verbatim from side_by_side_tm_400nm.build_base: pitch
516.83 nm, N=80, n_core 1.97 / n_clad 1.444, device-1 corrugation 400 nm, height 350 nm
(project default), y-symmetry off, ports-only monitors, narrow 30 nm window centered on
the 1558.5 nm defect. Device 2 keeps the shared global pitch; only its corrugation
changes. Detuning device 2 by +-40 nm shifts its resonance by well under a nm, so the
defect stays inside the window.

The generate_file_tag naming encodes ..._corr1{d1}nm_corr2{d2}nm, so the five corr-2
values produce DISTINCT .mat/.h5 filenames (no clobbering).

Output -> results/side_by_side_tm_detune_400nm/results/ (RUN_NAME = module name) ->
results_from_athena/side_by_side_tm_detune_400nm/. Per task .mat: T, R, loss (device-1),
resonance_transmission / resonance_wavelength_nm / spectral_fwhm_nm / Q,
coupling_left/right/total + loss_4port, plus geometry.

Run on Athena (default partition; serialize after any in-flight side-by-side sweep —
shared data/sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_detune_400nm
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_tm_400nm import build_base   # identical TM device + window


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    device_gap_nm          = [1000, 2000, 3000],          # Dy = 1, 2, 3 um
    device_stagger_nm      = [0],                         # Dx = 0 (max overlap)
    corrugation_depth_2_nm = [360, 380, 400, 420, 440],   # device 1 fixed at 400 (matched = 400)
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_detune_400nm",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
