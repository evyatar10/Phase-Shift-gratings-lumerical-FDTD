"""
TE corrugation-DETUNING sweep at the clean 300 nm / n=1.97 settings.

Companion to side_by_side_te_300nm.py (the gap × stagger distance grid). Here the
geometry is FIXED at the MOST-COUPLED TE point and only device 2's corrugation
depth is swept, to test whether detuning the second cavity changes device-1's
input->output transmission (the Friedrich-Wintgen / resonant-drain knob).

Why gap 1 µm, Δx = 0:
  TE side-by-side coupling is EVANESCENT (directional-coupler-like), so it is
  strongest at the smallest gap and at ZERO longitudinal stagger (maximum overlap).
  The distance grid (side_by_side_te_300nm) confirms it — at gap 1 µm the device-1
  drain is worst (T ~ 0.18) near Δx = 0 and eases with stagger. So gap 1 µm / Δx 0
  is where device 2 couples to device 1 hardest, i.e. where corrugation detuning has
  the most leverage to either kill the resonant drain (recover T) or reveal an
  interference (T peak). A large Δx would decouple the guides and wash the effect out.

  device_gap_nm          : {1000}                                  (fixed, nm)
  device_stagger_nm      : {0}                                     (fixed, max TE overlap)
  corrugation_depth_2_nm : {150, 200, 250, 300, 350, 400, 450}     (device-2 depth, nm)
  polarization           : {TE}
  => 7 cartesian points. Device 1 stays at 300 nm (build_base); 300 nm is the
     MATCHED reference (expected worst drain), bracketed by weaker/stronger device 2.

Inherits the clean baseline from side_by_side_te_300nm.build_base: pitch 500, N=80,
n_core 1.97 / n_clad 1.444, device-1 corrugation 300 nm, y-symmetry off, ports-only,
20 nm window centered on the 1558.9 nm defect (device-1 resonance is fixed there;
detuning device 2 only perturbs it by a few nm, well inside the window).

Output -> results/side_by_side_te_detune_300nm/results/ (RUN_NAME = module name) ->
results_from_athena/side_by_side_te_detune_300nm/.

Run on Athena (default partition; serialize after the distance grid finishes —
shared data/sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_te_detune_300nm
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_te_300nm import build_base


BASE = build_base()


# Most-coupled TE point (evanescent coupling peaks at min gap, zero stagger).
REP_GAP_NM     = 1000.0
REP_STAGGER_NM = 0.0


SPEC = SweepSpec(
    n_devices              = [2],
    device_gap_nm          = [REP_GAP_NM],
    device_stagger_nm      = [REP_STAGGER_NM],
    corrugation_depth_2_nm = [150, 200, 250, 300, 350, 400, 450],   # device 1 fixed at 300 (matched = 300)
    polarization           = ["TE"],
    mode  = "cartesian",
    label = "side_by_side_te_detune_300nm",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
