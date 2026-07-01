"""
TM radiative-lobe probe at Δy = 4 µm — device-2 placed near the forward-lobe stagger.

The Δx=0..6 µm stagger sweeps showed ~0 coupling into device 2 at the 3-4 µm gaps,
because the TM forward radiative lobe sits at Δx ≈ 5.7·gap (≈ 80° from broadside):
for Δy = 4 µm that is Δx ≈ 22.8 µm, far beyond the 6 µm we had sampled. The gap-1 µm
data confirmed the lobe (coupling rebounds to 0.63 right at its ~5.7-6 µm lobe).

This probes the Δy = 4 µm lobe with just the two best staggers bracketing 22.8 µm
(20 and 24 µm — 24 also covers the empirically-hinted ~6× ratio), and a few device-2
corrugation samples to see whether detuning matters WHERE there is finally coupling:

  device_gap_nm          : {4000}                (Dy = 4 µm)
  device_stagger_nm      : {20000, 24000}        (Dx = 20, 24 µm — bracket the 22.8 µm lobe)
  corrugation_depth_2_nm : {400, 500, 600}       (matched + two detuned; a few samples)
  polarization           : {TM}
  => 1 x 2 x 3 = 6 cartesian points.

Device-1 stays at 400 nm; device + window inherited from side_by_side_tm_400nm.build_base.
NOTE: a 24 µm stagger enlarges the x-domain substantially (the ~82 µm gratings still
overlap), so these sims run longer than the Δx=0 ones. Ports-only monitors keep RAM low.

Filenames encode ..._Xstag{stag}nm_..._corr2{d2}nm, distinct from all prior runs.

Output -> results/side_by_side_tm_detune_400nm_lobe/results/ ->
results_from_athena/side_by_side_tm_detune_400nm_lobe/.

Run on Athena (default partition; serialize behind any in-flight side-by-side sweep):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_detune_400nm_lobe
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_tm_400nm import build_base   # identical TM device + window


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    device_gap_nm          = [4000],                # Dy = 4 µm
    device_stagger_nm      = [20000, 24000],        # Dx = 20, 24 µm (bracket 22.8 µm lobe)
    corrugation_depth_2_nm = [400, 500, 600],       # device 1 fixed at 400; a few samples
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_detune_400nm_lobe",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
