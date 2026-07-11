"""
TM device-2 corrugation DETUNE — extended large-corr range, at gaps 3 and 4 um.

Follow-up to side_by_side_tm_detune_400nm.py. There the device-2 corrugation was
swept only +-40 nm around the matched 400 nm (360..440) at gaps 1/2/3 um, Dx=0. The
result: detuning is a weak lever because +-40 nm shifts device-2's resonance by < ~1 nm,
far less than the coupling splitting. To actually push device-2's resonance PAST the
splitting we need a much larger mismatch. This file extends the device-2 corrugation to
650 nm in 50 nm steps and adds a second gap (4 um):

  device_gap_nm          : {3000, 4000}                                       (Dy = 3, 4 um)
  device_stagger_nm      : {0}                                               (Dx = 0)
  corrugation_depth_2_nm : {360, 380, 400, 420, 440, 450, 500, 550, 600, 650} (device-2 depth)
  polarization           : {TM}
  => 2 x 1 x 10 = 20 cartesian points (one SLURM array task each).

Per the user's request:
  - gap 3 um: extends the existing 360..440 run with the larger 450..650 points (the
    360..440 columns are recomputed here so this run's folder holds the full 10-point
    curve self-contained; they reproduce the prior matched/near-matched results).
  - gap 4 um: the FULL 10-point curve (a brand-new gap; all 10 are new).

Device-1 stays at its 400 nm TE-matched corrugation. Device + window are inherited
bit-identical from side_by_side_tm_400nm.build_base (pitch 516.83, N=80, n 1.97/1.444,
30 nm window at 1558.5 nm). Geometry check: avg waveguide width is 800 nm, so
width_narrow_2 = 800 - corr2/2; at the deepest corr2 = 650 nm the narrow section is
475 nm (wide 1125 nm) — physical, no dead device.

The generate_file_tag naming encodes ..._Ygap{gap}nm_..._corr2{d2}nm, so all 20 points
get DISTINCT .mat/.h5 filenames (no clobbering).

Output -> results/side_by_side_tm_detune_400nm_ext/results/ (RUN_NAME = module name) ->
results_from_athena/side_by_side_tm_detune_400nm_ext/. Merge the .mat into
side_by_side_tm_detune_400nm/results before re-plotting the combined gap curves
(the _2pishift filenames encode the gap+corr2, so there are no collisions across runs).

Run on Athena (default partition; serialize after any in-flight side-by-side sweep —
shared data/sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_detune_400nm_ext
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
    device_gap_nm          = [3000, 4000],                                       # Dy = 3, 4 um
    device_stagger_nm      = [0],                                                # Dx = 0
    corrugation_depth_2_nm = [360, 380, 400, 420, 440, 450, 500, 550, 600, 650], # device 1 fixed at 400
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_detune_400nm_ext",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
