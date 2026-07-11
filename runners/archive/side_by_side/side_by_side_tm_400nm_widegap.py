"""
Wide-gap EXTENSION of the TM side-by-side scan (side_by_side_tm_400nm.py).

Why this file exists
--------------------
The original TM grid stopped at Δy gap = 2.0 µm, but TM couples longer-range than
TE: at 2.0 µm the defect resonance is still split into two supermodes (~1.4 nm at
Δx=0) and the device-1 peak T (0.30–0.61) has not recovered to the isolated
single-device value (0.827). TE, by contrast, is already a single peak by ~1.8 µm.
The supermode splitting decays ~×0.71 per +200 nm of gap (≈585 nm decay length), so
TM should decouple (splitting → ~0, T → ~0.83) around 3.5–4 µm.

This adds the next gaps at the SAME 200 nm spacing, up to 3 µm, with the SAME 7
staggers, the SAME device (build_base imported from side_by_side_tm_400nm so the
geometry/window are bit-identical). Results land in their own folder; merge the
.mat into side_by_side_tm_400nm/results before re-plotting the combined grid (the
_2pishift filenames encode the gap, so there are no collisions across the two runs).

  device_gap_nm      : {2200, 2400, 2600, 2800, 3000}          (5 new gaps, 200 nm steps)
  device_stagger_nm  : {0, 1000, 2000, 3000, 4000, 5000, 6000} (same 7 staggers)
  polarization       : {TM}
  => 5 × 7 × 1 = 35 cartesian points (one SLURM array task each).

Run on Athena as a parallel SLURM array (default partition — independent tasks):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_400nm_widegap
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.side_by_side.side_by_side_tm_400nm import build_base   # identical device + window


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    corrugation_depth_2_nm = [400],                   # device 2 = device 1 (identical, 400 nm)
    device_gap_nm          = [2200, 2400, 2600, 2800, 3000],
    device_stagger_nm      = [0, 1000, 2000, 3000, 4000, 5000, 6000],
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_400nm_widegap",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
