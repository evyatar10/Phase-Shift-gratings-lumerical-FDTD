"""
TM device-2 corrugation DETUNE at LONGITUDINAL STAGGER Dx = 6 um, gaps 3 and 4 um.

Companion to side_by_side_tm_detune_400nm_ext.py (same corrugation knob, but at Dx=0).
Here the second device is offset Dx = 6 um along the propagation direction and the
device-2 corrugation is swept over the large-detune range 400..650 nm (50 nm steps),
at two gaps. This probes whether the longitudinal stagger (which mixes in the radiative
/ forward-lobe coupling channel rather than the pure broadside evanescent overlap)
changes how device-2 detuning loads device 1.

  device_gap_nm          : {3000, 4000}                       (Dy = 3, 4 um -> 2 options)
  device_stagger_nm      : {6000}                             (Dx = 6 um)
  corrugation_depth_2_nm : {400, 450, 500, 550, 600, 650}     (device-2 depth, 400->650 step 50)
  polarization           : {TM}
  => 2 x 1 x 6 = 12 cartesian points (one SLURM array task each).

Device-1 stays at its 400 nm TE-matched corrugation. Device + window inherited
bit-identical from side_by_side_tm_400nm.build_base (pitch 516.83, N=80, n 1.97/1.444,
30 nm window at 1558.5 nm). Geometry: avg width 800 nm -> width_narrow_2 = 800 - corr2/2;
deepest corr2 = 650 nm -> narrow 475 nm (wide 1125 nm), physical.

Filenames encode ..._Ygap{gap}nm_Xstag{stag}nm_..._corr2{d2}nm, so all 12 points are
DISTINCT and do not collide with the Dx=0 runs.

Output -> results/side_by_side_tm_detune_400nm_stag6/results/ ->
results_from_athena/side_by_side_tm_detune_400nm_stag6/.

Run on Athena (default partition; SERIALIZE after the Dx=0 ext sweep finishes — the
two --option3 jobs share data/sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_detune_400nm_stag6
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
    device_gap_nm          = [3000, 4000],                   # Dy = 3, 4 um
    device_stagger_nm      = [6000],                         # Dx = 6 um
    corrugation_depth_2_nm = [400, 450, 500, 550, 600, 650], # device 1 fixed at 400
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_detune_400nm_stag6",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
