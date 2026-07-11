"""
CAVITY COMBO study — the user's question: does the SHAPE add anything on top of
the optimally-sized cavity? (Anchored device untouched: pitch 516.83, corr 400,
W800 — modifications to the CAVITY SEGMENT ONLY.)

Measured inputs: amount optimum = rect ~1050 nm (-30% @ +0.6% fwhm); at fixed
amount, shapes lean-toward-the-narrow-tooth win slightly (tri7 -19% vs barrel
-17% at area A0); a zero-area TILT in that orientation gives -10% alone.
Question: do the shape effects survive ON the 1050 base, or is the optimum
purely scalar?

Rows (zipped, 7 tasks, optimization mesh, converged box, window 1558.5/40/3001):
  0 control (rect 800)
  1 rect 1050 (in-study reference — the current champion)
  2 barrel  75 on 1050 base  (a bit MORE area than optimal: equiv ~1098)
  3 barrel 150 on 1050 base  (equiv ~1145 — past the optimum, expect worse)
  4 tri7   191 on 1050 base  (same extra area as row 3, leaned to narrow side)
  5 tilt 300 on 1050 base    (900 -> 1200 ramp, ZERO extra area vs row 1)
  6 tilt 150 on 1050 base    (975 -> 1125 ramp, zero extra area)
Registered predictions: rows 2-4 at/below row 1 (amount already optimal);
rows 5-6 the interesting ones — if the tilt's -10% mechanism is independent,
tilt-on-1050 beats the plain 1050.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER distributed_shift_study in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_combo_study
Output -> results/cavity_combo_study/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_cwid   = [None,   1050.0, 1050.0,  1050.0,  1050.0, 1050.0, 1050.0]
_cshape = ["rect", "rect", "barrel", "barrel", "tri7", "tilt", "tilt"]
_cdepth = [0.0,    0.0,    75.0,    150.0,   191.0,  300.0,  150.0]

_n = 7
assert len(_cwid) == len(_cshape) == len(_cdepth) == _n
assert len(set(zip(_cwid, _cshape, _cdepth))) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm       = _cwid,
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    mode  = "zipped",
    label = "cavity_combo_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
