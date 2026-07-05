"""
RECT-CAVITY WIDTH ladder — find the optimum of the simplest winning knob.

cavity_design_study (job 117434) proved the barrel-cavity effect is a pure
ADDED-DIELECTRIC-AREA effect: a plain RECTANGULAR cavity widened to the same
area performs identically (rect 895.5 == barrel 150; rect 950 == hann 300), and
no saturation was visible up to the equivalent of ~1050 nm (-29%). This ladder
extends the rectangle to find the turnover (expected eventually: the wide
segment goes effectively multimode laterally / detunes the defect phase).

Rows (zipped, 7 tasks, optimization mesh, converged box):
  0  TM control (rect 800 cavity)             window 1558.5/40
  1-4  rect cavity 1050 / 1150 / 1250 / 1400  window 1558.5/40
  5  W1000/corr500 control                    window 1573/40
  6  W1000/corr500 + rect cavity 1250         window 1573/40  (fab-simple STACK:
     replaces the barrel-190 stack row; expected ~ -46% or better vs anchored)

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_width_ladder
Output -> results/cavity_width_ladder/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_cwid   = [None,   1050.0, 1150.0, 1250.0, 1400.0, None,   1250.0]
_width  = [800.0] * 5                          + [1000.0, 1000.0]
_corr   = [400.0] * 5                          + [500.0,  500.0]
_center = [1558.5] * 5                         + [1573.0, 1573.0]

_n = 7
assert all(len(v) == _n for v in (_cwid, _width, _corr, _center))
assert len({(c, w) for c, w in zip(_cwid, _width)}) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm      = _cwid,
    avg_width_nm         = _width,
    corrugation_depth_nm = _corr,
    center_wavelength_nm = _center,
    mode  = "zipped",
    label = "cavity_width_ladder",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
