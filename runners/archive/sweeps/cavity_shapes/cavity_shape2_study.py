"""
CAVITY-SHAPE round 2 — "is there a BETTER shape than the barrel?" (user), asked
the controlled way: hold the added dielectric AREA fixed and vary only geometry,
plus a ZERO-added-area family that is pure shape.

Round-1 finding (job 117434): among centered, x-symmetric bulges (sin/hann/
gauss/rect) the effect is set by added area alone. What was NOT tested:
  * x-ASYMMETRIC placement — the device is asymmetric about the cavity (wide
    tooth on the left edge, narrow on the right), so the optimal bump may be
    off-center: tri3/tri5/tri7 = triangular bulge with apex at 30/50/70%,
    all depth 191 nm = the SAME AREA as the confirmed barrel-150.
  * material at the ENDS vs the middle: dbl2 (two bumps, apexes 25/75%),
    same area again.
  * pure shape at ZERO added area: sldn (cavity ramps 1000 -> 600 nm, an
    impedance bridge between the adjacent teeth) and slup (reverse).
Includes barrel-150 as an in-study same-numerics reference.

Rows (zipped, 8 tasks, optimization mesh, converged box, window 1558.5/40/3001):
  0 control (rect 800)     1 barrel 150 (ref)   2 tri5 191 (same area, centered)
  3 tri3 191 (apex 30%)    4 tri7 191 (apex 70%)  5 dbl2 191 (ends)
  6 sldn (zero area)       7 slup (zero area)

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER width_envelope_study in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_shape2_study
Output -> results/cavity_shape2_study/results/.
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

_cshape = ["rect", "barrel", "tri5", "tri3", "tri7", "dbl2", "sldn", "slup"]
_cdepth = [0.0,    150.0,    191.0,  191.0,  191.0,  191.0,  400.0,  400.0]
# (191 = 150*(pi/2)/... : triangle area d/2 == sine area (2/pi)*150 -> d = (4/pi)*150 = 190.99)

_n = 8
assert len(_cshape) == len(_cdepth) == _n
assert len(set(zip(_cshape, _cdepth))) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    mode  = "zipped",
    label = "cavity_shape2_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
