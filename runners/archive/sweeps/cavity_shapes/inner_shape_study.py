"""
CENTER-shape study — reshape only the innermost tooth/teeth at the pi-shift and
measure the radiation loss empirically (USER-DIRECTED: trust FDTD, not theory).

Why the center: the TM loss is localized at the defect (paper_8 — the abrupt
envelope kink at the pi-shift is what radiates; the mirror body's first-order
diffraction is evanescent), and the defect-region radiation is exactly the part
the analytic model calls "needs FDTD". Reshaping 1-2 teeth leaves kappa and the
SPATIAL MODE WIDTH essentially untouched (fwhm_m recorded per row = the user's
hard constraint), so anything found here is nearly free.

Shapes (same footprint half-pitch x same height corr/2, both walls, x-mirrored
on both sides of the cavity — all symmetry BCs stay ON):
  ellipse    smooth dome (no corners facing the cavity)
  tri        pointy symmetric triangle
  wedge_cav  slanted face TOWARD the cavity (face tilt steers the defect emission)
  wedge_out  slanted face AWAY from the cavity
Each at n_shaped = 1 and 2 innermost teeth per side.

Rows (zipped, 13 tasks, anchored TM device, converged box y=6.8/z=8.8 um,
optimization mesh — winners get an accurate pass + jitter control):
  0        rect control
  1-2      ellipse n=1, n=2
  3-4      tri     n=1, n=2
  5-6      wedge_cav n=1, n=2
  7-8      wedge_out n=1, n=2
  9-12     CAVITY reshaped (teeth rect): barrel +75/+150 nm, hourglass -75/-150 nm
           — the pi-shift segment's own sidewall profile as an impedance-
           transition knob (never touched before; length/mirrors unchanged)
Window 1558.5 / 60 nm / 4001 pts (shaped teeth carry less SiN area than rect ->
small blue-shift expected; all rows share the window; control at identical
numerics). Target resonance ~1558.6 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.inner_shape_study
Output -> results/inner_shape_study/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8       # converged y (job 116854)
BOX_Z_MULT = 5.42    # converged z (job 116870): z = 8.8 um

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
# Narrow window around the KNOWN anchored resonance (1558.6 nm); shaped teeth
# remove a little SiN -> expect a small blue-shift, well inside 40 nm.
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# Rows 0-8: tooth shapes. Rows 9-12: the CAVITY SEGMENT itself reshaped (the
# defect antenna's own sidewall profile — barrel bulge / hourglass pinch at two
# depths; teeth stay rect).
_shapes = ["rect",
           "ellipse", "ellipse",
           "tri", "tri",
           "wedge_cav", "wedge_cav",
           "wedge_out", "wedge_out",
           "rect", "rect", "rect", "rect"]
_nsh    = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]
_cshape = ["rect"] * 9 + ["barrel", "barrel", "hourglass", "hourglass"]
_cdepth = [0.0] * 9 + [75.0, 150.0, 75.0, 150.0]

assert len(_shapes) == len(_nsh) == len(_cshape) == len(_cdepth) == 13
# ("rect",1,"rect",0) is the control; every row unique -> unique file tags.
assert len(set(zip(_shapes, _nsh, _cshape, _cdepth))) == 13

SPEC = SweepSpec(
    inner_shape           = _shapes,
    n_shaped_teeth        = _nsh,
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    mode  = "zipped",
    label = "inner_shape_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
