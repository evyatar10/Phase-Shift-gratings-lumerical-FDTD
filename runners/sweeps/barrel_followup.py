"""
BARREL-CAVITY follow-up — confirm + extend the center-shape study's winner.

inner_shape_study (job 117000) found: bulging the pi-shift segment's sidewalls
(half-sine, +150 nm) cuts resonant loss by 17% relative (dT=+0.021) at +0.6%
spatial mode width and +0.08 nm resonance shift — and the depth trend 75->150
was ~linear (NOT saturated). This study answers, per CLAUDE.md §2 discipline:
 1. Does it survive ACCURATE mesh, above the jitter floor?  (rows 0-3: control,
    barrel 150, jitter partner 167.5 = half an accurate cell, barrel 300)
 2. Where is the depth optimum?           (rows 4-5: barrel 225 / 400 at
    optimization mesh — same numerics as the center study, whose control +
    barrel 75/150 points complete the ladder)
 3. Does it STACK with the width lever?   (rows 6-7: W1000/corr500 without and
    with a width-scaled barrel 190 nm, window centered on that device's
    resonance ~1573 nm)

Anchored TM device otherwise; converged box 6.8/8.8 um. NOTE: no optimization-
mesh control row here — rows 4-5 compare against the center study's control
(IDENTICAL numerics: same box, window 1558.5/40/3001, opt mesh); rows 6-7 carry
their own paired control.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.barrel_followup
Output -> results/barrel_followup/results/.
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

_cshape = ["rect",   "barrel", "barrel", "barrel",   "barrel", "barrel",   "rect",  "barrel"]
_cdepth = [0.0,      150.0,    167.5,    300.0,      225.0,    400.0,      0.0,     190.0]
_mesh   = ["accurate"] * 4                        + ["optimization"] * 2 + ["optimization"] * 2
_width  = [800.0] * 6                                                    + [1000.0, 1000.0]
_corr   = [400.0] * 6                                                    + [500.0,  500.0]
_center = [1558.5] * 6                                                   + [1573.0, 1573.0]

_n = 8
assert all(len(v) == _n for v in (_cshape, _cdepth, _mesh, _width, _corr, _center))
# unique (shape,depth,mesh,width) -> unique tags (mesh not in tag: verify no
# same-geometry row appears at both meshes)
assert len({(s, d, w) for s, d, w in zip(_cshape, _cdepth, _width)}) == _n, \
    "same geometry at two meshes -> tag collision"

SPEC = SweepSpec(
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    simulation_mode       = _mesh,
    avg_width_nm          = _width,
    corrugation_depth_nm  = _corr,
    center_wavelength_nm  = _center,
    mode  = "zipped",
    label = "barrel_followup",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
