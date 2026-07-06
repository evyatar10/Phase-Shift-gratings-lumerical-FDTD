"""
HANN-CAVITY 2D sweep — the user's question: the Hann was the best of the fixed
cavity shapes (loss 0.0838 @ base 800/peak 1100) but it lost to the plain
rect-1050 (0.0823). BUT the Hann was only ever tested on an 800 nm BASE. The
Hann keeps the tooth JUNCTIONS at base width (zero-slope ends) and only adds
area mid-cavity, so widening the RECT cavity blue-shifts the resonance −2.5 nm
(800→1050) while the Hann barely moves. That means a Hann whose BASE is raised
toward 1050 nm AND still carries a bump is a genuinely UNTESTED combination —
it stacks the rect-1050 benefit (wide junction) with extra center dielectric.

Two knobs the user named:
  • BASE width  = cavity_width_nm (the waveguide the bump sits on; W_cavity)
  • BUMP depth  = cavity_shape_depth_nm (peak = base + depth), w = base + depth·sin²(πu)

Question 1 (main): can a widened-base Hann beat rect-1050 (0.0823) / approach
the stack (0.0545, opt mesh)?
Question 2 (control): is any Hann win just its ADDED AREA? A Hann(base W,
depth d) has the same cross-sectional area as a plain rect at W + d/2 (mean of
sin² = ½). Each interesting Hann is paired with its equal-area rect so
"shape beats area?" is answered directly, in-study, at identical numerics.

Rows (zipped, 10 tasks, opt mesh, converged box y6.8/z8.8, TM anchored
pitch 516.83/corr 400, window 1558.5/40 nm/3001 pts — reproduces the two prior
studies' hann/rect points exactly for cross-checking):
  0  rect 800            in-study no-change control            (expect ~0.1098)
  1  rect 1050           the target reference                  (expect ~0.0823)
  2  hann 800 + 300      original tested point (base-800 anchor, expect 0.0838)
  3  hann 1050 + 200     ★ base raised to 1050 + bump, peak 1250
  4  hann 1050 + 400     ★ base 1050 + bigger bump, peak 1450
  5  hann 950 + 200      avg 1050 = EQUAL AREA to rect-1050, peak 1150
  6  hann 900 + 300      avg 1050 = EQUAL AREA to rect-1050, peak 1200
  7  rect 1150           equal-area control for row 3 (hann 1050+200, avg 1150)
  8  rect 1250           equal-area control for row 4 (hann 1050+400, avg 1250)
  9  hann 1052 + 200     jitter partner of row 3 (base +2 nm) — floor check

Reading: rows 5,6 vs row 1 answer "does the shape beat the plain rect-1050 at
equal area?"; rows 3,4 vs rows 7,8 answer "does base-1050 + bump beat the
equal-area rect?"; row 9−row 3 is the numerical floor. All Δ are in-study.

Registered prediction (honest): the whole program found every cavity shape that
helps only ADDS AREA and loses to an equal-area rectangle. If that holds, rows
5,6 ≥ 0.0823 (rect-1050) and rows 3,4 ≥ their equal-area rects 7,8. A surprise
(a Hann below its equal-area rect by more than the floor) would be the first
shape effect that is NOT pure area — worth an accurate-mesh confirm. Beating the
stack (0.0545) with a pure cavity shape is not expected (the stack's gain is
tooth-shift + see-saw physics a cavity shape cannot reach).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_hann_sweep
Output -> results/cavity_hann_sweep/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8       # converged y (job 116854)
BOX_Z_MULT = 5.42    # converged z (job 116870): z = 8.8 um

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
# Same window as cavity_design_study / inner_shape_study so the hann-300 and
# rect points reproduce exactly. Wide cavities blue-shift ≤3 nm — well inside.
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

#          0      1      2      3      4      5     6     7      8      9
#          r800   r1050  h8/300 h10/2  h10/4  h9/2  h9/3  r1150  r1250  h10/2jit
_cwid   = [None,  1050., None,  1050., 1050., 950., 900., 1150., 1250., 1052.]
_cshape = ["rect","rect","hann","hann","hann","hann","hann","rect","rect","hann"]
_cdepth = [0.0,   0.0,   300.,  200.,  400.,  200.,  300.,  0.0,   0.0,   200.]

_n = 10
assert all(len(v) == _n for v in (_cwid, _cshape, _cdepth))
# Uniqueness of (base, shape, depth) -> unique file tags (None==avg 800).
_keys = [(w, s, d if s != "rect" else 0.0) for w, s, d in zip(_cwid, _cshape, _cdepth)]
assert len(set(_keys)) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    polarization          = ["TM"] * _n,
    pitch_nm              = [516.83] * _n,
    corrugation_depth_nm  = [400.0] * _n,
    cavity_width_nm       = _cwid,
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    mode  = "zipped",
    label = "cavity_hann_sweep",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
