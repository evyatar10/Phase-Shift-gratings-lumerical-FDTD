"""
HANN peak-at-1050 family — the COMPLEMENT to cavity_hann_sweep (job 118529).

That study widened the Hann in the "more area" direction: every Hann peaked
ABOVE 1050 (bump on a wide base). This study fixes the PEAK at 1050 (= the
rect-1050 reference) and TAPERS the base below it, so each Hann has LESS area
than rect-1050 but SMOOTHER junctions (no abrupt 800->1050 width step where the
cavity meets the teeth; zero-slope Hann ends). It isolates the one thing the
"more area" grid cannot: can a shape MATCH rect-1050's loss with less material,
purely by smoothing the junction corner? (The junction step is what caused
rect-1050's -2.5 nm blue-shift — widening the junctions did real work — and the
pi-shift kink is the known TM radiator.)

Symmetric raised-cosine: w = base + depth·sin²(πu), peak = base + depth = 1050.

Each Hann is comparable BOTH to rect-1050 (same PEAK) and to its equal-area
rect at avg = base + depth/2 (same AREA), so junction-smoothness is separated
from area in-study, at identical numerics.

Rows (zipped, 10 tasks, opt mesh, converged box y6.8/z8.8, TM anchored
pitch 516.83/corr 400, window 1558.5/40 nm/3001 pts):
  0  rect 800          no-change control                 (expect ~0.1098)
  1  rect 1050         peak-match reference              (expect ~0.0823)
  2  hann 850 + 200    peak 1050, avg 950  (tapered, −area)
  3  hann 900 + 150    peak 1050, avg 975  (tapered)
  4  hann 950 + 100    peak 1050, avg 1000 (gentle taper)
  5  hann 1000 + 50    peak 1050, avg 1025 (near-flat)
  6  rect 950          equal-area control for row 2
  7  rect 1000         equal-area control for row 4
  8  rect 1025         equal-area control for row 5
  9  hann 852 + 200    jitter partner of row 2 (base +2 nm) — floor

Reading: rows 2–5 vs row 1 answer "does tapering the base (smoother junction,
less area) beat/hold rect-1050 at the same peak?"; rows 2,4,5 vs rects 6,7,8
answer "at equal area, does the tapered-to-1050 shape beat the flat rect?";
row 9−row 2 is the floor.

Registered prediction (honest): the program's rule is "shape helps only via
added AREA." A peak-1050 Hann has LESS area than rect-1050, so by that rule it
should be WORSE than rect-1050 (rows 2–5 > 0.0823), and roughly TIE its
equal-area rect (rows 2≈6, 4≈7, 5≈8). The interesting falsifier: a tapered Hann
that HOLDS rect-1050's loss despite less area, or beats its equal-area rect by
more than the floor — that would mean junction-smoothness, not area, and would
be the first genuine shape effect (earns an accurate-mesh confirm).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
run only AFTER 118529 drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_hann_peak1050
Output -> results/cavity_hann_peak1050/results/.
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
# Same window as cavity_hann_sweep / cavity_design_study so all rect/hann points
# reproduce exactly. Tapered cavities blue-shift ≤3 nm — well inside.
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

#          0      1      2      3      4      5      6     7      8      9
#          r800   r1050  h850/2 h900/1 h950/1 h1000/ r950  r1000  r1025  h852/2jit
_cwid   = [None,  1050., 850.,  900.,  950.,  1000., 950., 1000., 1025., 852.]
_cshape = ["rect","rect","hann","hann","hann","hann","rect","rect","rect","hann"]
_cdepth = [0.0,   0.0,   200.,  150.,  100.,  50.,   0.0,  0.0,   0.0,   200.]

_n = 10
assert all(len(v) == _n for v in (_cwid, _cshape, _cdepth))
# All Hann peaks = base + depth = 1050 (rows 2-5,9); unique (base,shape,depth).
for w, s, d in zip(_cwid, _cshape, _cdepth):
    if s == "hann":
        assert abs((w + d) - 1050.0) < 1e-6, f"hann base {w}+depth {d} != peak 1050"
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
    label = "cavity_hann_peak1050",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
