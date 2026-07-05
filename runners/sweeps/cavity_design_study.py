"""
CAVITY-DESIGN study — the user's 4-way comparison + shape refinement + TE check.

Questions answered (one row each, all vs in-study controls, converged box):
 A. Is the barrel just "more dielectric at the defect"? Compare the barrel-150
    against a plain RECT cavity widened to the SAME ADDED AREA (barrel adds
    (2/pi)*150 = 95.5 nm average -> rect 895.5 nm) and to the same PEAK (950 nm).
    If rect-wide matches, the fab-corner question is moot (no curved walls
    needed at all).
 B. User's "barrel with widened cavity": barrel-150 on a 950 nm base (peak 1100).
 C. Fab-friendly profiles at depth 300: HANN (raised-cosine, ZERO slope at the
    tooth junctions - no corner) and GAUSS, vs the half-sine (same depth, from
    barrel_followup's opt ladder the sin-300 point is interpolated 225/400 -> a
    dedicated sin-300 row is included for an exact same-numerics comparison).
 D. Does the barrel help TE? (TE loss is per-tooth vertical, not defect-local ->
    prediction: much weaker effect. Measured, not assumed.)

Rows (zipped, 11 tasks, optimization mesh, window 1558.5/40 nm/3001 pts):
  0  TM control (rect 800 cavity)
  1  TM rect cavity 895.5   (equal-AREA to barrel 150)
  2  TM rect cavity 950     (equal-PEAK to barrel 150)
  3  TM barrel 150          (reference, same numerics)
  4  TM barrel 150 on 950 base (user's combo, peak 1100)
  5  TM sin  300            (profile reference)
  6  TM hann 300            (no-corner profile)
  7  TM gauss 300           (no-corner profile)
  8  TE control
  9  TE barrel 300
 10  TE hann 300
Anchored TM (pitch 516.83/corr 400) and TE baseline (pitch 500/corr 300).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_design_study
Output -> results/cavity_design_study/results/.
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
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_pol    = ["TM"] * 8 + ["TE"] * 3
_pitch  = [516.83] * 8 + [500.0] * 3
_corr   = [400.0] * 8 + [300.0] * 3
#          ctrl    rectA   rectP   barrel  barrel+  sin300  hann    gauss   TEc    TEbar   TEhann
_cwid   = [None,   895.5,  950.0,  None,   950.0,   None,   None,   None,   None,  None,   None]
_cshape = ["rect", "rect", "rect", "barrel", "barrel", "barrel", "hann", "gauss",
           "rect", "barrel", "hann"]
_cdepth = [0.0,    0.0,    0.0,    150.0,  150.0,   300.0,  300.0,  300.0,  0.0,   300.0,  300.0]

_n = 11
assert all(len(v) == _n for v in (_pol, _pitch, _corr, _cwid, _cshape, _cdepth))
assert len({(p, w, s, d) for p, w, s, d in zip(_pol, _cwid, _cshape, _cdepth)}) == _n, \
    "duplicate row -> tag collision"

SPEC = SweepSpec(
    polarization          = _pol,
    pitch_nm              = _pitch,
    corrugation_depth_nm  = _corr,
    cavity_width_nm       = _cwid,
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    mode  = "zipped",
    label = "cavity_design_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
