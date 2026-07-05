"""
Tooth-SHAPE study — does the corrugation profile (not its strength) change the
radiation loss, for TM and TE?

Physics expectations (docs/loss_reduction_research_2026-07-03.md + k-space
analysis, stated up front so the result is judged honestly):
  * TM in-plane loss is ENVELOPE-driven (the cavity k-tail crossing the cladding
    light line): first-order diffraction of this sub-lambda grating is
    evanescent, so smooth teeth should change TM loss only weakly.
  * TE loss is per-tooth vertical scattering (corner-field / high-k driven), so
    smooth profiles have a real shot at reducing it at fixed kappa.
  * Wall-phase offset: the two-source model says aligned walls are already
    optimal (a single offset nulls only ONE of the +-y lobes and costs kappa
    faster than it cuts loss). Included as a 2-row FALSIFICATION test of that
    model, with its own symmetry-off control (BC change moves absolute numbers).

kappa-matching (the user's hard constraint is the SPATIAL MODE WIDTH): a smooth
profile's Bragg kappa is its fundamental Fourier coefficient — for the same
peak-to-peak depth, sin has 4/pi less kappa and tri 8/pi^2 less than rect. Depths
are boosted accordingly (TM: 400 -> sin 509.3 / tri 628.3 nm; TE: 300 -> sin
382.0 / tri 471.2 nm) so fwhm_m should come out ~equal; fwhm_m is recorded per
row and is part of the verdict.

Rows (zipped, 11 tasks):
  TM (anchored: pitch 516.83, h 350, n 1.97/1.444, converged box 6.8/8.8 um):
    0  rect 400  sym ON   (control)
    1  sin  509  sym ON
    2  tri  493  sym ON
    3  rect 400  sym OFF  (control for wall-phase rows)
    4  rect 400  sym OFF  wall_phase 90 deg
    5  rect 400  sym OFF  wall_phase 150 deg
  TE (baseline: pitch 500, corr 300, same box for numerics parity):
    6  rect 300  sym ON   (control)
    7  sin  382  sym ON
    8  tri  370  sym ON
    9  rect 300  sym OFF  (control)
   10  rect 300  sym OFF  wall_phase 90 deg
Window: shared 1558.5 / 60 nm / 4001 pts (TM res ~1558.6, TE ~1558.9; smooth
profiles keep the same avg width so shifts stay small). Optimization mesh
(survey; winners get an accurate pass). NOTE the smooth-profile arms end at the
avg width (800/nm) vs the 1000 nm feed — a small identical step at both far
ends, ~13 periods outside the mirrors; own controls make all Delta's honest.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER tm_width_lightline in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.shape_study
Output -> results/shape_study/results/.
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
# Narrow window around the known resonances (TM 1558.6 / TE ~1558.9 nm); smooth
# profiles keep the same avg width so shifts stay small.
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# kappa-matched depths. Fundamental Fourier coefficient per unit peak-to-peak
# depth d: rect (square wave) (4/pi)*(d/2); sin (d/2); tri (8/pi^2)*(d/2).
#   d_sin = d_rect * 4/pi        d_tri = d_rect * (4/pi)/(8/pi^2) = d_rect * pi/2
_PI = 3.14159265358979
_TM_SIN = round(400.0 * 4.0 / _PI, 1)                 # 509.3
_TM_TRI = round(400.0 * _PI / 2.0, 1)                 # 628.3
_TE_SIN = round(300.0 * 4.0 / _PI, 1)                 # 382.0
_TE_TRI = round(300.0 * _PI / 2.0, 1)                 # 471.2

_pol    = ["TM"] * 6 + ["TE"] * 5
_prof   = ["rect", "sin", "tri", "rect", "rect", "rect",
           "rect", "sin", "tri", "rect", "rect"]
_corr   = [400.0, _TM_SIN, _TM_TRI, 400.0, 400.0, 400.0,
           300.0, _TE_SIN, _TE_TRI, 300.0, 300.0]
_pitch  = [516.83] * 6 + [500.0] * 5
_wp     = [0.0, 0.0, 0.0, 0.0, 90.0, 150.0,
           0.0, 0.0, 0.0, 0.0, 90.0]
_ysym   = [True, True, True, False, False, False,
           True, True, True, False, False]

_n = len(_pol)
assert _n == 11
assert all(len(v) == _n for v in (_prof, _corr, _pitch, _wp, _ysym))
assert len(set(zip(_pol, _prof, _corr, _wp, _ysym))) == _n, "duplicate row -> tag collision"
# Wall-phase rows must run symmetry-off (builder raises otherwise) — verify spec-side too.
assert all((w == 0.0) or (not s) for w, s in zip(_wp, _ysym))

SPEC = SweepSpec(
    polarization         = _pol,
    corr_profile         = _prof,
    corrugation_depth_nm = _corr,
    pitch_nm             = _pitch,
    wall_phase_deg       = _wp,
    use_y_symmetry       = _ysym,
    mode  = "zipped",
    label = "shape_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
