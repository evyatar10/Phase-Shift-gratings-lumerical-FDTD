"""Stage X15 — on-axis (y 0) single r 50 hole: phase circles at Lambda 531/536/545 + 0 deg period points (Athena).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): Athena 151415
Purpose (user, "run exactly the same as the two-circle program but with one circle, r 50"): mirrors
stage X2/X3 (pair, r 80): 4-quadrant phase circles at Lambda 531, 536, 545 (12 rows) and the 0 deg
period points 539/542/548/551 (4 rows). Companions: X10-X12 (radius), X13 (270 deg period scan),
X14 (Lambda 524 phases), X16 (count scan + equal-width corr row, IGUM).
Controls: NOT re-run — ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb).
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

ROWS = ([(lam, ph) for lam in (531.0, 536.0, 545.0) for ph in (0.0, 90.0, 180.0, 270.0)] +
        [(lam, 0.0) for lam in (539.0, 542.0, 548.0, 551.0)])
N_HALF = 15                                    # 31 sites
R_NM, Y_NM, N_OXIDE = 50.0, 0.0, 1.444         # y 0 = one cylinder on the axis per site

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam, phase_deg):
    dx = round(lam * phase_deg / 360.0, 1)
    return [round(k * lam + dx, 1) for k in range(-N_HALF, N_HALF + 1)]

X_LISTS = [comb_x(lam, ph) for lam, ph in ROWS]

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(X_LISTS),
    scatterer_x_list_nm = X_LISTS,
    scatterer_y_list_nm = [[Y_NM] * (2 * N_HALF + 1)] * len(X_LISTS),
    scatterer_height_nm = [350.0] * len(X_LISTS),
    scatterer_index     = [N_OXIDE] * len(X_LISTS),
    mode  = "zipped",
    label = "scat_x15_incore_axis_r50_circles",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for (lam, ph), xl in zip(ROWS, X_LISTS):
        print(f"Lambda {lam:5.1f}  phase {ph:5.1f}: x {xl[0]} .. {xl[-1]}")
