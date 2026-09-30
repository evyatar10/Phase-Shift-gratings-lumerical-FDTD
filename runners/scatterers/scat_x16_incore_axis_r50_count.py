"""Stage X16 — on-axis (y 0) single r 50 hole: hole-count scan at Lambda 531 + equal-width corrugation row (IGUM, serial).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): IGUM 90625
Purpose (user, "run exactly the same as the two-circle program but with one circle, r 50"): mirrors
stage X2/X3's count rows (pair r 80: 3 and 9 holes at Lambda 531) — here 3 and 9 holes at the four
quadrant phases (8 rows) — plus the stage-X8 equal-width test for the AXIS r 50 comb: corr raised
400 -> 452 (q3db TM knob line 1/w = 0.0113512 + 0.000133986*corr: axis r 50 width 17.42 -> 15.53 um
needs +52 nm) at Lambda 524 / 270 deg / 31 holes (1 row). Pre-registered (EXPECTED): the X8 pair
result (T 0.792, Q_i -15% at matched width) predicts T ~0.83-0.86 here at width 15.3-15.8 um.
Dispatched to IGUM with --max-concurrent=1 (same-node ansyscl startup race killed IGUM 90599).
Controls: NOT re-run — ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb); cross-cluster repro proven.
Physics line (section 4): TM h350, pitch 516.83, corr 400 (452 last row), W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# (lambda, phase, n_holes, corr)
ROWS = ([(531.0, ph, n, 400.0) for n in (3, 9) for ph in (0.0, 90.0, 180.0, 270.0)] +
        [(524.0, 270.0, 31, 452.0)])
R_NM, Y_NM, N_OXIDE = 50.0, 0.0, 1.444         # y 0 = one cylinder on the axis per site

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam, phase_deg, n):
    dx = round(lam * phase_deg / 360.0, 1); h = (n - 1) // 2
    return [round(k * lam + dx, 1) for k in range(-h, h + 1)]

X_LISTS = [comb_x(lam, ph, n) for lam, ph, n, _ in ROWS]

SPEC = SweepSpec(
    corrugation_depth_nm = [c for _, _, _, c in ROWS],
    scatterer_radius_nm  = [R_NM] * len(ROWS),
    scatterer_x_list_nm  = X_LISTS,
    scatterer_y_list_nm  = [[Y_NM] * len(xl) for xl in X_LISTS],
    scatterer_height_nm  = [350.0] * len(ROWS),
    scatterer_index      = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x16_incore_axis_r50_count",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for (lam, ph, n, c), xl in zip(ROWS, X_LISTS):
        print(f"Lambda {lam:5.1f}  phase {ph:5.1f}  n {n:2d}  corr {c:.0f}: x {xl[0]} .. {xl[-1]}")
