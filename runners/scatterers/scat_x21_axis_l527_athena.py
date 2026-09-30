"""Stage x21 — single SiO2 cylinder on the axis at the r 60 OPTIMUM period Lambda 527: phase circle + radius series (Athena).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): Athena 151686
Purpose (user): the r 60 single-cylinder period scan (X18, job 151509) peaks at Lambda 527-530 (T 0.9270 at both;
pchip max 527), not at the pair's 524 where the phase circle and radius series were run. Redo both at Lambda 527:
phases 0/90/180 at r 60 (270 = X18 row, REUSED) and radii 30/40/50/80/110 at 270 deg (r 60 = X18 row, REUSED).
Split: scat_x21 (Athena, 6 rows) + scat_x22 (IGUM, 2 rows, --max-concurrent=1: ansyscl startup race).
Controls: NOT re-run — ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb).
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

ROWS = [(60.0, 0.0), (60.0, 90.0), (60.0, 180.0), (30.0, 270.0), (40.0, 270.0), (50.0, 270.0)]   # (r nm, phase deg)
LAM_NM, N_HALF = 527.0, 15                     # 31 sites
Y_NM, N_OXIDE = 0.0, 1.444                     # y 0 = one cylinder on the axis per site

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(phase_deg):
    dx = round(LAM_NM * phase_deg / 360.0, 1)
    return [round(k * LAM_NM + dx, 1) for k in range(-N_HALF, N_HALF + 1)]

X_LISTS = [comb_x(ph) for _, ph in ROWS]

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _ in ROWS],
    scatterer_x_list_nm = X_LISTS,
    scatterer_y_list_nm = [[Y_NM] * (2 * N_HALF + 1)] * len(ROWS),
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x21_axis_l527_athena",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for (r, ph), xl in zip(ROWS, X_LISTS):
        print(f"r {r:5.1f}  phase {ph:5.1f}: x {xl[0]} .. {xl[-1]}")
