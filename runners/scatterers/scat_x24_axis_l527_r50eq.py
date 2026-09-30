"""Stage x24 — single SiO2 cylinder on the axis at Lambda 527: r 50 equal-width rescale (corr 456) (IGUM).

Study dir: runners/scatterers/   |   Created 2026-09-16 (overnight)   |   Job(s): IGUM 91008
Purpose (user): complete the Lambda 527 program — the r 50 phase circle (270 deg = X21 row T 0.9225, REUSED)
and the equal-width (corrugation) rescale at Lambda 527 for r 30 and r 50.
Corr choice (DERIVED, knob 1/w = a + b*corr, effective b = 0.0001416 /nm — confirmed by X17 r 60 landing 15.52):
  r 30: width 16.30 (X21) -> d(1/w) = 1/15.53 - 1/16.30 = 0.00304 -> +21 nm -> corr 421
  r 50: width 17.72 (X21) -> d(1/w) = 1/15.53 - 1/17.72 = 0.00796 -> +56 nm -> corr 456
PRE-REGISTERED (EXPECTED, from the Lambda 524 rescales X17/X20: Q_i -6% @ r 30/40, -9% @ r 60):
  r 30 @ 421: T 0.85-0.875, Q_L ~1450-1600, Q_i -3..-8%; r 50 @ 456: T 0.82-0.86, Q_L ~1700-1900, Q_i -5..-12%; width 15.2-15.9 um.
Split: scat_x23 (Athena, 4 rows = one wave) + scat_x24 (IGUM, 1 row).
Controls: NOT re-run — ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb).
Physics line (section 4): TM h350, pitch 516.83, W800, N=80/side; box y=16, 20 nm / 1501 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

ROWS = [(50.0, 270.0, 456.0)]                                                                   # (r nm, phase deg, corr nm)
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

X_LISTS = [comb_x(ph) for _, ph, _ in ROWS]

SPEC = SweepSpec(
    corrugation_depth_nm = [c for _, _, c in ROWS],
    scatterer_radius_nm  = [r for r, _, _ in ROWS],
    scatterer_x_list_nm  = X_LISTS,
    scatterer_y_list_nm  = [[Y_NM] * (2 * N_HALF + 1)] * len(ROWS),
    scatterer_height_nm  = [350.0] * len(ROWS),
    scatterer_index      = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x24_axis_l527_r50eq",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for cfg, (r, ph, c) in zip(SPEC.expand(BASE), ROWS):
        print(f"built: r {r:.0f}  phase {ph:.0f}  corr {cfg.geometry.corrugation_depth_m*1e9:.0f} nm")
