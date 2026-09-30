"""Stage x13 — on-axis (y 0) single r 50 hole per period: period scan at 270 deg (Athena).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): Athena 151409
Purpose (user): repeat the +/-250 PAIR scans (stage X2-X4: phase circle + 270 deg period scan) with ONE
hole on the axis, r 50. The on-axis radius series (X10-X12) sits on the pair's T-vs-width line at a
lower dose (axis r 50 ~ pair r 42), so these rows test whether the phase/period behaviour is also the
same. Companion: scat_x13 (period scan, Athena) / scat_x14 (phases, Athena; both queue behind the 4-running QOS cap — user chose Athena for all after the
same-node ansyscl startup race killed IGUM 90599 on 2026-09-16).
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb);
the Lambda 524 / 270 deg r 50 axis row is REUSED from stage X10 (job 151320).
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAMS_NM   = [510.0, 515.0, 520.0, 527.0, 530.0, 534.0, 540.0, 545.0]   # 524 = stage X10 (job 151320), reused
PHASES_DEG = [270.0] * len(LAMS_NM)
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

X_LISTS = [comb_x(lam, ph) for lam, ph in zip(LAMS_NM, PHASES_DEG)]

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(X_LISTS),
    scatterer_x_list_nm = X_LISTS,
    scatterer_y_list_nm = [[Y_NM] * (2 * N_HALF + 1)] * len(X_LISTS),
    scatterer_height_nm = [350.0] * len(X_LISTS),
    scatterer_index     = [N_OXIDE] * len(X_LISTS),
    mode  = "zipped",
    label = "scat_x13_incore_axis_r50_lamscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, ph, xl in zip(LAMS_NM, PHASES_DEG, X_LISTS):
        print(f"Lambda {lam:5.1f}  phase {ph:5.1f}: x {xl[0]} .. {xl[-1]}")
