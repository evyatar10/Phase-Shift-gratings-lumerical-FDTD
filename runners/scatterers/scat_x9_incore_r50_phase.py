"""Stage X9 — in-core r 50 hole comb at Lambda 524: the three missing phases 0/90/180 deg (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-15   |   Job(s): Athena 150504
Purpose (user): complete the 4-point phase circle at r 50 / corr 400 / Lambda 524 so a
first-harmonic fit (as in plot_incore_circle_fit.m) can be drawn for r 50. The 270 deg
point is REUSED from stage X6 (job 150429: T 0.9244, width 18.51 um) — not re-run.
Phase = 360*dx/Lambda: dx 0 / 131 / 262 nm. Everything else = stage X6 exactly.
Controls: NOT re-run — plain ctrl MEASURED T 0.8851 (scat_h_retrocomb).

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x9_incore_r50_phase
Output -> results/scat_x9_incore_r50_phase/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM_NM, N_HALF = 524.0, 15
PHASES_DEG = [0.0, 90.0, 180.0]                  # 270 reused from X6
R_NM, Y_NM, N_OXIDE = 50.0, 250.0, 1.444

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

X_LISTS = [comb_x(p) for p in PHASES_DEG]

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(PHASES_DEG),
    scatterer_x_list_nm = X_LISTS,
    scatterer_y_list_nm = [[Y_NM] * (2 * N_HALF + 1)] * len(PHASES_DEG),
    scatterer_height_nm = [350.0] * len(PHASES_DEG),
    scatterer_index     = [N_OXIDE] * len(PHASES_DEG),
    mode  = "zipped",
    label = "scat_x9_incore_r50_phase",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for p, xl in zip(PHASES_DEG, X_LISTS):
        print(f"phase {p:5.1f}: x {xl[0]} .. {xl[-1]}")
