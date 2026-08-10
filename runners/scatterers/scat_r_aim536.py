"""Stage R — aim-corrected phase circle: Lambda=536 (true needle), 4 phases.

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD (Athena)
Purpose: stage P's overlap diagnosis gave eta = 0.31 — the anti-needle beam was
aimed at the box-16 needle IMAGE (ux 0.96); the instrument-corrected needle sits
at 0.98, wanting Lambda = lam/(n_eff + 0.98*n_clad) ~ 536 nm. Because the
interference phase rotates with aim, a single "best dx" row could alias — so run
the full 4-point phase circle at Lambda=536, r=110 (stage-P instrument): the
circle's T swing reads the interference amplitude directly. PASS = swing grows
from stage-P's +/-0.0098 toward ~+/-0.02 (eta ~ 0.7) => best row T ~ 0.892 >
ctrl; FAIL = swing unchanged => aim was not the limiter, circle-comb closes.

Controls: NOT re-run — Athena identical-numerics ctrl MEASURED twice, T = 0.8851
(stage H job 123563; trench_h350 125276_0). All dT vs 0.8851.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm (20 nm/1501).

Dispatch (queue must be EMPTY of other Athena --option3 arrays — section 6;
IGUM job 51285 is a DIFFERENT cluster tree, no conflict):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_r_aim536 --max-concurrent=3
Output -> results/scat_r_aim536/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
N_HALF    = 15                # 31 posts — stage-P geometry exactly
LAM_NM    = 536.0             # true-needle aim (ux 0.98, n_eff_emp 1.4936)
R_NM      = 110.0
HEIGHT_NM = 350.0
DX_LIST   = [0.0, 134.0, 268.0, 402.0]   # 0/90/180/270 deg of Lambda=536

BOX_Y_UM      = 16.0          # stage-H/P numerics exactly
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

def comb_x(dx_nm):
    return [round(k * LAM_NM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(DX_LIST),
    scatterer_x_list_nm = [comb_x(dx) for dx in DX_LIST],
    scatterer_y_list_nm = [[D_NM] * (2 * N_HALF + 1) for _ in DX_LIST],
    scatterer_height_nm = [HEIGHT_NM] * len(DX_LIST),
    mode  = "zipped",
    label = "scat_r_aim536",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for dx in DX_LIST:
        print(f"  dx={dx} nm ({dx/LAM_NM*360:.0f} deg), x=[{comb_x(dx)[0]}..{comb_x(dx)[-1]}]")
    print("dT vs MEASURED Athena ctrl 0.8851 (job 123563)")
