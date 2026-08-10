"""Stage S — night refine wave: radius bracket + period bracket + d-degeneracy.

Study dir: runners/scatterers/   |   Created 2026-08-10 (night session)   |   Job(s): TBD
Purpose: refine the stage-R winner (Lambda=536, r=110, dx=402 = 270 deg, d=1.8,
T 0.8928 = +0.0077). Theory (calibrated: net = 0.0260x - 0.0187x^2, x = ampl/
ampl_r110): (a) r* = 110*sqrt(0.695) ~ 92 nm, ceiling +0.0090 — BIGGER IS WORSE
(registered); rows r=85/92/100 bracket it. (b) Lambda parabola: rows 532/540 at
their own 270 deg (A_c ~ 0 measured twice => single-phase rows valid). (c) d is
amplitude-degenerate with r: d=2.0 um at r=110 gives x = e^(-1.95*0.2) = 0.68
~ r92's x=0.70 => row 5 must match rows' r~92 net within floor (degeneracy
check; also ~13 deg phase rotation predicted — registered caveat).

Controls: NOT re-run — stage-R rows measured yesterday at identical numerics
vs stored ctrl 0.8851 (job 123563); winner row = in-study anchor.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm (20 nm/1501).

Dispatch (queue EMPTY of other Athena --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_s_refine --max-concurrent=3
Output -> results/scat_s_refine/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

N_HALF    = 15
HEIGHT_NM = 350.0

# rows: (Lambda_nm, dx_nm, r_nm, d_nm)
ROWS = [(536.0, 402.0,  85.0, 1800.0),   # r bracket (theory optimum ~92)
        (536.0, 402.0,  92.0, 1800.0),
        (536.0, 402.0, 100.0, 1800.0),
        (532.0, 399.0, 110.0, 1800.0),   # Lambda bracket at own 270 deg
        (540.0, 405.0, 110.0, 1800.0),
        (536.0, 402.0, 110.0, 2000.0),   # d-degeneracy check (x ~ 0.68)
        (536.0, 372.0, 110.0, 1800.0),   # phase fine: 250 deg (is 270 the peak?)
        (536.0, 432.0, 110.0, 1800.0),   # phase fine: 290 deg
        # ── edge wave (appended after wave-1 verdict: 532 beat 536 by 2x floor;
        # carrier-out-coupling cutoff at 530.7 nm — probe the edge) ──────────
        (530.0, 398.0, 110.0, 1800.0),   # just BELOW cutoff: predicted dark
        (531.0, 398.0, 110.0, 1800.0),   # just above cutoff
        (534.0, 401.0, 110.0, 1800.0),   # between 532 and 536
        (532.0, 399.0,  92.0, 1800.0)]   # r-optimum at the new best Lambda

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

def comb_x(lam_nm, dx_nm):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

for lam, dx, r, d in ROWS:
    assert d + r + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
    assert d - r >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
tags = [(tuple(comb_x(lam, dx)), r, d) for lam, dx, r, d in ROWS]
assert len(set(tags)) == len(tags), "rows must differ in x-list, radius or d"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, _, r, _ in ROWS],
    scatterer_x_list_nm = [comb_x(lam, dx) for lam, dx, _, _ in ROWS],
    scatterer_y_list_nm = [[d] * (2 * N_HALF + 1) for _, _, _, d in ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_s_refine",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, dx, r, d in ROWS:
        print(f"  Lam {lam} dx {dx} r {r} d {d}")
    print("anchors: winner T 0.8928 (130091_3), ctrl 0.8851 (123563)")
