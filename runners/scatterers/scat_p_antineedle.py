"""Stage P — anti-needle comb: aim the carrier-out-coupled beam AT the needle.

Study dir: runners/scatterers/   |   Created 2026-08-09   |   Job(s): TBD
Purpose: the stage-O full-depth comb (Lambda 551, d 1.8) radiated a NEW beam at
ux -0.925 = first-order out-coupling of the guided carrier (calibrated design
study python_tools/antineedle_comb_design.py, docs/antineedle_comb_design.png).
Retuning Lambda moves that beam ONTO the needle (ux -0.96 imaged / 0.98 true)
where it can cancel it (Friedrich-Wintgen-style two-channel interference).
Rows: (1) Lambda scan 539-551 nm at dx=0 brackets the aim uncertainty;
(2) dx = Lambda/4, /2, 3/4 at Lambda 545 maps the interference phase circle —
model predicts needle power x0.22 (best) .. x3.3 (worst dx); (3) r=80 partner
= amplitude/linearity check. Comb: 31 posts (span ~16.4 um, width-matched to
the needle's 0.05-ux FWHM), core height h350 (single-litho), d = 1.8 um.

Controls: NOT re-run. The identical-numerics control is MEASURED at T = 0.8851
twice (stage H job 123563; trench_h350 job 125276 task 0) — all dT vs 0.8851.
Nothing on the CLAUDE.md section-2 numerics list changes (same box/window/
points/mesh, dispatched on Athena like the stored baseline).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm (20 nm/1501).

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_p_antineedle --max-concurrent=3
Output -> results/scat_p_antineedle/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0            # trench-optimum standoff (strong coupling, measured)
N_HALF    = 15                # 31 posts -> span ~16.4 um, width-matched beam
HEIGHT_NM = 350.0             # core height = single-litho (matched amplitude regime)

# rows: (Lambda_nm, dx_nm, r_nm)
LAM_SCAN = [539.0, 542.0, 545.0, 548.0, 551.0]
ROWS = ([(lam, 0.0, 110.0) for lam in LAM_SCAN] +
        [(545.0, 136.0, 110.0), (545.0, 273.0, 110.0), (545.0, 409.0, 110.0)] +
        [(545.0, 0.0, 80.0)])

BOX_Y_UM      = 16.0          # stage-H numerics exactly
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

def comb_x(lam_nm, dx_nm):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

for lam, dx, r in ROWS:
    assert D_NM + r + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
    assert D_NM - r >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
tags = [(tuple(comb_x(lam, dx)), r) for lam, dx, r in ROWS]
assert len(set(tags)) == len(tags), "rows must differ in x-list or radius (file tags)"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, _, r in ROWS],
    scatterer_x_list_nm = [comb_x(lam, dx) for lam, dx, _ in ROWS],
    scatterer_y_list_nm = [[D_NM] * (2 * N_HALF + 1) for _ in ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_p_antineedle",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (lam, dx, r) in enumerate(ROWS):
        print(f"row {i}: Lambda {lam} nm, dx {dx} nm, r {r} nm, "
              f"31 posts x=[{comb_x(lam, dx)[0]}..{comb_x(lam, dx)[-1]}]")
    print("dT vs MEASURED ctrl 0.8851 (job 123563)")
