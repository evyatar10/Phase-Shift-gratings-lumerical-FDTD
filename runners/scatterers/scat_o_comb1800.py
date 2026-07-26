"""Stage O — the 551 nm retro comb moved INTO the near zone (d = 1.8 um).

Study dir: runners/scatterers/   |   Created 2026-07-24   |   Job(s): TBD
Purpose (user): stage H ran the needle-matched comb (Lambda = lambda/(2*
n_clad*ux) = 551 nm, ux = 0.980, theta 11.5 deg) at standoffs d >= 3 um and
measured NULL (transparent). The air trench WINS at d = 1.8 um — test the
comb there too: (1) planar h = 350 nm, (2) full-z pillars (12 um, through
both z-PMLs, stage-M trench convention). Deliberately BELOW stage H's
2.5 um near-field floor — registered risk: the comb phase-matches the
guided carrier into the cladding light cone (n_eff 1.523 - lambda/Lambda =
-1.31, inside n_clad 1.444) => drain signature = lambda drag + T loss.

Controls: NOT re-run (user: lean on existing). The identical-numerics
control is MEASURED at T = 0.8851 twice (stage H job 123561; trench_h350
job 125276 task 0, 2026-07-24) — all dT quoted vs 0.8851.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_o_comb1800 --max-concurrent=3
Output -> results/scat_o_comb1800/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAMBDA_X_NM = 551.0           # stage-H needle-matched period (re-derived, correct)
COMB_R_NM   = 110.0           # stage-H comb radius
N_HALF      = 75              # 151 sites, span +/-41.3 um (full arm)
D_NM        = 1800.0          # the trench optimum standoff — the point of the test
HEIGHTS_NM  = [350.0, 12000.0]   # planar | full-z through both z-PMLs

BOX_Y_UM      = 16.0          # stage-H numerics exactly
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

COMB_X_NM = [round(k * LAMBDA_X_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

assert D_NM + COMB_R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
assert D_NM - COMB_R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
assert len(set(HEIGHTS_NM)) == len(HEIGHTS_NM), \
    "heights must be unique (the _H file tag is the only difference between rows)"

SPEC = SweepSpec(
    scatterer_radius_nm = [COMB_R_NM] * len(HEIGHTS_NM),
    scatterer_x_list_nm = [COMB_X_NM] * len(HEIGHTS_NM),
    scatterer_y_list_nm = [[D_NM] * len(COMB_X_NM)] * len(HEIGHTS_NM),
    scatterer_height_nm = HEIGHTS_NM,
    mode  = "zipped",
    label = "scat_o_comb1800",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"comb: {len(COMB_X_NM)} sites, Lambda {LAMBDA_X_NM} nm, d {D_NM} nm, "
          f"heights {HEIGHTS_NM} nm; dT vs MEASURED ctrl 0.8851")
