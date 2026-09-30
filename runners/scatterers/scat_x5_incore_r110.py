"""Stage X5 — in-core oxide comb at its best point (Lambda 524, 270 deg) with r 110 (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-15   |   Job(s): TBD (Athena)
Purpose (user): the in-core r 80 comb peaks at Lambda 524 / 270 deg with T 0.9280
(stage X4, job 149982; ctrl 0.8851) at the cost of a wider mode (23.0 um vs 15.5)
and a 2.3 nm resonance shift. ONE run with the hole radius raised 80 -> 110 (the
SiN outside comb's radius; no in-core radius was ever scanned) to see whether T,
width, or both move. Hole at y 250 with r 110 spans y 140-360 nm; the narrow
segment half-width is 300 nm, so 60 nm of the hole sits in the cladding (already
30 nm at r 80) — oxide in oxide, harmless.
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851; compare
with the r 80 row result_..._scR80_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.

Row: Lambda 524, dx 393 (270 deg), 31 holes, r 110, n 1.444, h 350, y +/-250.
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x5_incore_r110
Output -> results/scat_x5_incore_r110/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15      # 270 deg, 31 holes — the X4 best row exactly
R_NM, Y_NM, N_OXIDE = 110.0, 250.0, 1.444

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

X_LIST = [round(k * LAM_NM + DX_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM],
    scatterer_x_list_nm = [X_LIST],
    scatterer_y_list_nm = [[Y_NM] * len(X_LIST)],
    scatterer_height_nm = [350.0],
    scatterer_index     = [N_OXIDE],
    mode  = "zipped",
    label = "scat_x5_incore_r110",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"x range {min(X_LIST)} .. {max(X_LIST)}  phase {360*DX_NM/LAM_NM:.1f} deg")
