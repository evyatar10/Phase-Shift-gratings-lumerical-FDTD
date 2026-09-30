"""Stage X7 — in-core oxide comb at Lambda 524 / 270 deg with r 40 and r 30 (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-15   |   Job(s): Athena 150458
Purpose (user): radius is a monotone width/lambda lever for the in-core hole comb
(r50 18.5 / r80 23.0 / r110 28.2 um; ctrl 15.5) while T stays ~0.92-0.93 for r 50-80
(X4 149982, X5 150391, X6 150429). Two smaller radii to see whether the width keeps
closing toward the control while T holds. Both are sub-cell at dx 50 nm (conformal
mesher averages them in, ~area-weighted) — behaviour/trend only, absolute T is a
candidate; user accepted this explicitly.
Controls: NOT re-run — identical-numerics ctrl MEASURED T 0.8851 (scat_h_retrocomb).

Rows: Lambda 524, dx 393 (270 deg), 31 holes, n 1.444, h 350, y +/-250, r 40 | r 30.
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x7_incore_r40_r30
Output -> results/scat_x7_incore_r40_r30/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15      # 270 deg, 31 holes — the X4 best row exactly
RADII_NM = [40.0, 30.0]
Y_NM, N_OXIDE = 250.0, 1.444

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

X_LIST = [round(k * LAM_NM + DX_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

SPEC = SweepSpec(
    scatterer_radius_nm = RADII_NM,
    scatterer_x_list_nm = [X_LIST] * len(RADII_NM),
    scatterer_y_list_nm = [[Y_NM] * len(X_LIST)] * len(RADII_NM),
    scatterer_height_nm = [350.0] * len(RADII_NM),
    scatterer_index     = [N_OXIDE] * len(RADII_NM),
    mode  = "zipped",
    label = "scat_x7_incore_r40_r30",
)

if __name__ == "__main__":
    print(SPEC.describe())
