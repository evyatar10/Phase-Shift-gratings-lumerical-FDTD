"""Stage X10 — in-core r 50 comb at Lambda 524 / 270 deg with ONE hole per period ON THE AXIS (y 0) instead of the +/-250 pair (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): Athena 151320
Purpose (user): one test run — same comb as stage X6 (job 150429: pair at y +/-250, T 0.9244,
width 18.51 um) but a single hole per period on the symmetry axis (y = 0). The builder draws
ONE object for an on-axis site (its own mirror image), so the TM y-symmetry BC stays valid.
EXPECTED (not measured): same mechanism, dose ~ the pair (axis field max vs two half-intensity
sites) -> T ~0.92-0.93, width ~18.5-19.5 um, lambda shift ~-1 nm. The layout .fsp is the
deliverable the user asked for (download from results/<study>/layouts/, no results inside).
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851; compare with the X6 row.

Row: Lambda 524, dx 393 (270 deg), 31 holes, r 50, n 1.444, h 350, y 0 (single, on axis).
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x10_incore_r50_axis
Output -> results/scat_x10_incore_r50_axis/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15      # 270 deg, 31 holes — the X4 best row exactly
R_NM, Y_NM, N_OXIDE = 50.0, 0.0, 1.444

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
    label = "scat_x10_incore_r50_axis",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"x range {min(X_LIST)} .. {max(X_LIST)}  phase {360*DX_NM/LAM_NM:.1f} deg")
