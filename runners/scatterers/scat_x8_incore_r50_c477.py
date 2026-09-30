"""Stage X8 — in-core r 50 hole comb with corrugation raised 400 -> 477 to restore the 15.5 um width (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-15   |   Job(s): Athena 150488
Question (user): does the in-core hole comb beat the plain device AT EQUAL MODE WIDTH,
or does it only ride the plain corr->width lever? One run: r 50 holes (X6 150429:
T 0.9244, width 18.51 um, lambda 1557.80) with the global corrugation raised so the
width returns to the ctrl's 15.5 um.
Corr choice (DERIVED, q3db TM knob line 1/w = 0.0113512 + 0.000133986*corr, w in um,
q3db_calibration.csv; reproduces ctrl 15.4 vs 15.53 measured): the holes lower 1/w by
1/15.53 - 1/18.51 = 0.01037 -> +77.4 nm of corr -> 477 nm.
PRE-REGISTERED expectation (EXPECTED): the hole series' dT/dwidth (+0.018/um) equals
the plain family's at N=80 (tm_width_lightline W900 C400->450: 0.025/um; W1000
C400->500: 0.018/um) -> NULL band T 0.87-0.90 at width 15-16.5 um; a GAIN needs
T >= 0.905 with width <= 16.5 um. Whatever width lands, judge vs the plain family's
T(width) at that width, not vs 15.5 exactly.
Resonance: corr up at fixed avg width shifts lambda ~-0.6 nm (stored ladder) + holes
-0.8 nm -> ~1557.2 nm, inside the 1548.5-1568.5 window; no pitch retune.
Controls: NOT re-run — plain corr-400 ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb).

Row: corr 477, Lambda 524, dx 393 (270 deg), 31 holes, r 50, n 1.444, h 350, y +/-250.
Physics line (section 4): TM h350, pitch 516.83, W800, N=80/side; box y=16, 20 nm / 1501.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x8_incore_r50_c477
Output -> results/scat_x8_incore_r50_c477/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM = 477.0
LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15      # 270 deg, 31 holes — the X4/X6 best row exactly
R_NM, Y_NM, N_OXIDE = 50.0, 250.0, 1.444

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

X_LIST = [round(k * LAM_NM + DX_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM],
    scatterer_radius_nm  = [R_NM],
    scatterer_x_list_nm  = [X_LIST],
    scatterer_y_list_nm  = [[Y_NM] * len(X_LIST)],
    scatterer_height_nm  = [350.0],
    scatterer_index      = [N_OXIDE],
    mode  = "zipped",
    label = "scat_x8_incore_r50_c477",
)

if __name__ == "__main__":
    print(SPEC.describe())
