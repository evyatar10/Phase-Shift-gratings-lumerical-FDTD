"""Stage X17 — on-axis single-hole combs r 60 and r 80 (Lambda 524 / 270 deg) with the corrugation raised to the
control's 15.5 um mode width: equal-width comparison of T and Q vs the plain device (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): Athena 151476
Purpose (user): "given the same mode width, how do peak T and Q compare to the original device" for the r 60
and r 80 axis combs; then external q3db-engine estimates only — NO further runs.
Corr choice (DERIVED): q3db TM width knob 1/w = a + b*corr with b = 0.000133986 /nm (q3db_calibration.csv),
corrected by the stage-X8 equal-width run (pair r 50, corr 477 landed 15.39 vs 15.53 target -> effective
b = 0.0001416 /nm). Axis widths MEASURED (X11 job 151333 / X12 job 90593): r 60 = 18.11 um, r 80 = 20.10 um.
  r 60: d(1/w) = 1/15.53 - 1/18.11 = 0.00917 -> +65 nm -> corr 465
  r 80: d(1/w) = 1/15.53 - 1/20.10 = 0.01464 -> +103 nm -> corr 503
PRE-REGISTERED (EXPECTED, from X8 pair r 50 @ corr 477: T 0.792, Q_L 2102, Q_i -15% vs ctrl at 15.39 um):
  r 60 @ 465: T 0.80-0.85, Q_L ~1800-2000, width 15.2-15.9 um; r 80 @ 503: T 0.72-0.80, Q_L ~2300-2700.
  Verdict metric = Q_i = Q_L/(1 - sqrt(T)) at matched width vs ctrl Q_i 22.4k (T 0.8851, Q_L 1327, 15.53 um).
Controls: NOT re-run — ctrl MEASURED (scat_h_retrocomb). Resonance expected ~1556-1557 nm, inside the window.
Physics line (section 4): TM h350, pitch 516.83, W800, N=80/side; box y=16, 20 nm / 1501 — stage X exactly.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 --spec=runners.scatterers.scat_x17_incore_axis_eqwidth
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

ROWS = [(60.0, 465.0), (80.0, 503.0)]          # (hole radius nm, corrugation nm)
LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15       # 270 deg, 31 sites — the X10-X12 row exactly
Y_NM, N_OXIDE = 0.0, 1.444                     # y 0 = one cylinder on the axis per site

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

X_LIST = [round(k * LAM_NM + DX_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

SPEC = SweepSpec(
    corrugation_depth_nm = [c for _, c in ROWS],
    scatterer_radius_nm  = [r for r, _ in ROWS],
    scatterer_x_list_nm  = [X_LIST] * len(ROWS),
    scatterer_y_list_nm  = [[Y_NM] * len(X_LIST)] * len(ROWS),
    scatterer_height_nm  = [350.0] * len(ROWS),
    scatterer_index      = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x17_incore_axis_eqwidth",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for cfg, (r, c) in zip(SPEC.expand(BASE), ROWS):
        print(f"built: r {r:.0f}  corr {cfg.geometry.corrugation_depth_m*1e9:.0f} nm")
