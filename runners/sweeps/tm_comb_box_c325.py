"""Gate A0: does the corr-325 campaign box (y 6.8 / z 6.8) hold WITH the comb?

Study dir: runners/sweeps/   |   Created 2026-08-13   |   Job(s): TBD
Purpose: the 6.8/6.8 box was converged on the BARE device only (tm_span_conv_c325,
its recorded open flank #1): the comb sits off-axis in y and reshapes the
radiation lobe — exactly what a grazing y-PML handles worst. Before the lumopt2
campaign commits to the small box, measure the seed comb in both boxes.

Rows (zipped, both N=100 surrogate, seed comb Λ531/δx401(270°)/r80/d1.9/57 posts):
  0  campaign box   y 6.8, z 6.8 (mult 3.50)
  1  q3db box       y 8.0, z 8.8 (mult 5.42)
Verdict: |dT| < 0.002 AND Q_i = Q/(1-sqrt(T)) flat  =>  campaign runs at 6.8/6.8;
else 8.0/8.8 (the lumopt2_design CampaignSpec box_y_um/box_z_mult get updated).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_comb_box_c325
Output -> results/tm_comb_box_c325/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common
from runners.lumopt2_design.lumopt2_design import (
    COMB_LAM_NM, COMB_DX_NM, COMB_R_NM, COMB_D_NM, COMB_N_HALF, N_COMB)

CORR_NM        = 325.0
N_PERIODS      = 100                     # the settled surrogate (2*kappa*L = 3.65)
SCAN_CENTER_NM = 1559.5                  # q3db family numerics (tm_nladder_c325)
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001

# WAVE 2 (2026-08-13, after job 131496): wave 1 task 0 was dispatched with
# mult 3.50 = z 5.8 by mistake (the rejected span-study rung); its result is
# kept as data. This corrected single row is the actual campaign-box candidate.
_Y_UM   = [6.8]
_Z_MULT = [4.14]              # z = 0.35 + 4.14*1.5595 = 6.8 um
# (wave-1 rows, job 131496, kept for the record: y [6.8, 8.0], mult [3.50, 5.42])

X_COMB = [round(k * COMB_LAM_NM + COMB_DX_NM, 1)
          for k in range(-COMB_N_HALF, COMB_N_HALF + 1)]

# y-PML clearance in the SMALL box; comb inside the grating (comb_q3db asserts)
assert COMB_D_NM + COMB_R_NM + 1200.0 <= min(_Y_UM) * 1000.0 / 2.0
assert COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + COMB_R_NM + 2000.0 <= N_PERIODS * 516.83

BASE = _common.build_ports_base()
BASE.y_span_override_m = None            # per-row via y_span_um
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"

_N = len(_Y_UM)
SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * _N,
    n_periods_each_side  = [N_PERIODS] * _N,
    center_wavelength_nm = [SCAN_CENTER_NM] * _N,
    scatterer_radius_nm  = [COMB_R_NM] * _N,
    scatterer_x_list_nm  = [X_COMB] * _N,
    scatterer_y_list_nm  = [[COMB_D_NM] * N_COMB] * _N,
    scatterer_height_nm  = [350.0] * _N,
    y_span_um            = list(_Y_UM),
    span_mult            = list(_Z_MULT),
    mode  = "zipped",
    label = "tm_comb_box_c325",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (y, m) in enumerate(zip(_Y_UM, _Z_MULT)):
        z = 0.35 + m * SCAN_CENTER_NM * 1e-3
        print(f"  task {i}: y={y:.1f} um  z={z:.2f} um  + seed comb (57 posts)")
    print("verdict: |dT| < 0.002 AND Q_i flat -> campaign at 6.8/6.8, else 8.0/8.8")
