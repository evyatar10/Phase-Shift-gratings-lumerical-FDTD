"""Stage X4 — comb period BELOW 530 nm at 270 deg, in-core oxide AND SiN outside (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-15   |   Job(s): TBD (Athena)
Purpose (user): the in-core oxide comb at 270 deg rises monotonically as Lambda
falls 545 -> 530 (stage X3: 0.689 -> 0.860) and the scan stops at 530. Extend
both families below 530 so the T-vs-period graph shows the in-core peak and the
turn-over beyond it. SiN outside peaks at ~530 (stored 524 @270: 0.8924 IGUM
scat_aim_extend; 530: 0.8967 stage S).
Reused, NOT re-run: in-core 270 deg at 530/531/532/534/536/540/545 (X2/X3);
SiN 270 deg at 524/530/531/532/534/536/540/545 (aim_extend / S / R / P).
Geometry as stored: in-core r 80 n 1.444 y +/-250; SiN r 110 (index 1.97 =
n_core, so the file tag stays the SiN pattern) y +/-1800; 31 posts; h 350.
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851.

Rows (Lambda, r, y, n): 270 deg => dx = 0.75 * Lambda
  in-core : 527 / 524 / 520 / 515 / 510
  SiN out : 527 / 520 / 515 / 510        (524 stored)

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena, a100-public):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x4_incore_below530 --max-concurrent=4
Output -> results/scat_x4_incore_below530/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

N_HALF = 15
BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam_nm):
    dx = round(0.75 * lam_nm, 1)                       # 270 deg
    return [round(k * lam_nm + dx, 1) for k in range(-N_HALF, N_HALF + 1)]

# rows: (Lambda_nm, r_nm, y_nm, index)
ROWS = ([(lam,  80.0,  250.0, 1.444) for lam in (527.0, 524.0, 520.0, 515.0, 510.0)] +
        [(lam, 110.0, 1800.0, 1.97)  for lam in (527.0, 520.0, 515.0, 510.0)])

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, r, _, _ in ROWS],
    scatterer_x_list_nm = [comb_x(lam) for lam, _, _, _ in ROWS],
    scatterer_y_list_nm = [[y] * (2 * N_HALF + 1) for _, _, y, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [n for _, _, _, n in ROWS],
    mode  = "zipped",
    label = "scat_x4_incore_below530",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, r, y, n in ROWS:
        print(f"Lambda {lam:5.1f}  r {r:5.1f}  y {y:6.1f}  n {n}  dx {0.75*lam:6.1f}")
