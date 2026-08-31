"""Extend the aim (period) scan to the left, on BOTH phase branches.

Study dir: runners/scatterers/   |   Created 2026-08-27   |   Job(s): IGUM TBD
Purpose (user, for the presentation figure): the measured T-vs-period curves
start at their own edges -- the 270 deg branch starts AT its peak (530 nm) and
the 0 deg branch has no point below 536 nm. Four rows fill that in:

  0   Lambda 530, dx 0     (0 deg)   -- pairs with the existing 270 deg point at 530
  1   Lambda 534, dx 0     (0 deg)   -- pairs with the existing 270 deg point at 534
  2   Lambda 524, dx 0     (0 deg)   -- well below the pack, brackets the 0 deg branch
  3   Lambda 524, dx 393   (270 deg) -- one point BEFORE the 530 peak, so the
                                        cancelling branch shows a turnover

REGISTERED (so the figure cannot be read as a fit to noise): the 0 deg branch is
expected to keep falling toward smaller Lambda (it is monotone 551 -> 536 already,
0.8730 -> 0.8664, all below the control 0.8851); the 270 deg branch is expected to
DROP at 524 relative to 530 (0.8967) -- if it instead keeps rising, the optimum is
not at 530-531 and the design period must be revisited.

Controls: NOT re-run (CLAUDE.md section 6). The identical-numerics control is
MEASURED at T = 0.8851 and every delta is quoted against it. Existing points
reused, never re-measured: 270 deg at Lambda 530/531/532/534/536/540/545 and
0 deg at 536/539/542/545/551.

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side;
resonance ~1558.6 nm, window 1548.5-1568.5 nm (20 nm / 1501), box y = 16 um,
optimization mesh -- the stage-P/S numerics EXACTLY, so the new points are
bit-comparable with the stored series they extend.

Dispatch (IGUM; license seats shared with Athena -- check both):
    ARRAY_TIME=02:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_aim_extend --max-concurrent=4
Output -> results/scat_aim_extend/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
R_NM      = 110.0
N_HALF    = 15                # 31 posts — stage-P/S geometry exactly
HEIGHT_NM = 350.0

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

# (Lambda_nm, dx_nm) — dx = 0 is the reference branch, dx = 0.75*Lambda the cancelling one
ROWS = [(530.0,   0.0),
        (534.0,   0.0),
        (524.0,   0.0),
        (524.0, 393.0)]       # 0.75 * 524 = 393 nm = 270 deg

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam_nm, dx_nm):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

for lam, dx in ROWS:
    assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to the y PML"
    assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
    assert N_HALF * lam + dx + R_NM + 2000.0 <= 80 * 516.83, "comb end too close to the x edge"
tags = [tuple(comb_x(lam, dx)) for lam, dx in ROWS]
assert len(set(tags)) == len(tags), "rows must be tag-unique"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = [comb_x(lam, dx) for lam, dx in ROWS],
    scatterer_y_list_nm = [[D_NM] * (2 * N_HALF + 1) for _ in ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_aim_extend",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, dx in ROWS:
        print(f"  Lambda {lam:5.0f}  dx {dx:5.0f}  = {360.0 * dx / lam:5.1f} deg   "
              f"x {comb_x(lam, dx)[0]:.0f} .. {comb_x(lam, dx)[-1]:.0f} nm")
