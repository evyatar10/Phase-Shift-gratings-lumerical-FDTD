"""The -3 dB confirmation runs for the r-110 comb layouts on the q3db device (corr 325).

Study dir: runners/metal_mirror/   |   Created 2026-09-12   |   Job(s): Athena 146681 (2 tasks, 2026-09-12) — MEASURED: pair N172 T 0.5003 Q 18093 w 19.76; single N171 T 0.5060 Q 17557 w 19.79 (both in band)
Purpose: predict-q3db (extend mode, anchored on the MEASURED N=165 rows of comb_q3db_layouts.py, job
146639: pair T 0.5704 / Q_L 15014, single 61-post comb T 0.5659 / Q_L 14971) puts the -3 dB crossing at
  row 0  pair (31 @ +12 um / 88.5 nm + 31 @ -12 um / 177 nm, r 110, d 1.8, Lambda 531) at N* = 172:
         EXPECTED T 0.4977 (-3.03 dB), Q_L 18155, lambda 1559.03 nm, spectral fwhm 85.9 pm, width 19.77 um;
  row 1  one centred 61-post comb (dx 398 nm, r 110, d 1.8, Lambda 531) at N* = 171:
         EXPECTED T 0.5036 (-2.98 dB), Q_L 17620, lambda 1559.04 nm, fwhm 88.5 pm, width 19.80 um.
  REGISTERED pass bands (engine, design-grade): Q_L +-10 %, T +-0.03, width +-5 %; expected deviation
  at span 6-7 periods: Q_L +-3.2 %, T +-0.007.  Comparison: stored comb lock N169 Q 16203 (-3.04 dB),
  full-z trench lock N168 Q 18777.  These rows also give the second anchor that pins the Q_i shape
  (refit calibrate_q3db afterwards).
Numerics = comb_q3db.py / comb_q3db_layouts.py EXACTLY: ports base, box y 8.0 / z 8.8 um, 20 nm window
centred 1559.5 nm, 4001 pts, opt mesh, far-field OFF, z-symmetry on.  No control row (stored).

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.metal_mirror.comb_q3db_lock --max-concurrent=3
Output -> results/comb_q3db_lock/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM   = 325.0
R_NM      = 110.0
D_NM      = 1800.0
H_NM      = 350.0
LAM_NM    = 531.0

# row = (N per side, combs); comb = (n_posts, centre um, site shift nm)
ROWS = [
    (172, [(31, 12.0, 88.5), (31, -12.0, 177.0)]),   # 0: the pair at its N*
    (171, [(61, 0.0, 398.0)]),                        # 1: one long comb at its N*
]

BOX_Y_UM       = 8.0
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
assert BASE.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"

def comb_x(n, xc_um, dx_nm):
    return [round(k * LAM_NM + dx_nm + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]

X_ROWS = [sum((comb_x(*c) for c in combs), []) for _, combs in ROWS]

assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to the y PML"
assert D_NM - R_NM >= _common.TOOTH_EDGE_NM + _common.GAP_MIN_NM, "comb too close to the teeth"
for (n_side, _), xs in zip(ROWS, X_ROWS):
    assert max(abs(x) for x in xs) + R_NM + 2000.0 <= n_side * 516.83, "comb reaches the grating end"
    assert min(b - a for a, b in zip(sorted(xs)[:-1], sorted(xs)[1:])) > 2 * R_NM, "posts overlap"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(ROWS),
    n_periods_each_side  = [n for n, _ in ROWS],
    center_wavelength_nm = [SCAN_CENTER_NM] * len(ROWS),
    scatterer_radius_nm  = [R_NM] * len(ROWS),
    scatterer_x_list_nm  = X_ROWS,
    scatterer_y_list_nm  = [[D_NM] * len(xs) for xs in X_ROWS],
    scatterer_height_nm  = [H_NM] * len(ROWS),
    mode  = "zipped",
    label = "comb_q3db_lock",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, ((n_side, combs), xs) in enumerate(zip(ROWS, X_ROWS)):
        print(f"  row {i}: N {n_side}, {len(xs)} posts; " + ", ".join(f"n{n}@{xc:+.0f}um/{dx:.0f}nm" for n, xc, dx in combs))
    print("EXPECTED: row0 T 0.4977 Q 18155 w 19.77; row1 T 0.5036 Q 17620 w 19.80 (bands Q +-10%, T +-0.03, w +-5%)")
