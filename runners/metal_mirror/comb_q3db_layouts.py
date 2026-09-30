"""Comb layouts on the q3db device (corr 325, N = 165): one long comb vs the displaced pair.

Study dir: runners/metal_mirror/   |   Created 2026-09-12   |   Job(s): Athena 146639 (2 tasks, 2026-09-12)
Purpose: on the uniform corr-400 N=80 device, one centred 61-post comb (T 0.9064) and two displaced
31-post combs (0.9070) tie, and both beat the old 31-post comb (0.8966) because a comb's cost is set
by where its ENDS fall (sign period pi/(beta - k_c) = 12.2 um; ends must sit in the good half-period,
i.e. beyond ~10 um).  The q3db family's stored comb (comb_q3db.py, job 130458) is a centred 57-post
comb at r 80 / d 1.9 (+0.0455 vs ctrl 0.4906).  Two anchor rows at r 110 / d 1.8 (the corr-400
optimum), same numerics as comb_q3db EXACTLY:
  0  one centred 61-post comb, Lambda 531, dx 398 nm (270 deg), ends at +-16 um;
  1  the pair: 31 posts at +12 um / dx 88.5 nm (60 deg) + 31 posts at -12 um / dx 177 nm (120 deg)
     — the measured corr-400 winner geometry.
MODEL (k-space model transferred to corr 325: kappa rescaled, needle share calibrated on the stored
57-post row, EXPECTED-grade; scratch predict_c325b): row 0 +0.110, row 1 +0.107 (kappa 0.031;
+0.104..+0.115 over kappa 0.028-0.035) vs ctrl 0.4906 -> a TIE, as measured on corr 400.  The model
runs optimistic by a growing amount with post count; magnitudes are not the point, the ORDER is.
  REGISTERED: (i) both rows beat the stored 57-post r-80 comb (0.5361) by >= 2x floor: the r-110 /
  end-rule transfer holds; (ii) |T_row1 - T_row0| <= 0.0036: one comb = two combs on corr 325 too
  (the main question); if row 1 exceeds row 0 by > 0.0036 the pair has a real advantage on the
  longer envelope and gets the q3db lock; if row 0 exceeds row 1 the single comb gets it.
  Floor: 0.0018 (corr-400 jitter); at T ~ 0.5 the mirror balance is ~3x more sensitive, so treat
  0.005 as the working floor until a jitter twin is run.
Numerics = comb_q3db.py (job 130458/130548) EXACTLY: ports base, box y 8.0 / z 8.8 um, 20 nm window
centred 1559.5 nm, 4001 pts, opt mesh, far-field OFF, z-symmetry on.  Control = results_from_athena/
comb_q3db/results/result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat, T 0.4906 / Q 13930 (NOT re-run).
Next step after the verdict: predict-q3db extend mode anchored on the winning row -> N for -3 dB ->
one confirmation run.

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.metal_mirror.comb_q3db_layouts --max-concurrent=3
Output -> results/comb_q3db_layouts/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM   = 325.0
N_SIDE    = 165
R_NM      = 110.0
D_NM      = 1800.0
H_NM      = 350.0
LAM_NM    = 531.0

# one comb = (n_posts, centre um, site shift nm)
ROWS = [
    [(61, 0.0, 398.0)],                           # 0: one long comb, 270 deg
    [(31, 12.0, 88.5), (31, -12.0, 177.0)],       # 1: the pair, 60 / 120 deg
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

X_ROWS = [sum((comb_x(*c) for c in row), []) for row in ROWS]

assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to the y PML"
assert D_NM - R_NM >= _common.TOOTH_EDGE_NM + _common.GAP_MIN_NM, "comb too close to the teeth"
for xs in X_ROWS:
    assert max(abs(x) for x in xs) + R_NM + 2000.0 <= N_SIDE * 516.83, "comb reaches the grating end"
    assert min(b - a for a, b in zip(sorted(xs)[:-1], sorted(xs)[1:])) > 2 * R_NM, "posts overlap"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(ROWS),
    n_periods_each_side  = [N_SIDE] * len(ROWS),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(ROWS),
    scatterer_radius_nm  = [R_NM] * len(ROWS),
    scatterer_x_list_nm  = X_ROWS,
    scatterer_y_list_nm  = [[D_NM] * len(xs) for xs in X_ROWS],
    scatterer_height_nm  = [H_NM] * len(ROWS),
    mode  = "zipped",
    label = "comb_q3db_layouts",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (row, xs) in enumerate(zip(ROWS, X_ROWS)):
        print(f"  row {i}: {len(xs)} posts; " + ", ".join(f"n{n}@{xc:+.0f}um/{dx:.0f}nm({dx/LAM_NM*360:.0f}deg)" for n, xc, dx in row) + f"; x=[{min(xs)}..{max(xs)}]")
    print("ctrl N165 corr325: T 0.4906 (job 130458); stored 57-post r80 comb 0.5361; model: row0 +0.110, row1 +0.107 (tie)")
