"""The displaced comb PAIR: transfer to the widened cavity, and the model's remaining candidates on
the uniform device (2026-09-12, after jobs 146364 / 146419).

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146553 (4 tasks, 2026-09-12)
The measured pair (scat_offcentre2 row 0, job 146419): 31 posts at +12 um / dx 134 nm and 31 posts
at -12 um / dx 134 nm, Lambda 536, r 110, d 1.8 -> T 0.9040 (+0.0189 vs ctrl 0.8851).
Rows (all stage-H numerics = scat_t_confirm exactly: far-field base, box y 16 / z 8.8 um, 1501 pts /
20 nm, opt mesh, far-field monitors at the default 0.8 wls):
  0  the pair on the W1050 cavity.  Control = results_from_athena/air_trench_w1050/results/
     result_N80_TM_W1050_Ybox16p0_Zbox8p8_ff.mat, T 0.921850 (job 124400, NOT re-run); the centred
     single comb there gave 0.931045 (+0.0092, scat_t_confirm row 3, job 130154).
     REGISTERED: pair dT >= +0.013 (the single's +0.0092 scaled as on the uniform device,
     0.0189/0.0115); REFUTED if dT <= +0.0092 + floor (no gain over the single).
  1  the pair with ROD posts (rect 140 x 270 nm, elongated along y, equal area to r 110) on the
     uniform device.  A dielectric post's induced polarization follows the local field through
     its polarizability tensor; elongating the post along y raises the in-plane transverse
     component relative to the vertical one and tilts the comb's radiation pattern.  Never tried
     (the rect study used squares only).  REGISTERED: |dT - 0.0189| <= 0.0036 EXPECTED (shape did
     not matter for squares); a deviation beyond that in either direction is the finding.
  2+ the k-space model's remaining candidates (see ROWS; predictions in the row comments, from
     scratchpad pattern_search_wide -> results_from_athena/comb_physics_rethink/data/
     pattern_search_wide.json).  dT vs ctrl 0.8851 (job 123563).
Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, N = 80/side; W800 except row 0.

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_pair_transfer --max-concurrent=3
Output -> results/scat_pair_transfer/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
R_NM      = 110.0
HEIGHT_NM = 350.0
BOX_Y_UM       = 16.0
N_WL_POINTS    = 1501
SCAN_WIDTH_NM  = 20.0
ROD_X_UM, ROD_Y_NM = 0.140, 270.0        # rod cross-section: 140 x 270 nm = 37800 nm^2 (circle r110 = 38013)

PAIR = [(31, 12.0, 134.0, 536.0), (31, -12.0, 134.0, 536.0)]   # (n_posts, centre um, site shift nm, Lambda nm)

# row = dict(combs=[...], cavity=W nm or None, shape='cylinder'|'rect')
PAIR_531 = [(31, 12.0, 88.5, 531.0), (31, -12.0, 177.0, 531.0)]    # model's best pair at the cutoff period: +12 @60deg, -12 @120deg
PAIR_531_H = [(31, 12.0, 132.8, 531.0), (31, -12.0, 132.8, 531.0)]  # hedge: both at 90deg of 531 (the phase rule measured at 536)
ROWS = [
    dict(combs=PAIR, cavity=1050.0, shape="cylinder"),       # 0: pair on W1050
    dict(combs=PAIR, cavity=None,   shape="rect"),           # 1: pair with rods 140 x 270
    dict(combs=PAIR_531, cavity=None, shape="cylinder"),     # 2: pair at Lambda 531, model phases (model 'top' +0.0324 vs +0.0285 for row-0 geometry)
    dict(combs=PAIR_531_H, cavity=None, shape="cylinder"),   # 3: pair at Lambda 531, both 90deg (phase hedge)
]

def comb_x(n, xc_um, dx_nm, lam_nm):
    return [round(k * lam_nm + dx_nm + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]

X_ROWS = [sum((comb_x(*c) for c in row["combs"]), []) for row in ROWS]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
for xs in X_ROWS:
    assert max(abs(x) for x in xs) + R_NM <= _common.FF_X_SPAN_UM * 1000.0 / 2.0, "comb reaches past the far-field monitor"
    assert min(b - a for a, b in zip(sorted(xs)[:-1], sorted(xs)[1:])) > max(2 * R_NM, ROD_X_UM * 1e3), "posts overlap"

SPEC = SweepSpec(
    scatterer_radius_nm  = [R_NM] * len(ROWS),
    scatterer_x_list_nm  = X_ROWS,
    scatterer_y_list_nm  = [[D_NM] * len(xs) for xs in X_ROWS],
    scatterer_height_nm  = [HEIGHT_NM] * len(ROWS),
    scatterer_shape      = [row["shape"] for row in ROWS],
    scatterer_x_span_um  = [ROD_X_UM if row["shape"] == "rect" else None for row in ROWS],
    scatterer_y_span_nm  = [ROD_Y_NM if row["shape"] == "rect" else None for row in ROWS],
    cavity_width_nm      = [row["cavity"] for row in ROWS],
    mode  = "zipped",
    label = "scat_pair_transfer",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (row, xs) in enumerate(zip(ROWS, X_ROWS)):
        print(f"  row {i}: cavity {row['cavity'] or 'W800'}, {row['shape']}, {len(xs)} posts; combs "
              + ", ".join(f"n{n}@{xc:+.0f}um/{dx:.0f}nm/L{lam:.0f}" for n, xc, dx, lam in row["combs"]) + f"; x=[{min(xs)}..{max(xs)}]")
