"""Two and three displaced combs — the second-comb test (physics rethink 2026-09-12).

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146419 (2 tasks, 2026-09-12)
Purpose: job 146364 (scat_offcentre) measured that a comb displaced +/-12 um from the cavity
loses its phase-independent cost (Lambda-536 circle pedestal +0.178 -> -0.009 in units of the
radiative rate) and reaches T 0.8959 (+12 um, 90 deg) / 0.8967 (-12 um, 90 deg) vs the
centred best 0.8928 (ctrl 0.8851).  The k-space model refitted on those rows
(python_tools/comb_kspace_model.py; scratch pattern_search_dx over centre, site shift and
post count) says two displaced combs, one on each side at its own phase, are nearly
ADDITIVE (their beams share the direction but carry decorrelated fine-k phase, unlike the
stacked rows of scat_t which were the same beam twice), while a third comb adds ~+0.001
because the carrier at |x| > 20 um is too weak.  Every earlier second-comb attempt shared
the first comb's centre and therefore only added amplitude past the optimum.

REGISTERED (final refit on 16 rows, rms 0.0034; results_from_athena/comb_physics_rethink/
data/pattern_search_dx.json; both variants agree):
  P4 the pair (row 0) gives T >= 0.905 (dT >= +0.020 vs ctrl 0.8851; model +0.0285 'top' /
     +0.0254 'lorentz'; the sum of the two measured singles is +0.0225).  REFUTED if
     T <= 0.8985 (best single + floor): then two beams into one direction compete after all.
  P5 the triple (row 1 = the model's best triple, +0.0319 / +0.0307) adds no more than
     2x floor (+0.0036) over the pair (model +0.0010 / +0.0008): the third comb sits where
     the carrier is 0.45x and buys nothing.  Both circles measured optimum phase 90 deg
     (sinusoid fits: pedestal -0.009 / -0.002, swing 0.091 / 0.110, T_opt 0.8956 / 0.8970).
Numerics = scat_offcentre exactly (box y 16 / z 8.8 um, 1501 pts / 20 nm, opt mesh, far-field
monitors at 2.0 wls); dT vs the stored ctrl 0.8851 (job 123563; re-measured with the moved
monitors as scat_offcentre row 8).  Physics line (CLAUDE.md section 4): TM h350, pitch 516.83,
corr 400, W800, N = 80/side; target resonance 1558.6 nm.

Dispatch (queue must be EMPTY of other Athena --option3 arrays — section 6):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_offcentre2 --max-concurrent=3
Output -> results/scat_offcentre2/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
LAM_NM    = 536.0
R_NM      = 110.0
HEIGHT_NM = 350.0

# one comb = (n_posts, centre x_c um, site shift dx nm); a row = the combs it holds
COMB_A = (31,  12.0, 134.0)     # measured winner, job 146364 row 1: T 0.8959
COMB_B = (31, -12.0, 134.0)     # measured winner, job 146364 row 5: T 0.8967
ROWS = [
    [COMB_A, COMB_B],                              # the pair
    [(31, 10.0, 0.0), (21, -6.0, 0.0), (31, -20.0, 179.0)],   # model's best triple ('top' refit, 5 rows); re-set from the final refit
]

BOX_Y_UM       = 16.0
N_WL_POINTS    = 1501
SCAN_WIDTH_NM  = 20.0
FF_DIST_WLS    = 2.0

def comb_x(n, xc_um, dx_nm):
    return [round(k * LAM_NM + dx_nm + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]

X_ROWS = [sum((comb_x(*c) for c in row), []) for row in ROWS]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.farfield.farfield_dist_wls = FF_DIST_WLS

assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
for xs in X_ROWS:
    assert max(abs(x) for x in xs) + R_NM <= _common.FF_X_SPAN_UM * 1000.0 / 2.0, "comb reaches past the far-field monitor"
    assert min(b - a for a, b in zip(sorted(xs)[:-1], sorted(xs)[1:])) > 2 * R_NM, "posts overlap"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = X_ROWS,
    scatterer_y_list_nm = [[D_NM] * len(xs) for xs in X_ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_offcentre2",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (row, xs) in enumerate(zip(ROWS, X_ROWS)):
        print(f"  row {i}: {len(xs)} posts; combs " + ", ".join(f"n{n}@{xc:+.0f}um/{dx:.0f}nm({dx/LAM_NM*360:.0f}deg)" for n, xc, dx in row)
              + f"; x=[{min(xs)}..{max(xs)}]")
    print("dT vs MEASURED Athena ctrl 0.8851 (job 123563); singles: +12/90deg 0.8959, -12/90deg 0.8967 (job 146364)")
