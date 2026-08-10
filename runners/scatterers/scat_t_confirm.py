"""Stage T — night confirm wave: section-2 two-step + 2D lattice + apodized comb.

Study dir: runners/scatterers/   |   Created 2026-08-10 (night session)   |   Job(s): TBD
Rows (B* = current best comb: Lambda 536, dx 402 = 270 deg, r 100, d 1.8, h350;
wave-1 early read: r85/92/100 plateau T 0.8932-0.8936 vs r110 0.8928):
  0 phase-twin: B* with dx+Lambda (938) — same phase mod Lambda, every post at a
    DIFFERENT mesh registration => the comb's jitter floor (section-2 step 1).
  1 accurate-mesh control (no scatterer)   } section-2 step 2: the winner at
  2 accurate-mesh B*                       } dx~35 nm vs its own accurate ctrl.
  3 W1050 + B* comb — transfer test. Anchor: W1050 box16 ctrl T 0.9218 MEASURED
    (job 124400, stage L2, identical numerics — NOT re-run). Trench transferred
    (+0.0157); pillar-pair did not; comb sits over the ARMS => predict transfers.
  4 2-row lattice r110 (d = 1.8, 2.336 um)     } user-requested 2D array test.
  5 4-row lattice r110 (d = 1.8..3.408, step Lambda)  } REGISTERED: rows add
  6 4-row lattice r80 (amplitude-compensated)  } ~x1.4 amplitude with ~35 deg/row
    phase lag (drive x0.35/row) => r110 lattices OVERSHOOT (T < single-row);
    r80 4-row ~ single-row optimum (equivalence, not gain). A BEAT would mean
    the model misses physics.
  7 envelope-apodized comb: 31 posts, r_j = r0 e^{-kappa|x_j|/2}, kappa 0.0446,
    total Sum r^2 = 0.695 x (31 x 110^2) => r0 ~ 100, edge ~84 (mesh-feasible).
    REGISTERED: expected ~ +0.001 vs uniform = at/below floor (user-requested).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800 (row 3
W1050), N = 80/side; resonance 1558.6 (W1050: 1558.8), window 1548.5-1568.5.

Dispatch (after wave-1 drain; queue EMPTY of other Athena --option3 arrays):
    SBATCH_MEM=180G ARRAY_TIME=05:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_t_confirm --max-concurrent=3
Output -> results/scat_t_confirm/results/.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM, DX, R_BEST, D_NM = 531.0, 398.0, 110.0, 1800.0   # B* (wave-1+edge verdict:
# plateau 530-532 all T 0.896-0.897 sub-floor ties; 531 = plateau center; r110
# beats r92 at the plateau — r-optimum moves UP near the soft cutoff)
N_HALF    = 15
HEIGHT_NM = 350.0
KAPPA     = 0.0446    # 1/um, measured leak-envelope decay

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(dx_nm):
    return [round(k * LAM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

X31 = comb_x(DX)
ROWS_4 = [D_NM + i * LAM for i in range(4)]           # 1800, 2336, 2872, 3408

# apodized radii: r_j^2 ∝ e^{-kappa|x_j|}, Sum r^2 = 0.695 * 31 * 110^2
_env = [math.exp(-KAPPA * abs(x) / 1000.0) for x in X31]
_r0sq = 0.695 * 31 * 110.0 ** 2 / sum(_env)
R_APOD = [round(math.sqrt(_r0sq * e), 1) for e in _env]
assert min(R_APOD) >= 80.0, "apodized radii below the r=80 mesh floor"

def flat(xlist, ylists):
    """All (x, y) pairs for a multi-row lattice: x-list repeated per row."""
    xs, ys = [], []
    for y in ylists:
        xs += xlist
        ys += [y] * len(xlist)
    return xs, ys

X2, Y2 = flat(X31, ROWS_4[:2])
X4, Y4 = flat(X31, ROWS_4)

SPEC = SweepSpec(
    scatterer_radius_nm = [R_BEST, 0.0, R_BEST, R_BEST, 110.0, 110.0, 80.0, max(R_APOD)],
    scatterer_x_list_nm = [comb_x(DX + LAM), [0.0], X31, X31, X2, X4, X4, X31],
    scatterer_y_list_nm = [[D_NM] * 31, [D_NM], [D_NM] * 31, [D_NM] * 31,
                           Y2, Y4, Y4, [D_NM] * 31],
    scatterer_r_list_nm = [None, None, None, None, None, None, None, R_APOD],
    scatterer_height_nm = [HEIGHT_NM] * 8,
    simulation_mode     = ["optimization", "accurate", "accurate", "optimization",
                           "optimization", "optimization", "optimization", "optimization"],
    cavity_width_nm     = [None, None, None, 1050.0, None, None, None, None],
    mode  = "zipped",
    label = "scat_t_confirm",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"apod radii: {R_APOD[15]} center .. {R_APOD[0]} edge")
    print("anchors: B* r100 T 0.8936 | winner r110 0.8928 | ctrl 0.8851 | W1050 ctrl 0.9218")
