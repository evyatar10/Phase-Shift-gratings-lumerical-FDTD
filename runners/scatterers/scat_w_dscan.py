"""Stage W — d-degeneracy PROPER test: other distances at their OWN optima.

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user): all prior non-1.8 d rows froze phase/radius at the 1.8-um
settings. This run re-tunes both at each d: r = 110 * e^{gamma(d-1.8)/2}
(gamma 1.95/um, amplitude rule) and dx shifted by the beam transit phase
(k_y ~ 0.58/um at the near-horizon beam => -/+15 nm per -/+0.3 um).
Rows: d=1.5 (r 82, dx 383) | d=2.1 (r 147, dx 407); Lambda 531, h350, 31 posts.
REGISTERED: both ~ +0.010..+0.012 = tie with the 1.8-um optimum (+0.0115) =>
degeneracy proven at the optima, d closes. Any row > +0.0133 (floor above the
winner) = REAL d-physics beyond the amplitude model (candidate: comb-device
multiple scattering at small d) => refine.

Controls: NOT re-run (stored ctrl 0.8851 job 123563; winner anchors 130117/135).

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16, opt mesh.

Dispatch (queue EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_w_dscan --max-concurrent=3
Output -> results/scat_w_dscan/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM = 531.0
N_HALF = 15

# rows: (d_nm, r_nm, dx_nm) — r from the amplitude rule, dx transit-corrected
ROWS = [(1500.0, 82.0, 383.0), (2100.0, 147.0, 407.0)]

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(dx_nm):
    return [round(k * LAM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

for d, r, _ in ROWS:
    assert d + r + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
    assert d - r >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, r, _ in ROWS],
    scatterer_x_list_nm = [comb_x(dx) for _, _, dx in ROWS],
    scatterer_y_list_nm = [[d] * (2 * N_HALF + 1) for d, _, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    mode  = "zipped",
    label = "scat_w_dscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors: d1.8 optimum +0.0115 (T 0.8966) | ctrl 0.8851 | beat-gate 0.8984")
