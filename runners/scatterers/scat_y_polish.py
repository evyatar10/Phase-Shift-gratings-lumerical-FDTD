"""Stage Y — final polish wave: fine (d, phase, r, Lambda) around the plateau top.

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user): small last search for a summit on the flat optimum. Center =
the nominal-best branch d=1.5 (T 0.8974, needle x0.589; r82, dx383, Lam531).
Rows: phase +/-20 deg | r +/-10% | d=1.65 midpoint (rule-interpolated r/phase)
| Lambda 529 on the d1.5 branch. All h350 (program standard), 31 posts.
REGISTERED: plateau => ties; SUCCESS GATE = any row > 0.8992 (best + floor).

Controls: NOT re-run (stored ctrl 0.8851; anchors 130117/130135/130179).

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16, opt mesh.

Dispatch (queue EMPTY of other Athena --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_y_polish --max-concurrent=3
Output -> results/scat_y_polish/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# rows: (lam_nm, d_nm, r_nm, dx_nm, n_half)  — n_half 15 = the standard 31 posts
ROWS = [(531.0, 1500.0,  82.0, 413.0, 15),   # phase +20 deg
        (531.0, 1500.0,  82.0, 353.0, 15),   # phase -20 deg
        (531.0, 1500.0,  92.0, 383.0, 15),   # r up (near-cutoff r-shift hypothesis)
        (531.0, 1500.0,  74.0, 383.0, 15),   # r down
        (531.0, 1650.0,  95.0, 390.0, 15),   # d midpoint, rule-interpolated
        (529.0, 1500.0,  82.0, 382.0, 15),   # Lambda edge on the d1.5 branch
        # length axis (user q: "is full-length better?") at d=1.8 anchors,
        # amplitude-matched r ~ 110*sqrt(31/N). Model: LONGER = narrower beam
        # = worse needle overlap; registered: N=41 ~ tie, N=61 below.
        (531.0, 1800.0,  96.0, 398.0, 20),   # 41 posts, +/-10.6 um
        (531.0, 1800.0,  78.0, 398.0, 30),   # 61 posts, +/-16 um
        # appended (user, after N=41 won at +0.0148): fill the 41-61 bracket +
        # the full-device span. Full device: matched r would be ~50 (infeasible)
        # => keep r=80 and move to d=2.28 um (amplitude rule), transit phase
        # +24 nm (registered +-20 deg risk). REGISTERED: 47/53 ~ tie with 41;
        # full device WELL below (beam 5x narrower than the needle).
        (531.0, 1800.0,  89.0, 398.0, 23),   # 47 posts
        (531.0, 1800.0,  84.0, 398.0, 26),   # 53 posts
        (531.0, 2280.0,  80.0, 422.0, 75)]   # 151 posts = FULL DEVICE, d-compensated

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam_nm, dx_nm, n_half):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-n_half, n_half + 1)]

for lam, d, r, dx, nh in ROWS:
    assert d + r + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
    assert d - r >= _common.TOOTH_EDGE_NM
    assert nh * lam + dx + r + 2000.0 <= 46000.0, "comb end too close to x edge"
tags = [(lam, tuple(comb_x(lam, dx, nh)), r, d) for lam, d, r, dx, nh in ROWS]
assert len(set(tags)) == len(tags), "rows must be tag-unique"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, _, r, _, _ in ROWS],
    scatterer_x_list_nm = [comb_x(lam, dx, nh) for lam, _, _, dx, nh in ROWS],
    scatterer_y_list_nm = [[d] * (2 * nh + 1) for _, d, _, _, nh in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    mode  = "zipped",
    label = "scat_y_polish",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors: d1.5/r82/dx383 T 0.8974 | d1.8 winner 0.8966 | gate 0.8992")
