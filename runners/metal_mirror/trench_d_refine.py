"""Full-z trench d-refine on W800 N=80 — does the optimum move from d=1.8 um?

Study dir: runners/metal_mirror/   |   Created 2026-07-21   |   Job(s): TBD
Purpose: the only measured d-scan (job 124379) used the h=2 um trench; the
final devices use FULL-Z trenches (through the z-PML), which intercept more
per unit standoff — the optimum may sit at the same d or slightly farther.
User: "small search on the 80, short total time." 5 tasks: own control +
full-z trench at d = 1500/1800/2100/2400 nm (brackets the h2 optimum 1.8).
h2 anchor curve at box16 (dT): 1.5 +0.0070 / 1.8 +0.0159 / 2.1 +0.0097 /
2.4 +0.0059. Verdict rule: any d beating 1800 by > floor 0.0018 => optimum
moved, port the winner to N=150; ties => 1.8 stands, no further scan.

Numerics: FAST frame — ports-only, box y = 8 um, 1501 pts / 20 nm window
centered on the W800 resonance 1558.61 (own in-study control at identical
numerics; box-16 numbers not comparable in absolute T). Trench: air rect,
L 84 um (N=80 arm), w 800 nm, h 12 um (full-z), mirrored +/-y. TM h350,
pitch 516.83, corr 400, N = 80/side.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_d_refine --max-concurrent=3
Output -> results/trench_d_refine/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_LEN_UM = 84.0
TRENCH_W_NM   = 800.0
TRENCH_H_NM   = 12000.0                       # full-z (z domain ~8.8 um)
D_SCAN_NM     = [1800.0, 2100.0]   # user: <45 min total => one %3 wave;
                                   # 1800 = reference, 2100 = outward candidate
                                   # (inward falls fast + drag only grows)

BOX_Y_UM      = 8.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_H_NM * 1e-9

# rows: (x_span_um, y_span_nm, d_nm) — x_span 0 = in-study control
ROWS = [(0.0, 0.0, 1500.0)]
ROWS += [(TRENCH_LEN_UM, TRENCH_W_NM, d) for d in D_SCAN_NM]

_PML_CLEAR_NM = 1200.0
for _, w, d in ROWS:
    assert d + 0.5 * w + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, f"y-PML: d {d}"
    assert d - 0.5 * w >= 500.0, f"tooth clearance: d {d}"
assert len({d for _, _, d in ROWS[1:]}) == len(ROWS) - 1, "unique d per row"

SPEC = SweepSpec(
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [r[0] for r in ROWS],
    scatterer_y_span_nm = [r[1] for r in ROWS],
    scatterer_y_nm      = [r[2] for r in ROWS],
    scatterer_index     = [1.0] * len(ROWS),
    mode  = "zipped",
    label = "trench_d_refine",
)

if __name__ == "__main__":
    print(SPEC.describe())
