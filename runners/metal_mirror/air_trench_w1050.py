"""Air trench on the width-optimal cavity (plain W1050) — does the TIR lever stack?

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: stage L (job 124379) measured the air trench WORKING on W800 corr-400
(d-opt 1.8 um: dT +0.0159, loss -13.6%, needle 16x suppressed). The user's
question: does it also work on the width-OPTIMAL cavity, plain W1050 (the
+0.0354 device)? Evidence cuts both ways: the 2026-07-07 stack result (job
118893) put the trench ON a 1050-cavity device (stack: W1050 + gap-pair +
see-saw) and won -22% loss — but the C4 lesson is that overlay levers do NOT
transfer to width-optimized devices without direct measurement (pillar pair
+0.0227 on W800 flips to -0.0114 on W1050). W1050's leak is 0.47x W800's and
arm-distributed. This is the direct 4-task answer.

Rows (zipped): plain-W1050 control | W1050 + air trench (n=1.0, rect,
L 84 um x w 800 nm x h 2 um, mirrored +/-y, x=0) at d = 1500/1800/2100 nm
(brackets the W800/stack optimum; the weaker arm-distributed leak may shift it).
Registered predictions: P1 transfer -> dT ~ +0.007..+0.015 scaled by the 0.47x
leak (floor 0.0018); P2 non-transfer (C4-style) -> |dT| <= floor or negative;
P3 lambda drag small at d >= 1.5 (W800 measured -2.2/-0.7/-0.2 nm).

Numerics: identical to air_trench_dscan (box y = 16 um, 1501 pts / 20 nm
window, opt mesh, ff base). Lambda lock: the C4 sidecar (W1050 resonates
1558.79, measured job 121830/121848 — the W800 stage-H sidecar must NOT be
used); the 20 nm window covers any residual box-16 offset and the resonance
finder picks the true peak per-run. Physics line (CLAUDE.md section 4): TM
h350, pitch 516.83, corr 400, cavity 1050, N = 80/side; target resonance
1558.8 nm, window 1548.8-1568.8 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.air_trench_w1050 --max-concurrent=3
Output -> results/air_trench_w1050/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CAVITY_W_NM      = 1050.0
TRENCH_INDEX     = 1.0
TRENCH_LEN_UM    = 84.0
TRENCH_W_NM      = 800.0
TRENCH_HEIGHT_NM = 2000.0
D_SCAN_NM        = [1500.0, 1800.0, 2100.0]

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

# W1050 resonance sidecar (job 121848 prelim, 1558.79) — never the W800 one.
LOCKED_LAMBDA_FILE = "/work/results/scat_c4_w1050_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_HEIGHT_NM * 1e-9

# rows: (x_span_um, y_span_nm, d_nm) — x_span 0 = in-study control
ROWS = [(0.0, 0.0, 1500.0)]
ROWS += [(TRENCH_LEN_UM, TRENCH_W_NM, d) for d in D_SCAN_NM]

_PML_CLEAR_NM = 1200.0
for _, w, d in ROWS:
    assert d + 0.5 * w + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, \
        f"trench too close to the y PML: d {d}"
    assert d - 0.5 * w >= _common.TOOTH_EDGE_NM + (CAVITY_W_NM - 800.0) / 2.0, \
        f"trench overlaps the W1050 teeth (tips at 625 nm): d {d}"
assert len({d for _, _, d in ROWS[1:]}) == len(ROWS) - 1, \
    "d-scan rows must have unique distances (file-tag uniqueness)"

SPEC = SweepSpec(
    cavity_width_nm     = [CAVITY_W_NM] * len(ROWS),
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [r[0] for r in ROWS],
    scatterer_y_span_nm = [r[1] for r in ROWS],
    scatterer_y_nm      = [r[2] for r in ROWS],
    scatterer_index     = [TRENCH_INDEX] * len(ROWS),
    mode  = "zipped",
    label = "air_trench_w1050",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"plain W1050 ctrl + air trench (n={TRENCH_INDEX}) L={TRENCH_LEN_UM} um x "
          f"w={TRENCH_W_NM} nm x h={TRENCH_HEIGHT_NM:.0f} nm at d {D_SCAN_NM} nm")
