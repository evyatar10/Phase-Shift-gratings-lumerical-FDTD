"""Air trench at exactly the CORE height (350 nm) — single-litho-layer test.

Study dir: runners/metal_mirror/   |   Created 2026-07-23   |   Job(s): TBD
Purpose: every winning trench so far was 2 um / 4 um / full-z tall — a second
litho layer. User question 2026-07-23: does a trench of exactly the core
height 350 nm (same litho layer as the teeth) retain any of the gain?
Two tasks: in-study control + W800 trench d = 1800 nm (the d-scan optimum)
at h = 350 nm, at the EXACT air_trench_dscan numerics (box y = 16 um,
1501 pts / 20 nm window, opt mesh, stage-H sidecar) — compare against job
124379 at identical numerics: control T 0.8851, h = 2000 trench dT +0.0159.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83 nm, corr 400 nm,
W800, N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_h350 --max-concurrent=3
Output -> results/trench_h350/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_INDEX     = 1.0            # air
TRENCH_LEN_UM    = 84.0           # full arm span (d-scan geometry)
TRENCH_W_NM      = 800.0          # d-scan geometry
TRENCH_HEIGHT_NM = 350.0          # THE knob: exactly the core layer
D_NM             = 1800.0         # d-scan optimum

BOX_Y_UM      = 16.0              # stage-H/J numerics
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

# Reuse the stage-H sidecar (job 123561) — identical box/window/device numerics.
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_HEIGHT_NM * 1e-9

# rows: (x_span_um, y_span_nm, d_nm) — x_span 0 = in-study control
ROWS = [(0.0, 0.0, D_NM), (TRENCH_LEN_UM, TRENCH_W_NM, D_NM)]

assert D_NM - 0.5 * TRENCH_W_NM >= _common.TOOTH_EDGE_NM, "trench overlaps the teeth"
assert D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, \
    "trench too close to the y PML"

SPEC = SweepSpec(
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [r[0] for r in ROWS],
    scatterer_y_span_nm = [r[1] for r in ROWS],
    scatterer_y_nm      = [r[2] for r in ROWS],
    scatterer_index     = [TRENCH_INDEX] * len(ROWS),
    mode  = "zipped",
    label = "trench_h350",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"air trench L={TRENCH_LEN_UM} um x w={TRENCH_W_NM} nm x h={TRENCH_HEIGHT_NM:.0f} nm "
          f"(core layer), d={D_NM} nm, vs 124379 control 0.8851 / h2000 +0.0159")
