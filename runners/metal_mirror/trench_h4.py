"""Taller air trench (h = 4 um) on W1050 — the one unscanned optimization axis.

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: the trench height (2 um) was inherited from the 2026-07-07 stack
study, never scanned. The stage-J 350 nm film showed z-interception is the
limiter for wall structures; a 4 um trench intercepts more of the radiation
cone's vertical spread. Single task: W1050 + trench d = 1800, h = 4000 nm —
compared against job 124400's control (T 0.9218) and h = 2000 trench
(T 0.9375) at otherwise IDENTICAL numerics (box 16, 1501 pts / 20 nm, opt
mesh, C4 sidecar). dT(h4 vs h2) > +0.0018 => height is live, scan further;
within floor => h = 2 um is already sufficient (z-extent ~1-1.5 um, stage I).

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_h4 --max-concurrent=3
Output -> results/trench_h4/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_HEIGHT_NM = 4000.0
D_NM             = 1800.0

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

LOCKED_LAMBDA_FILE = "/work/results/scat_c4_w1050_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_HEIGHT_NM * 1e-9

# z-clearance: trench half-height 2 um << z half-span ~4.4 um (mult 5.42)
SPEC = SweepSpec(
    cavity_width_nm     = [1050.0],
    scatterer_shape     = ["rect"],
    scatterer_x_span_um = [84.0],
    scatterer_y_span_nm = [800.0],
    scatterer_y_nm      = [D_NM],
    scatterer_index     = [1.0],
    mode  = "zipped",
    label = "trench_h4",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"W1050 + air trench d={D_NM} h={TRENCH_HEIGHT_NM} (vs 124400 h=2000)")
