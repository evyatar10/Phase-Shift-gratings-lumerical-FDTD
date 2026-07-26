"""Stage G — do scatterers still help on an APODIZED device?

Study dir: runners/scatterers/   |   Created 2026-07-16   |   Job(s): TBD
Purpose: the known pair [0, 270] @ y=700 (r=80, mirrored) gained dT +0.0227 on the
uniform W800 device, but ~88% of that was effective-cavity-widening. On apodized
devices cavity widening is MEASURED NEGATIVE (tm_pareto_stack_vs_apod: apod n=10 +
W1000/1050/1100 -> T 0.9674/0.9597/0.9480 vs plain 0.9767), so the two mechanisms
now predict OPPOSITE signs. One pair on each of the user-chosen apod devices
(linear, default depth, n=5 and n=10, W800 avg cavity, N=80/side) vs its own
identical-numerics control.

REGISTERED PREDICTION (2026-07-16, before dispatch): dT ~ -0.005..0. The
recycling share (~+0.003 on the uniform device) scales with the leak budget
(loss 0.042/0.023 vs 0.117) to ~+0.0006..0.001 — under the 0.0018 jitter floor —
while the width-like share turns negative. Any dT > +0.0018 falsifies this and
means real non-width physics survives apodization (-> re-aim the response matrix
at the apod device). Null/negative => scatterer overlays CLOSED for apodized
devices too.

Dispatch (queue empty of other --option3 arrays; no prelim — ports-only, T at own
resonance, the shared lambda sidecar is NOT used because apod shifts lambda_res):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.scatterers.scat_g_apod --max-concurrent=3
Output -> results/scat_g_apod/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

PAIR_X_NM = [0.0, 270.0]      # the measured pair optimum on the uniform device
PAIR_Y_NM = 700.0             # standoff row, like the original stage-C/E winner

# (n_apod_periods_each_side, radius_nm) — radius 0 = that device's control
ROWS = [
    (5,  0.0),
    (5,  _common.RADIUS_NM),
    (10, 0.0),
    (10, _common.RADIUS_NM),
]

_apn = [n for n, _ in ROWS]
_rs  = [r for _, r in ROWS]
_xl  = [list(PAIR_X_NM)] * len(ROWS)
_yl  = [[PAIR_Y_NM] * len(PAIR_X_NM)] * len(ROWS)

BASE = _common.build_ports_base()

SPEC = SweepSpec(
    apod_method              = ["linear"] * len(ROWS),
    n_apod_periods_each_side = _apn,
    scatterer_radius_nm      = _rs,
    scatterer_x_list_nm      = _xl,
    scatterer_y_list_nm      = _yl,
    mode  = "zipped",
    label = "scat_g_apod",
)

if __name__ == "__main__":
    print(SPEC.describe())
