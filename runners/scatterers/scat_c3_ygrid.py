"""Stage C3 — full 2D response grid: single-scatterer rows at constant dy=135.

Study dir: runners/scatterers/   |   Created 2026-07-15   |   Job(s): TBD
Purpose: the 2D equivalent of stage C, per the user's design — measure the complex
far-field response of ONE mirrored r=RADIUS_NM pillar pair per simulation, on a grid
of rows y = 970, 1105, 1240, 1375, 1510 nm (constant dy = 135 nm, starting 270 nm
above the measured y=700 row) x 31 x-sites (135 nm step, +/-2.025 um). Rows run in
order NEAREST FIRST so partial results are usable mid-run. Combined offline with the
already-measured rows y=700 (97 sites) and y=900 (41 sites), the joint linear solve
then searches ALL rows simultaneously — no row's pattern is assumed.

Solve-time rule (from the measured gates, applied later, not here): combinations may
not place two pillar pairs closer than ~270 nm center-to-center (40-110 nm oxide gaps
are measured non-linear / physically overlapping below 160 nm). Acquisition itself is
singles-only, so ANY row spacing is measurable.

  idx 0        r=0 control (fresh b vector)
  idx 1-31     row y= 970, x = -2025 : 135 : +2025
  idx 32-62    row y=1105
  idx 63-93    row y=1240
  idx 94-124   row y=1375
  idx 125-155  row y=1510

Dispatch (QOS caps 100 submitted -> TWO chunks of the SAME study, serialized):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.scatterers.scat_c3_ygrid --max-concurrent=3 --array-tasks=0-99
    ... wait for drain, then:  ... --array-tasks=100-155
Then: python runners/scatterers/solve_response_matrix.py solve2 (extended for N rows)
Output -> results/scat_c3_ygrid/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

ROWS_Y_NM = [970.0, 1105.0, 1240.0, 1375.0, 1510.0]   # nearest first (mid-run use)
N_SITES   = 31                                         # per row, 135 nm step
_half = (N_SITES - 1) // 2
SITES_X_NM = [round((i - _half) * 135.0, 1) for i in range(N_SITES)]

_rows_x = [[0.0]] + [[x] for y in ROWS_Y_NM for x in SITES_X_NM]
_ys     = [ROWS_Y_NM[0]] + [y for y in ROWS_Y_NM for _ in SITES_X_NM]
_rs     = [0.0] + [_common.RADIUS_NM] * (len(ROWS_Y_NM) * N_SITES)

_keys = {(r, round(xs[0], 1), round(y, 1)) for r, xs, y in zip(_rs, _rows_x, _ys)}
assert len(_keys) == len(_rs), "stage-C3 rows must be unique in (r, x, y)"
assert max(_ys) + _common.RADIUS_NM < 3400.0 - 800.0, "row too close to the transverse PML"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_c3_ygrid",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"rows: {ROWS_Y_NM}  sites/row: {N_SITES}  total tasks: {len(_rs)}")
