"""Stage C2 — second-row response acquisition (2D-grid extension).

Study dir: runners/scatterers/   |   Created 2026-07-14   |   Job(s): TBD
Purpose: measure the complex far-field response of one mirrored r=RADIUS_NM pillar
pair at each row-2 candidate position (x, +/-ROW2_Y_NM), so the inverse solve can
choose COMBINATIONS across TWO rows (row 1 = stage C at Y0_NM is reused as-is).
Also measures the three gate rows the 2-row model needs:

  idx 0                 r=0 control (fresh b vector, same job as the new columns)
  idx 1..len(sites)     row-2 singles at _common.row2_positions_nm(), scalar y=ROW2_Y_NM
                        (33 aligned on the 135 nm step +/-2.16 um + 8 half-step
                        staggered near the cavity — "half-phase" test)
  last 3                gate rows: (a) cross-row STACK  x=[135,135],  y=[Y0, ROW2]
                                   (b) cross-row STAGGER x=[135,202.5], y=[Y0, ROW2]
                                   (c) in-row-2 pair     x=[0,270]  at  y=ROW2
                        -> solve2 checks superposition r_ab ~ r_a + r_b against the
                        5% gate BEFORE trusting any cross-row prediction.

All rows lambda-locked to the stage-A sidecar; identical numerics to stages A/B/C/E.
Row-2 strength is ~0.43x of row 1 (MEASURED, stage-B (r,y) grid) — well above the
~1e-6 relative noise floor.

Dispatch (queue empty of other --option3 arrays; ~45 tasks, ~5-6 h at %3):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.scatterers.scat_c2_row2 --max-concurrent=3
Then:  python runners/scatterers/solve_response_matrix.py solve2
Output -> results/scat_c2_row2/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

_sites = _common.row2_positions_nm()
_Y0, _Y2 = _common.Y0_NM, _common.ROW2_Y_NM

# control + row-2 singles + in-row-2 pair all use SCALAR y (clean _Y900 tags, stored
# scatterer_y_m correct); the per-site y list is passed ONLY for the two mixed rows.
_rows_x  = [[0.0]] + [[x] for x in _sites] + [[135.0, 135.0], [135.0, 202.5], [0.0, 270.0]]
_rows_yl = [None] * (1 + len(_sites))      + [[_Y0, _Y2],     [_Y0, _Y2],     None]
_ys      = [_Y2] * (1 + len(_sites) + 3)   # scalar y (geometry for None rows; metadata else)
_rs      = [0.0] + [_common.RADIUS_NM] * (len(_sites) + 3)

# File-tag collision guard: the tag encodes (r, N sites, first/last x, first/last y)
# — every row must be unique in that tuple or two tasks would share .h5/.mat names (§6).
_keys = set()
for r, xs, yl, ysc in zip(_rs, _rows_x, _rows_yl, _ys):
    ys = yl if yl else [ysc] * len(xs)
    _keys.add((r, len(xs), round(xs[0], 1), round(xs[-1], 1), round(ys[0], 1), round(ys[-1], 1)))
assert len(_keys) == len(_rs), "stage-C2 rows must be unique in (r, N, x0, x-1, y0, y-1)"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_list_nm = _rows_yl,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_c2_row2",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"row-2 sites: {len(_sites)}  total tasks: {len(_rs)}")
