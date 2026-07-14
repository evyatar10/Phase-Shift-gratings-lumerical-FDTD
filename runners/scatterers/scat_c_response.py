"""Stage C — response acquisition: one single-scatterer run per candidate position.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): TBD
Purpose: measure the COMPLEX far-field response of one mirrored scatterer pair at
each of the N_POSITIONS candidate x positions (_common.positions_nm(): odd count,
symmetric about the cavity, same RADIUS_NM/Y0_NM for all). idx 0 is the r=0 control
run whose complex far field IS the b vector of the inverse problem — same job, same
locked window, exactly the same recorded far-field wavelength as every column.

All rows lambda-locked to the stage-A sidecar; identical numerics throughout.
response_j(u) = E_ff_j(u) - E_ff_control(u)  (complex difference, formed offline).

Dispatch (AFTER stage B PASSED; queue empty of other --option3 arrays; smoke first
by setting N_POSITIONS = 5 in _common.py, dispatch, verify, restore):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_c_response
Then:  python runners/scatterers/solve_response_matrix.py solve
Output -> results/scat_c_response/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

_positions = _common.positions_nm()

_rows_x = [[0.0]] + [[x] for x in _positions]          # idx 0 = r=0 control (the b vector)
_rs     = [0.0] + [_common.RADIUS_NM] * len(_positions)
_ys     = [_common.Y0_NM] * (len(_positions) + 1)

_keys = {(r, round(xs[0], 1)) for r, xs in zip(_rs, _rows_x)}
assert len(_keys) == len(_rs), "stage-C rows must be unique in (r, x)"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_c_response",
)

if __name__ == "__main__":
    print(SPEC.describe())
