"""Stage C4 — single-scatterer response row on the WIDENED (W1050) cavity.

Study dir: runners/scatterers/   |   Created 2026-07-15   |   Job(s): TBD
Purpose: the W800 response-matrix program closed with the discovery that plain
cavity widening to 1050 nm supersedes the pillar route (+0.0354 vs +0.0377) and
that W800-optimized pillar positions HURT the widened cavity. This stage acquires
the W1050 device's OWN response row so the joint solve can answer whether any
scatterer arrangement still helps it (residual resonant loss 7.7%, mostly
arm-distributed per the k-space diagnostic — expect a modest ceiling).

Geometry: r=80 mirrored pillar pairs at y=825 nm — edge at 745, i.e. the same
220 nm clearance above the W1050 cavity edge (525) that the validated W800 row
(y=700 over edge 400) had. 15 sites, x = -945..+945 step 135 (every historical
winner lay within |x|<=810; extend outward later only if the edge site wins).

Dispatch (ONE command — deploy chains prelim -> main via --dependency=afterok;
serialize: queue must be empty of other --option3 arrays):
    PRELIM_TIME=00:30:00 ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_c4_w1050row --max-concurrent=3
Then: python runners/scatterers/solve_response_matrix.py solve with
--dir-c results_from_athena/scat_c4_w1050row (16 columns, W1050 basis).
Output -> results/scat_c4_w1050row/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CAVITY_W_M = 1050e-9
Y_ROW_NM   = 825.0
N_SITES    = 15
_half = (N_SITES - 1) // 2
SITES_X_NM = [round((i - _half) * 135.0, 1) for i in range(N_SITES)]

# Own sidecar: the W1050 device resonates at ~1558.79 (measured, job 121830) —
# the W800 program sidecar (1558.61) must NOT be reused for the lambda lock.
LOCKED_LAMBDA_FILE = "/work/results/scat_c4_w1050_lambda_res.json"

# Prelim: ports-only resonance find on the W1050 device, identical numerics.
PRELIM_BASE = _common.build_ports_base()
PRELIM_BASE.grating.cavity_width_m = CAVITY_W_M
PRELIM_SPEC = SweepSpec(
    scatterer_radius_nm = [0.0],
    label = "scat_c4_prelim",
)

BASE = _common.build_ff_base()
BASE.grating.cavity_width_m = CAVITY_W_M

_rows_x = [[0.0]] + [[x] for x in SITES_X_NM]
_ys     = [Y_ROW_NM] * (N_SITES + 1)
_rs     = [0.0] + [_common.RADIUS_NM] * N_SITES

_keys = {(r, round(xs[0], 1)) for r, xs in zip(_rs, _rows_x)}
assert len(_keys) == len(_rs), "stage-C4 rows must be unique in (r, x)"
assert Y_ROW_NM - _common.RADIUS_NM > CAVITY_W_M * 1e9 / 2.0, \
    "row edge must clear the widened cavity (no fused/buried pillars in acquisition)"

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_c4_w1050row",
)

if __name__ == "__main__":
    print(PRELIM_SPEC.describe())
    print(SPEC.describe())
    print(f"row y={Y_ROW_NM} over W{CAVITY_W_M*1e9:.0f} cavity; sites: {SITES_X_NM}")
