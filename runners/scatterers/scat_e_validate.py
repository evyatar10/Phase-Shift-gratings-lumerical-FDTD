"""Stage E — FDTD validation of the chosen scatterer COMBINATIONS.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): TBD
Purpose: build each candidate combination (list of x positions, all RADIUS_NM/Y0_NM
mirrored pairs) in ONE simulation and measure the real figure of merit — far-field
power, T, R, loss at resonance — against an identical-numerics r=0 control.
Compare with stage D's predictions via `solve_response_matrix.py validate`.

>>> USER STEP: paste the position lists printed by
>>>     python runners/scatterers/solve_response_matrix.py solve
>>> into CANDIDATE_ARRAYS below (and MIN_SPACING_NM from the stage-B gates output).

Dispatch (AFTER stage D; queue empty of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_e_validate
Output -> results/scat_e_validate/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

# ── PASTE stage-D output here ──────────────────────────────────────────────────
# Each entry = one combination = list of scatterer x centers in nm (each drawn as a
# mirrored +/-Y0 pair). Stage-D solve of job 120976 (2026-07-14): binary combinations
# saturate at ~30-31% leak-power reduction (exhaustive 2/3-site Gram check agrees
# with greedy; LS 99.2% ceiling needs unphysical |w| up to 30x).
# ROUND 1 (job 121239, all COMPLETED): [135] 30.0%/dT+0.0203 | [0,270] 32.7%/+0.0227
#   | [0,270,5535] 32.9%/+0.0227 | comb [-3240,135,3510] 30.4%/+0.0201.
# ROUND 2 (job 121372, COMPLETED): off-grid interpolation test [0,229] — measured FF
#   reduction 33.4% (interpolation +0.7pp confirmed vs [0,270]'s 32.7%) but dT +0.0224
#   = tie with [0,270]'s +0.0227 at the jitter floor. Landscape top is FLAT in T;
#   final device stays [0,270]. Off-grid positions are legal: the builder takes
#   continuous coordinates; the 135-nm grid was only the MEASUREMENT set.
CANDIDATE_ARRAYS = [
    [0.0, 229.0],                 # interpolated partner optimum (round-2 test)
]
MIN_SPACING_NM = 225.0        # round-2: 229-nm spacing is below the 270 gate value;
                              # fine here because FDTD measures ground truth directly
                              # (superposition is only needed for PREDICTIONS)
# ───────────────────────────────────────────────────────────────────────────────

_half_span_nm = _common.SPAN_UM * 1000.0 / 2.0
for arr in CANDIDATE_ARRAYS:
    assert len(arr) >= 1 and list(arr) == sorted(arr), f"array must be sorted: {arr}"
    assert all(abs(x) <= _half_span_nm + 1.0 for x in arr), \
        f"position outside the measured grid span (+/-{_half_span_nm} nm): {arr}"
    gaps = [b - a for a, b in zip(arr, arr[1:])]
    assert all(g >= MIN_SPACING_NM - 1.0 for g in gaps), \
        f"spacing below MIN_SPACING_NM={MIN_SPACING_NM}: {arr}"

_rows_x = [[0.0]] + [list(map(float, arr)) for arr in CANDIDATE_ARRAYS]
_rs     = [0.0] + [_common.RADIUS_NM] * len(CANDIDATE_ARRAYS)
_ys     = [_common.Y0_NM] * (len(CANDIDATE_ARRAYS) + 1)

# File-tag collision guard (tag encodes r, N sites, first/last x): every row's
# key must be unique or two array tasks would share layout/.h5/.mat names (§6).
_keys = {(r, len(xs), round(xs[0], 1), round(xs[-1], 1))
         for r, xs in zip(_rs, _rows_x)}
assert len(_keys) == len(_rs), \
    "stage-E rows must differ in (r, N, first x, last x) — rename/trim a candidate"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_e_validate",
)

if __name__ == "__main__":
    print(SPEC.describe())
