"""Stage B — noise floor + linearity (Born) gates + the (r, y) SIZE/DISTANCE gate.

Study dir: runners/scatterers/   |   Created 2026-07-12, (r,y) grid 2026-07-13   |   Job(s): TBD
Purpose: decide, from data, (1) whether single-scatterer responses beat the numerics
floor, (2) the minimum spacing at which two-scatterer superposition holds, and
(3) THE BIGGEST RADIUS AT THE CLOSEST y THAT STAYS LINEAR — the (r, y) the whole
program then uses. Pairs sit near the cavity center (strongest drive = worst case
for linearity), so a combo that passes here passes everywhere.

Rows (all lambda-locked to the stage-A sidecar, identical numerics to stage C):
  idx 0        r=0 control            — cross-job repeat of stage A: numerics floor.
  idx 1-3      half-mesh-cell jittered singles at (X_REF+25 / y+25 / both), default
               (RADIUS_NM, Y0_NM) — mesh-snap floor + phase-ramp smoothness.
  spacing set  at (RADIUS_NM, Y0_NM): single at X_REF, singles at X_REF+s and pairs
               [X_REF, X_REF+s] for each s in SEPARATIONS_NM  -> min spacing.
  (r, y) grid  for r in R_GATE_LIST_NM x y in Y_GATE_LIST_NM: the cavity-straddling
               pair GATE_PAIR_X_NM (one scatterer left, one right of the cavity) +
               each single alone -> per-combo superposition error + |response|.

Gate analysis:  python runners/scatterers/solve_response_matrix.py gates
PASS = |response| >> floor AND pair error < 20%; the verdict table recommends the
largest passing radius at its closest passing y -> paste into _common.py RADIUS_NM /
Y0_NM. FAIL everywhere = noise-limited or nonlinear: STOP (valid negative).

Dispatch (AFTER stage A completed; queue empty of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_b_gates
Output -> results/scat_b_gates/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

R0 = _common.RADIUS_NM
Y0 = _common.Y0_NM
X0 = _common.X_REF_NM
J  = _common.JITTER_NM
PAIR = sorted(_common.GATE_PAIR_X_NM)

# Task ORDER = mid-run information order (SLURM starts array tasks roughly by
# index): control + noise rows first (the floor every analysis needs), then the
# (r, y) grid with y ascending and r DESCENDING inside each y — so partial results
# already grade the biggest scatterers at the closest row — spacing series last.
_rows = []                                    # (r_nm, x_list_nm, y_nm)
_rows.append((0.0, [0.0], Y0))                                    # control
_rows.append((R0, [X0], Y0))                                      # reference single
_rows += [(R0, [X0 + J], Y0), (R0, [X0], Y0 + J), (R0, [X0 + J], Y0 + J)]   # jitter
for y in _common.Y_GATE_LIST_NM:                                  # (r, y) grid
    for r in sorted(_common.R_GATE_LIST_NM, reverse=True):
        if y < _common.y_min_nm(r):           # tooth-edge gap rule (125/150 skip 700)
            continue
        for xs in ([PAIR[0]], [PAIR[1]], PAIR):
            _rows.append((r, xs, y))
for s in _common.SEPARATIONS_NM:                                  # spacing set
    _rows.append((R0, [X0 + s], Y0))
    _rows.append((R0, [X0, X0 + s], Y0))

_keys = {(r, len(xs), round(xs[0], 1), round(xs[-1], 1), round(y, 1))
         for r, xs, y in _rows}
assert len(_keys) == len(_rows), "stage-B rows must differ in (r, N, first/last x, y)"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _, _ in _rows],
    scatterer_x_list_nm = [xs for _, xs, _ in _rows],
    scatterer_y_nm      = [y for _, _, y in _rows],
    mode  = "zipped",
    label = "scat_b_gates",
)

if __name__ == "__main__":
    print(SPEC.describe())
