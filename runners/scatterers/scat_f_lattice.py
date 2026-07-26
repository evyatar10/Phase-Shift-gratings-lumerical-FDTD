"""Stage F — direct 2D-lattice sweep: row-row interference measured in FDTD (no model).

Study dir: runners/scatterers/   |   Created 2026-07-15   |   Job(s): TBD
Purpose: test the user's y-interference hypothesis with GROUND-TRUTH transmission:
stack 9-pair pillar combs (x period 540 nm = grid-matched AND retro-matched
lambda/(2*0.99*n_clad); span +/-2.16 um) above the device at swept row spacing dy,
with the first row NOT assumed (winner-based, absent, and lattice-only variants).
Bragg row spacing candidate: dy = lambda/(2 n_clad) ~ 540 nm; dy = 270 nm is the
destructive control point. Readout: peak T from each run's own T(lambda) — every
number is measured, immune to superposition-validity caveats (dimer coupling included
physically).

  idx 0   control (r=0)
  idx 1-5 winner [0,270]@700 + comb @ y=700+dy, dy in {270,405,540,675,810}
  idx 6   winner + comb@1240 staggered +270 in x
  idx 7   winner + combs @ 1240 AND 1780 (two-comb stack, Bragg spacing)
  idx 8   bare device + comb @ 1240              (no first row)
  idx 9   comb@700 + comb@1240                   (pure 540x540 lattice, no winner)
  idx 10  comb@700 alone                         (single-comb baseline)

Dispatch (queue empty; ~11 tasks, ~4 h at %3):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.scatterers.scat_f_lattice --max-concurrent=3
Readout: check-result / plot T(dy) from results/scat_f_lattice/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

COMB_X = [-2160.0, -1620.0, -1080.0, -540.0, 0.0, 540.0, 1080.0, 1620.0, 2160.0]
WIN_X, WIN_Y = [0.0, 270.0], 700.0
BRAGG_Y = 1240.0                       # 700 + lambda/(2 n_clad) ~ 700 + 540


def _sites(*groups):
    """groups = (x_list, y) -> merged site list sorted by (x, y)."""
    s = sorted((x, y) for xs, y in groups for x in xs)
    return [x for x, _ in s], [y for _, y in s]


_rows = [([0.0], [WIN_Y], 0.0)]                                     # 0 control
for dy in (270.0, 405.0, 540.0, 675.0, 810.0):                      # 1-5 T(dy) curve
    xs, ys = _sites((WIN_X, WIN_Y), (COMB_X, WIN_Y + dy))
    _rows.append((xs, ys, _common.RADIUS_NM))
xs, ys = _sites((WIN_X, WIN_Y), ([x + 270.0 for x in COMB_X], BRAGG_Y))
_rows.append((xs, ys, _common.RADIUS_NM))                           # 6 staggered
xs, ys = _sites((WIN_X, WIN_Y), (COMB_X, BRAGG_Y), (COMB_X, 1780.0))
_rows.append((xs, ys, _common.RADIUS_NM))                           # 7 two-comb stack
_rows.append((list(COMB_X), [BRAGG_Y] * len(COMB_X), _common.RADIUS_NM))  # 8 bare+comb
xs, ys = _sites((COMB_X, 700.0), (COMB_X, BRAGG_Y))
_rows.append((xs, ys, _common.RADIUS_NM))                           # 9 pure lattice
_rows.append((list(COMB_X), [700.0] * len(COMB_X), _common.RADIUS_NM))    # 10 comb@700

_rows_x  = [r[0] for r in _rows]
_rows_yl = [r[1] for r in _rows]
_rs      = [r[2] for r in _rows]

# File-tag collision guard (tag = r, N sites, first/last x, first/last y) — §6.
_keys = {(r, len(xs), round(xs[0], 1), round(xs[-1], 1), round(ys[0], 1), round(ys[-1], 1))
         for r, xs, ys in zip(_rs, _rows_x, _rows_yl)}
assert len(_keys) == len(_rs), "stage-F rows must be unique in (r, N, x0, x-1, y0, y-1)"
# Geometry sanity: pillars must stay inside the converged box (y half-span 3.4 um).
assert max(y for ys in _rows_yl for y in ys) + _common.RADIUS_NM < 3400.0 - 800.0, \
    "lattice row too close to the transverse PML"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_list_nm = _rows_yl,
    mode  = "zipped",
    label = "scat_f_lattice",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"total tasks: {len(_rs)}")
