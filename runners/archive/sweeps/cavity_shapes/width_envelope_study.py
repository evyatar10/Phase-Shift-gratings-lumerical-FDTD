"""
WIDTH-ENVELOPE ("graded island") study — the next level of the shape question.

Measured facts driving this design: added dielectric at the defect cuts TM loss
in proportion to AREA (cavity-only: -29% at +0.5% fwhm) while GLOBAL widening
costs mode width (-23% at +5%). The optimal device should therefore be a width
envelope W(x) peaked at the defect and decaying over the radiating region —
"shaped like the mode", which is exactly the user's matched-shape idea applied
where it can win. This study maps the extent/taper dimension empirically.

All island rows: cavity widened to 1050 nm AND the innermost N teeth per side
widened by +250 nm (both narrow and wide widths -> avg +250, corr unchanged, so
kappa per tooth ~unchanged, local n_eff up).

Rows (zipped, 7 tasks, optimization mesh, converged box, window 1558.5/60/3001
— islands red-shift the resonance by several nm as local n_eff rises):
  0  control (uniform, rect 800 cavity)
  1  cavity 1050 only                    (= ladder point; in-study reference)
  2  island N=1 (+250 on tooth 1)  + cavity 1050
  3  island N=2                    + cavity 1050
  4  island N=4                    + cavity 1050
  5  island N=8                    + cavity 1050
  6  TAPERED island N=8 (+250 linearly falling to +31) + cavity 1050

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER cavity_width_ladder in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.width_envelope_study
Output -> results/width_envelope_study/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 60.0
BASE.spectral.n_wl_points = 3001

W_N, W_W = 600.0, 1000.0              # anchored narrow/wide (nm)
ADD = 250.0                            # island width boost (nm)
CAV = 1050.0                           # island rows' cavity width (nm)


def step_island(n):
    return ([W_N + ADD] * n, [W_W + ADD] * n)


def taper_island(n):
    adds = [ADD * (n - k) / n for k in range(n)]          # innermost first: 250 -> 31
    return ([W_N + a for a in adds], [W_W + a for a in adds])


_narrows, _wides, _cw = [None], [None], [None]
_cw += [CAV]
_narrows += [None]; _wides += [None]
for n in (1, 2, 4, 8):
    nn, ww = step_island(n)
    _narrows.append(nn); _wides.append(ww); _cw.append(CAV)
nn, ww = taper_island(8)
_narrows.append(nn); _wides.append(ww); _cw.append(CAV)

_n = 7
assert len(_narrows) == len(_wides) == len(_cw) == _n
_keys = {(tuple(a) if a else None, c) for a, c in zip(_wides, _cw)}
assert len(_keys) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    width_narrow_per_tooth_nm = _narrows,
    width_wide_per_tooth_nm   = _wides,
    cavity_width_nm           = _cw,
    mode  = "zipped",
    label = "width_envelope_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
