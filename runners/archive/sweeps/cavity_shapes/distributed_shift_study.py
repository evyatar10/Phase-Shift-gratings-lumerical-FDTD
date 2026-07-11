"""
DISTRIBUTED pi-SHIFT study — the theory-derived cavity design (no optimizer).

Math (user-requested theory-first): the TM loss is kink radiation — the pi
phase slip is lumped in one half-pitch segment, and the radiated amplitude into
near-axis cladding channels is the Fourier transform of the phase-slip density
g(x) at Delta-k = beta - k_clad ~ 0.26 rad/um. A lumped slip has flat Fourier
content; SPREADING the same total pi over a length L >~ 1/Delta-k ~ 4 um
suppresses everything except the irreducible DC term (the P_rad bound).
Gaussian g(x) is the transform-optimal smooth density.

Implementation via the existing per-tooth gap-shift machinery (inner_shift_nm):
NEGATIVE shift s_d lengthens the narrow gap at tooth d by |s_d| on each side,
and lengthen_cavity=True automatically SHRINKS the central segment by 2*sum|s|,
so total grating length and total round-trip phase are conserved — the pi slip
is redistributed, corrugation depth NEVER changes (this is not apodization, and
not the single-tooth shift tried before).

Rows (zipped, 6 tasks, optimization mesh, converged box, window 1558.5/40/3001):
  0 control (lumped pi-shift, the standard device)
  1 N=2 uniform, HALF distributed  (cavity 258->129 nm)
  2 N=2 uniform, fully distributed (cavity -> 20 nm)
  3 N=4 Gaussian, fully distributed
  4 N=8 Gaussian, fully distributed (the theory-optimal candidate)
  5 N=8 uniform, fully distributed (density-shape control)
Predictions (registered): loss falls with spread length; Gaussian <= uniform at
equal N; mode fwhm grows by roughly the spread length (~+1-4 um) — the measured
exchange rate is the deliverable. lambda_res should stay ~fixed (phase
conserved); a big shift means the implementation, not the theory, is wrong.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER cavity_shape2_study in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.distributed_shift_study
Output -> results/distributed_shift_study/results/.
"""

import math
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
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_HALF_CAV = 516.83 / 4.0              # cavity_length/2 = 129.21 nm
_FULL = _HALF_CAV - 10.0              # fully distributed, 20 nm cavity sliver left


def _gauss_shifts(n, total, sigma):
    w = [math.exp(-(((d - 1) / sigma) ** 2)) for d in range(1, n + 1)]
    s = sum(w)
    return [round(-total * wi / s, 2) for wi in w]


def _uniform_shifts(n, total):
    return [round(-total / n, 2)] * n


_rows = [
    None,                                  # control
    _uniform_shifts(2, _HALF_CAV / 2.0),   # half distributed
    _uniform_shifts(2, _FULL),
    _gauss_shifts(4, _FULL, 2.0),
    _gauss_shifts(8, _FULL, 3.5),
    _uniform_shifts(8, _FULL),
]
_nfree = [1] + [len(r) for r in _rows[1:]]

_n = 6
assert len(_rows) == len(_nfree) == _n
_keys = {tuple(r) if r else () for r in _rows}
assert len(_keys) == _n, "duplicate shift row -> tag collision"
# never shrink the cavity below zero
for r in _rows[1:]:
    assert 2.0 * sum(abs(v) for v in r) < 516.83 / 2.0, "cavity length would go negative"

SPEC = SweepSpec(
    inner_shift_list_nm = _rows,
    n_free_inner_teeth  = _nfree,
    mode  = "zipped",
    label = "distributed_shift_study",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
