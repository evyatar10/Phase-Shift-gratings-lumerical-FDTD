"""
DERIVED-PROFILE TEST — the FDTD referee of the "golden shape" question
(user-selected route; docs chain: derive_boundary_profile*.py, field job
118462, kernel validated on the measured shift dose curve).

The constrained boundary-perturbation solve on the STACK's true resonant field
(mode B) returned a per-segment sidewall profile that is NOT a taper and NOT
any previously-scanned pattern — its largest components are GAP-width changes
(gap1 +19 nm, gap2 -13 nm) coupled with inner-tooth retunes, applied ON TOP of
the stack (W1050 + gap pair [+20,+20] + see-saw 1040/980). Truncated to the
inner 3 periods (outer segments < +-6 nm, at/below fab+mesh relevance):

  segment      stack value   derived x1
  cavity       1050          1031.2
  tooth1 wide  1040          1027.7
  gap1         600           618.7
  tooth2 wide  980           997.8
  gap2         600           587.0
  tooth3 wide  1000          1009.8
  gap3         600           587.7

Registered predictions (filed before dispatch):
  P1 x1 profile IMPROVES the stack: model floor -5.1% relative in-cone
     (loss 0.0545 -> <= 0.052); the kernel underpredicts measured magnitudes
     3-6x, so up to ~ -15-30% relative is plausible. Below -2x accurate floor
     (~4e-4 absolute) = null.
  P2 SIGN-FLIP (x-1) must WORSEN by a comparable amount — the falsification
     row. If x-1 also improves, the kernel is wrong and the result is just
     more frontier dose.
  P3 DOSE: response ~linear at x0.5, near-optimal between x1 and x2 (the
     model is first-order; expect rollover by x2).
  P4 fwhm_m within +-1% of the W800 control (the x^2-curvature constraint
     enforced this in the solve; FDTD referees).

Rows (zipped, 7 tasks, accurate mesh, converged box, window 1558.5/40/3001):
   0  control W800 avg          (anchor 0.1174)
   1  the stack                 (reproduce 0.0545)
   2  stack + derived x1
   3  stack + derived x0.5
   4  stack + derived x2
   5  stack + derived x(-1)     (falsification)
   6  jitter partner: derived x1 with cavity +2 nm (sub-mesh)

Dispatch (queue must be EMPTY of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_derived_profile
Output -> results/tm_derived_profile/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# stack values and derived per-segment deltas (nm)
CAV0, T0, G0 = 1050.0, [1040.0, 980.0, 1000.0], 600.0
D_CAV, D_T, D_G = -18.8, [-12.3, +17.8, +9.8], [+18.7, -13.0, -12.3]
PAIR = [20.0, 20.0]


def derived(scale, cav_extra=0.0):
    return dict(
        cav=CAV0 + scale * D_CAV + cav_extra,
        ptww=[t + scale * d for t, d in zip(T0, D_T)],
        ptwn=[G0 + scale * d for d in D_G],
    )


_cw, _ishl, _nfree, _ptwn, _ptww = [], [], [], [], []


def row(w=None, ishl=None, nfree=1, ptwn=None, ptww=None):
    _cw.append(w); _ishl.append(ishl); _nfree.append(nfree)
    _ptwn.append(ptwn); _ptww.append(ptww)


row()                                                          # 0 control
row(w=CAV0, ishl=PAIR, nfree=2,
    ptwn=[G0, G0], ptww=[1040.0, 980.0])                       # 1 the stack
for sc, extra in ((1.0, 0.0), (0.5, 0.0), (2.0, 0.0),
                  (-1.0, 0.0), (1.0, 2.0)):                    # 2-6
    p = derived(sc, extra)
    row(w=p["cav"], ishl=PAIR, nfree=2, ptwn=p["ptwn"], ptww=p["ptww"])

_n = 7
assert all(len(v) == _n for v in (_cw, _ishl, _nfree, _ptwn, _ptww))
_keys = [(w, tuple(pw) if pw else None) for w, pw in zip(_cw, _ptww)]
assert len(set(_keys)) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm           = _cw,
    inner_shift_list_nm       = _ishl,
    n_free_inner_teeth        = _nfree,
    width_narrow_per_tooth_nm = _ptwn,
    width_wide_per_tooth_nm   = _ptww,
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_derived_profile",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
