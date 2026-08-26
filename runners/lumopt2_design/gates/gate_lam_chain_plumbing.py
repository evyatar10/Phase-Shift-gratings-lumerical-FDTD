"""PLUMBING GATE for defect #19's selector passes (2026-08-26).

WHY THIS EXISTS: job 137267 died after 2:03 of GPU time on
    IndexError: invalid index to scalar variable
because the selector was written `anp.abs(x[0])[i]`, assuming x was a LIST of
FOM entry results. It is not: autograd takes the jacobian w.r.t. the FLAT
vector [T(lam_0) .. T(lam_{n_wl-1}), softW]. The math gates all passed --
they tested the FORMULA, never the CALL PATH. This gate drives the selectors
through autograd exactly as lumopt2's base_fom.get_jacobian does, so the same
failure is caught locally in under a second with zero GPU.

Run: python gate_lam_chain_plumbing.py
"""
import sys

sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")

import numpy as np
import autograd.numpy as anp
from autograd import jacobian

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_proj import SPEC

ok = True

# Build the REAL fct over the REAL flat layout the engine uses.
wl = np.linspace(SPEC.scan_center_nm - SPEC.scan_width_nm / 2,
                 SPEC.scan_center_nm + SPEC.scan_width_nm / 2,
                 SPEC.n_wl_points)
n_wl = len(wl)
fct_top = eng.make_fct_v2(wl, SPEC)

# A plausible flat x: a Lorentzian T grid plus the softW entry at the end.
g = 0.405
T = 0.9012 * g**2 / ((wl - SPEC.scan_center_nm) ** 2 + g**2)
x = np.concatenate([T, [18.35]])
print(f"flat x: {x.shape[0]} entries = {n_wl} wavelengths + 1 softW")

# 1. the layout assumption itself
scalar_ok = np.isscalar(x[0]) or np.ndim(x[0]) == 0
print(f"  {'OK  ' if scalar_ok else 'FAIL'} x[0] is a SCALAR (not the T array) "
      f"-- the assumption that killed 137267")
ok &= scalar_ok

# 2. the top-level fct must run on this layout (it is what wg_project uses)
try:
    v = float(fct_top(x))
    print(f"  OK   make_fct_v2 evaluates on the flat x -> {v:.6f}")
except Exception as e:
    print(f"  FAIL make_fct_v2 raised {e!r}")
    ok = False

# 3. THE ACTUAL FAILING CALL PATH: autograd jacobian of each selector.
i_pk = int(np.argmax(T))
k = max(1, min(int(round(0.5 * 0.81 / (wl[1] - wl[0]))), i_pk - 1, n_wl - 2 - i_pk))
i_lo, i_hi = i_pk - k, i_pk + k
print(f"  peak idx {i_pk}, k {k} -> i_lo {i_lo}, i_hi {i_hi}")

for name, sel, idx in (("width  x[-1]", lambda z: z[-1], n_wl),
                       ("Tlo    x[i_lo]", lambda z: anp.abs(z[i_lo]), i_lo),
                       ("Thi    x[i_hi]", lambda z: anp.abs(z[i_hi]), i_hi)):
    try:
        J = jacobian(sel)(x)
        onehot = (J.shape == x.shape and abs(J[idx] - 1.0) < 1e-9
                  and abs(J.sum() - 1.0) < 1e-9)
        print(f"  {'OK  ' if onehot else 'FAIL'} {name}: jacobian is one-hot at "
              f"{idx} (shape {J.shape}, J[{idx}]={J[idx]:+.3g}, "
              f"sum={J.sum():+.3g})")
        ok &= onehot
    except Exception as e:
        print(f"  FAIL {name}: jacobian raised {type(e).__name__}: {e}")
        ok = False

# 4. the OLD broken form must still fail -- proves this gate has teeth
try:
    jacobian(lambda z: anp.abs(z[0])[i_lo])(x)
    print("  FAIL the old x[0][i] form did NOT raise -- gate has no teeth")
    ok = False
except IndexError:
    print("  OK   the old x[0][i] form still raises IndexError (gate has teeth)")

print("\n" + ("ALL PASS" if ok else "*** GATE FAILED ***"))
raise SystemExit(0 if ok else 1)
