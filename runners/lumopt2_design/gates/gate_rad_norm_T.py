"""Gate: detuning-free far-field objective -log(V / T(lambda_rad)) through the REAL fct.

Study dir: runners/lumopt2_design/ | Created 2026-10-11 | rad_norm_T (S2 far-field lane)
Synthetic S2-like line (Lorentzian, FWHM 0.2 nm) on the S2 grid (2 nm / 501):
  N1  jacobian through make_fct_v2 + autograd: dJ/dV = -1/V, dJ/dT nonzero ONLY at the
      two bracket samples with weights (1-w)/T_i, w/T_i, dJ/dsoftW = 0
  N2  detuning invariance: move the line 0.05 nm off the plane lambda (V and T scale by the
      same Lorentzian) -> |dJ| < 1e-3; the legacy -log V changes by > 0.15 (gate can fail)
  N3  knob off -> legacy objective, plane lambda outside the grid -> raises
"""
import dataclasses
import os
import sys

import autograd
import autograd.numpy as anp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from runners.lumopt2_design import lumopt2_design as eng  # noqa: E402
from runners.lumopt2_design import campaign_te_s2 as s2  # noqa: E402

base = dataclasses.replace(s2.SPEC_V3, rad_fom=True, rad_farfield=True)
spec = dataclasses.replace(base, rad_norm_T=True)
wl = np.linspace(1559.464, 1561.464, 501)
lam_rad = 1560.4637
spec._rad_lam_nm = lam_rad
base._rad_lam_nm = lam_rad


def lor(c, fwhm=0.2):
    return 1.0 / (1.0 + (2.0 * (wl - c) / fwhm) ** 2)


def lor_at(c, x, fwhm=0.2):
    return 1.0 / (1.0 + (2.0 * (x - c) / fwhm) ** 2)


V0, softw = 3.7e4, 19.6
fct = eng.make_fct_v2(wl, spec)
fct_old = eng.make_fct_v2(wl, base)


def x_of(c):
    return anp.concatenate([0.98 * lor(c), anp.array([V0 * lor_at(c, lam_rad), softw])])


x = x_of(lam_rad)
J = autograd.jacobian(fct)(x)
nz = np.nonzero(np.abs(J[:wl.size]) > 0)[0]
assert nz.size == 2 and abs(J[wl.size] + 1.0 / x[wl.size]) < 1e-12 and J[wl.size + 1] == 0.0, ("N1", nz)
k = int(np.searchsorted(wl, lam_rad) - 1)
w = (lam_rad - wl[k]) / (wl[k + 1] - wl[k])
Ti = (1 - w) * x[k] + w * x[k + 1]
assert np.allclose(J[[k, k + 1]], [(1 - w) / Ti, w / Ti], rtol=1e-9), "N1 weights"

dJ = fct(x_of(lam_rad + 0.05)) - fct(x)
dJ_old = fct_old(x_of(lam_rad + 0.05)) - fct_old(x)
assert abs(dJ) < 1e-3, ("N2", dJ)
assert dJ_old > 0.15, ("N2 legacy", dJ_old)

assert abs(fct_old(x) + np.log(x[wl.size])) < 1e-12, "N3 legacy"
spec._rad_lam_nm = 1570.0
try:
    fct(x)
    raise AssertionError("N3 outside grid did not raise")
except RuntimeError:
    pass
print(f"GATE rad_norm_T PASS  dJ/dV ok, 2 T weights ok, detune 0.05 nm: normalized {dJ:+.2e} vs legacy {dJ_old:+.3f}")
