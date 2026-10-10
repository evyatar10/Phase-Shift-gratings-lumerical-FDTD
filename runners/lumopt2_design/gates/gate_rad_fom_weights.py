"""Gate: rad_fom value and adjoint source carry the SAME trapz area weights.

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | lane 174499_10 review
Reviewer finding 2026-10-10: a dA-weighted value with a plain conj(E) source is
1/dA (~4e14) too large. Asserts, on a NON-uniform y grid shaped like the real
plane (dy = W/13 in the centre, 50 nm outside, E (nx, ny, 1, 1, 3)):
  V1  rad value == trapz(trapz(|E|^2, y), x)                       (same objective)
  V2  dV = 2 Re sum(src * dE) for a random perturbation (central diff)  (source = dV/dE*)
  V3  the OLD plain conj(E) source fails V2 by >1e6                 (gate can fail)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from runners.lumopt2_design import lumopt2_design as eng

rng = np.random.default_rng(0)
x = np.linspace(-25e-6, 25e-6, 1001)
y = np.concatenate([np.arange(-2.4e-6, -0.5e-6, 50e-9), np.arange(-0.5e-6, 0.5e-6, 61.5e-9),
                    np.arange(0.5e-6, 2.4e-6 + 1e-12, 50e-9)])
E = (rng.normal(size=(x.size, y.size, 1, 1, 3)) + 1j * rng.normal(size=(x.size, y.size, 1, 1, 3)))
W = eng.rad_plane_weights(E, x, y)
I = np.sum(np.abs(E) ** 2, axis=-1)
val = float(np.sum(W * I))
ref = float(np.trapz(np.trapz(I[:, :, 0, 0], y, axis=1), x))
assert abs(val - ref) <= 1e-9 * abs(ref), (val, ref)
src = eng.rad_adjoint_source(E, W)
dE = 1e-6 * (rng.normal(size=E.shape) + 1j * rng.normal(size=E.shape))
V = lambda F: float(np.sum(W * np.sum(np.abs(F) ** 2, axis=-1)))
dV_meas = (V(E + dE) - V(E - dE)) / 2.0                           # central: exact for a quadratic
dV_pred = 2.0 * float(np.real(np.sum(src * dE)))
assert abs(dV_meas - dV_pred) <= 1e-6 * abs(dV_pred), (dV_meas, dV_pred)
dV_old = 2.0 * float(np.real(np.sum(np.conj(E) * dE)))          # the pre-fix source
assert abs(dV_old / dV_pred) > 1e6, dV_old / dV_pred
print(f"GATE rad_fom_weights PASS  V={val:.4e}  dV meas/pred={dV_meas/dV_pred:.6f}  "
      f"old-source ratio={dV_old/dV_pred:.2e}")
