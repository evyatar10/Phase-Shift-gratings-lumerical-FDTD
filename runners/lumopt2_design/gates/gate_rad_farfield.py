"""Gate: far-field (angular-spectrum) objective: value/source consistency + physics.

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | rad_farfield
  F1  central difference: dV = 2 Re sum(src * dE)  (source = conj(dV/dE*) on the MONITOR
      grid, through the y-resampling transpose)
  F2  physics: a plane wave 30 deg off-normal gives V/V(normal) = cos 30 to 2 %;
      an evanescent wave (k_par = 1.5 k0) gives V < 5e-3 of normal (window leak)
  F3  the plain |E|^2 plane sum (old objective) is angle-blind (the gate can fail)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from runners.lumopt2_design import lumopt2_design as eng  # noqa: E402

rng = np.random.default_rng(1)
lam, n = 1.56e-6, 1.444
x = np.arange(-25e-6, 25e-6, 50e-9)
y = np.concatenate([np.arange(-2.4e-6, -0.5e-6, 50e-9), np.arange(-0.5e-6, 0.5e-6, 61.5e-9),
                    np.arange(0.5e-6, 2.4e-6 + 1e-12, 50e-9)])
shape = (x.size, y.size, 1, 1, 3)
E = rng.normal(size=shape) + 1j * rng.normal(size=shape)
V, G = eng.farfield_power(E, x, y, lam, n)
src = eng.farfield_source(G)
dE = 1e-6 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
dV = (eng.farfield_power(E + dE, x, y, lam, n)[0] - eng.farfield_power(E - dE, x, y, lam, n)[0]) / 2
pred = 2.0 * float(np.real(np.sum(src * dE)))
assert abs(dV - pred) <= 1e-6 * abs(pred), (dV, pred)

k0 = 2 * np.pi * n / lam


def wave(kpar):
    F = np.zeros(shape, complex)
    F[:, :, 0, 0, 1] = np.exp(1j * kpar * x)[:, None]
    return F


V0 = eng.farfield_power(wave(0.0), x, y, lam, n)[0]
V30 = eng.farfield_power(wave(k0 * np.sin(np.radians(30))), x, y, lam, n)[0]
Ve = eng.farfield_power(wave(1.5 * k0), x, y, lam, n)[0]   # finite-window sinc leak ~1e-3 at 1.3 k0
assert abs(V30 / V0 - np.cos(np.radians(30))) < 0.02, V30 / V0
assert Ve / V0 < 5e-3, Ve / V0


def plain(F):
    return float(np.sum(eng.rad_plane_weights(F, x, y) * np.sum(np.abs(F) ** 2, axis=-1)))


assert abs(plain(wave(k0 * np.sin(np.radians(30)))) / plain(wave(0.0)) - 1.0) < 1e-6   # old objective is angle-blind
print(f"GATE rad_farfield PASS  dV meas/pred={dV / pred:.6f}  V30/V0={V30 / V0:.4f} "
      f"(cos30={np.cos(np.radians(30)):.4f})  evanescent={Ve / V0:.1e}")
