# postarray_calibrate.py (study 2026-10-23, doc docs/HANDOFF_postarray_2026-10-23.md): fit the calibrated post model
# E = C(1 + kappa s x1) + s Bp on the 13-device PRODUCTION dataset (postarray_work/dataset_postdevices.npz), leave-one-out test,
# saves postarray_work/calib_s_kappa.npy. The kappa term is SUSPECT (Fable review): refit on CONVERGED devices before trusting it.
import numpy as np
d = np.load(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/postarray_work/dataset_postdevices.npz", allow_pickle=True)
C, dW, names = d["C"], d["dW"], d["names"]; ip = lambda a, b: np.sum(np.conj(a) * b * dW); cc = ip(C, C).real; Qi0 = 5.354e5; n = len(names)
x1 = np.array([ip(C, d[f"Bp_{i}"]).real / cc for i in range(n)]); x2 = np.array([ip(d[f"Bp_{i}"], d[f"Bp_{i}"]).real / cc for i in range(n)])
g = np.array([d[f"Qi_{i}"] / Qi0 for i in range(n)])
# model: E = C (1 + kap s x1) + s Bp  ->  P = m^2 + 2 m s x1 + s^2 x2,  m = 1 + kap s x1  (C-aligned "recycling" term ~ Born interference)
S_, K_ = np.meshgrid(np.linspace(0.4, 1.4, 101), np.linspace(0, 5, 101), indexing="ij")
def P(i, s, k): m = 1 + k * s * x1[i]; return m * m + 2 * m * s * x1[i] + s * s * x2[i]
def fit(idx):
    err = sum((1 / P(i, S_, K_) - g[i]) ** 2 for i in idx); j = np.unravel_index(np.argmin(err), err.shape); return S_[j], K_[j], np.sqrt(err[j] / len(idx))
s, k, rms = fit(range(n)); born = np.array([1 / P(i, 0.7, 0) for i in range(n)])
print(f"fit all 13: s {s:.2f} kappa {k:.2f} rms gain error {rms:.3f}   | plain Born x0.70 rms {np.sqrt(np.mean((born-g)**2)):.3f}")
print("leave-one-out (fit on other 12, predict this one):")
for j in range(n):
    s_, k_, _ = fit([i for i in range(n) if i != j]); print(f"  {names[j]:26s} measured x{g[j]:.3f}  calibrated x{1/P(j, s_, k_):.3f}  Born x{born[j]:.3f}")
np.save(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/postarray_work/calib_s_kappa.npy", np.array([s, k]))
