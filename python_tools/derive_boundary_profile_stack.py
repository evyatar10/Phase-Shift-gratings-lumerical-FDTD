"""
DERIVED BOUNDARY PROFILE — MODE B: the real 2D resonant field of the STACK
(W1050 + gap pair [+20,+20] + see-saw 1040/980) from tm_field_export
(job 118462), no EIM wall approximation.

Method (same validated kernel structure as derive_boundary_profile.py):
  E_wall(x)   = complex E_z sampled just inside the local sidewall y=W(x)/2-eps
  A0(kx)      = in-cone FFT of E_z sampled on a parallel line in the cladding
                (y0 ~ 1.6 um) — the device's residual radiating spectrum
  dA(kx)      = C * FFT[delta(x) * E_wall(x)] propagated to y0
  calibration = |C| fit on the measured dose step pair[20,20]->[30,30]
                (W1050 base, job 118214: 0.0549 -> 0.0454, +2.5% fw — the
                fw-violating part rides the constrained-out manifold; the
                in-cone kernel sees mostly the matching part), validated on
                pair[25,25] and see-saw steps.
  solve       = constrained least squares over per-half-pitch segments,
                constraints: resonance moment, 2*beta (kappa), x^2 curvature
                (fwhm guard) — as validated in mode A.

Outputs: segment table (nm) -> DIRECTLY mappable to width_wide_per_tooth /
width_narrow_per_tooth lists; predicted residual in-cone change; decomposition
vs known knobs; comparison against the mode-A profile.

Run:  python python_tools/derive_boundary_profile_stack.py
"""

import os
import sys

import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")

PITCH = 0.51683
HALF = PITCH / 2
S_UM = 0.020                  # gap-shift of the stack (both inner gaps)
CAV_LEN = HALF + 2 * S_UM     # lengthen_cavity absorbed 2*sum
N_CLAD = 1.444

# ------------------------------------------------------------------- load
# EZSLICE format (server-side reduction of the 655-MB full export): x, y (m),
# Ez_re/Ez_im at the resonance-nearest frequency.
m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6          # um
y = np.asarray(m["y"], float) * 1e6
Ez2d = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam_res = float(m["resonance_wavelength_nm"])
print(f"field loaded: {Ez2d.shape}, lam used {float(m['lam_used_nm']):.2f} "
      f"vs res {lam_res:.2f}")

# uniform x resample (monitor grids are non-uniform — the 2026-07 k-space bug)
xu = np.linspace(x.min(), x.max(), 1 << 12)
dx = xu[1] - xu[0]


def resample_row(y_target):
    j = int(np.argmin(np.abs(y - y_target)))
    row = Ez2d[:, j]
    return np.interp(xu, x, row.real) + 1j * np.interp(xu, x, row.imag), y[j]


# ------------------------------------------------- stack width profile W(x)
def width_profile(xs):
    """Stack: cavity (1.05, len CAV_LEN centered); tooth1 (1.04, len HALF)
    shifted S toward cavity; gap1 shortened by S; tooth2 0.98; then nominal
    1.0/0.6 alternation. Symmetrized in |x| (kernel approximation, as
    validated in mode A)."""
    W = np.full_like(xs, 0.600)
    ax = np.abs(xs)
    c2 = CAV_LEN / 2
    W[ax <= c2] = 1.050
    # tooth1: starts right at the cavity edge (it moved inward by S)
    t1a, t1b = c2, c2 + HALF
    W[(ax > t1a) & (ax <= t1b)] = 1.040
    # gap1 shortened by S: from t1b to t1b + (HALF - S)... total period kept
    g1b = t1b + HALF - S_UM
    # tooth2:
    t2b = g1b + HALF
    W[(ax > g1b) & (ax <= t2b)] = 0.980
    # beyond: nominal alternation starting with gap
    seg = np.floor((ax - t2b) / HALF).astype(int)
    wide = (ax > t2b) & (seg % 2 == 1)
    W[wide] = 1.000
    return W


Wx = width_profile(xu)
# sample E just inside the wall: y = W/2 - 0.05 um, per x
E_wall = np.empty_like(xu, dtype=complex)
for w in np.unique(Wx):
    sel = Wx == w
    row, yr = resample_row(w / 2 - 0.05)
    E_wall[sel] = row[sel]

# residual radiating spectrum from a cladding line
E_rad, y0 = resample_row(1.6)
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xu), dx))
kc = 2 * np.pi / (lam_res * 1e-3) * N_CLAD
IN = np.abs(k) < kc
F = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))
A0 = F(E_rad)[IN]
P0 = float(np.sum(np.abs(A0) ** 2))
# propagation phase wall->y0 for the perturbation source
ky = np.sqrt(np.maximum(kc ** 2 - k[IN] ** 2, 0.0))
prop = np.exp(1j * ky * (y0 - 0.6))          # mean wall y ~ 0.5-0.7


def dA_of(delta):
    return F(delta * E_wall)[IN] * prop


def rel_dloss(delta, C):
    return float(np.sum(np.abs(A0 + C * dA_of(delta)) ** 2)) / P0 - 1.0


def deposit(d, x0, length_um, height):
    for sgn in (+1, -1):
        a, b = sorted((sgn * x0, sgn * (x0 + length_um)))
        lo, hi = np.searchsorted(xu, a) - 1, np.searchsorted(xu, b) + 1
        for i in range(max(lo, 0), min(hi, len(xu) - 1)):
            ov = max(0.0, min(b, xu[i] + dx / 2) - max(a, xu[i] - dx / 2))
            d[i] += height * ov / dx
    return d


def extra_pair_shift(s_extra_nm):
    """delta(x) of increasing BOTH inner gap shifts by s_extra (on top of the
    stack's +20): tooth edges move inward; cavity grows. Deposits at the three
    moving boundaries per side (see mode A discussion)."""
    d = np.zeros_like(xu)
    s = s_extra_nm * 1e-3
    c2 = CAV_LEN / 2
    deposit(d, c2 - s, s, +(1.040 - 1.050) / 2)          # tooth1 in over cavity edge
    deposit(d, c2 + HALF - s, s, +(1.040 - 0.600) / 2)   # tooth1 claims gap1... sign:
    # tooth1 shifts inward: its outer edge recedes -> gap widens there? NO —
    # increasing the SHIFT moves tooth1 further toward cavity: outer edge at
    # c2+HALF moves in by s -> gap material (0.6) replaces tooth (1.04):
    d = np.zeros_like(xu)
    deposit(d, c2 - s, s, +(1.040 - 1.050) / 2)
    deposit(d, c2 + HALF - s, s, -(1.040 - 0.600) / 2)
    # tooth2 shift: same structure one period out (gap1 end / tooth2 edges)
    g1b = c2 + HALF + HALF - S_UM
    deposit(d, g1b - s, s, +(0.980 - 0.600) / 2)
    deposit(d, g1b + HALF - s, s, -(0.980 - 0.600) / 2)
    return d


# ------------------------------------------------------------- calibration
# measured on W1050 base (fields ~ stack): [20,20]->[30,30]: loss 0.0549->0.0454
MEAS_STEP = (0.0454 - 0.0549) / 0.0549
best = None
for phase in np.linspace(0, 2 * np.pi, 73):
    for mag in np.geomspace(1e-3, 1e3, 121):
        C = mag * np.exp(1j * phase)
        pred = rel_dloss(extra_pair_shift(10.0), C)
        if best is None or abs(pred - MEAS_STEP) < best[0]:
            best = (abs(pred - MEAS_STEP), C, pred)
C = best[1]
print(f"calibrated C = {abs(C):.3g} exp(i{np.angle(C):+.2f}); "
      f"pred step {best[2]*100:+.1f}% vs meas {MEAS_STEP*100:+.1f}%")
# validation: [20,20]->[25,25] measured 0.0549->0.0511 = -6.9%
v = rel_dloss(extra_pair_shift(5.0), C)
print(f"VALIDATION [25,25] step: pred {v*100:+.1f}% vs meas -6.9%  "
      f"{'OK' if np.sign(v) < 0 else 'SIGN-FAIL'}")

# ------------------------------------------------ constrained solve (as A)
N_SEG = 14
c2 = CAV_LEN / 2
segs = [(np.abs(xu) <= c2).astype(float)]
for j in range(N_SEG):
    lo = c2 + j * HALF
    segs.append(((np.abs(xu) > lo) & (np.abs(xu) <= lo + HALF)).astype(float))
B = np.array(segs)
M = np.array([C * dA_of(b) for b in B]).T
w_res = np.abs(E_wall) ** 2
beta = 2 * np.pi / (lam_res * 1e-3) * (lam_res * 1e-3 / (2 * PITCH))
Cons = np.vstack([
    B @ w_res,
    B @ (w_res * np.cos(2 * beta * xu)),
    B @ (w_res * np.sin(2 * beta * xu)),
    B @ (w_res * xu ** 2 / max(np.max(xu ** 2), 1e-30)),
])
Mr = np.vstack([M.real, M.imag])
A0r = np.concatenate([A0.real, A0.imag])
ridge = 1e-3 * np.linalg.norm(Mr, ord=2) ** 2
H_ = Mr.T @ Mr + ridge * np.eye(B.shape[0])
g = -Mr.T @ A0r
nb, nc = B.shape[0], Cons.shape[0]
KKT = np.block([[H_, Cons.T], [Cons, np.zeros((nc, nc))]])
sol = np.linalg.solve(KKT, np.concatenate([g, np.zeros(nc)]))
d_opt = sol[:nb]
pred_opt = rel_dloss(B.T @ d_opt, C)
print(f"\nOPTIMAL constrained profile (STACK field): predicted in-cone change "
      f"{pred_opt*100:+.1f}%")
print("segment delta (nm): cavity, then per half-pitch outward")
print("  " + "  ".join(f"{v*1e3:+.1f}" for v in d_opt))
print("\nWidth-list mapping (2*delta per segment): tooth/gap width changes, nm:")
print("  " + "  ".join(f"{2*v*1e3:+.1f}" for v in d_opt))
