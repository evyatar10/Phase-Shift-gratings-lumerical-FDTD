"""
PHASE-0 (4.3) — Counterdiabatic (shortcut-to-adiabaticity) apodization. Zero GPU.

ANALYTIC MAPPING (the derivation, summarized; printed in full below):
With counter-propagating envelopes (A, B) in a grating kappa(x)*e^{i*phi(x)},
the gauge transform A~ = A e^{-i phi/2}, B~ = B e^{+i phi/2} gives
    dA~/dx = -i (phi'/2) A~ + i kappa B~
    dB~/dx = +i (phi'/2) B~ - i kappa A~
i.e. a grating PHASE GRADIENT phi'(x) acts exactly as a local detuning; the
pi-shift is phi' = pi*delta(x). The two physical quadratures are therefore:
  * IN-PHASE:   kappa(x) modulation  = tooth/gap WIDTH profile  -> this is the
    channel the derived-profile solve already optimized to a local optimum
    (route 2A, spent).
  * QUADRATURE: phi'(x) modulation   = per-tooth POSITION shifts -> the
    counterdiabatic channel. Berry's transitionless term H_cd = i(dU/dx)U^+
    for this system is purely off-diagonal-imaginary, i.e. sits in the
    QUADRATURE channel, with amplitude ~ d(theta)/dx concentrated where the
    mixing angle turns = at the defect, decaying outward with the envelope.
    Translated to hardware: a specific per-tooth position-shift profile s_j
    near the defect (sign pattern allowed), NOT a width profile.
The already-used uniform gap-shift pair [+20,+20] is the crudest member of
this family (2 equal inner kicks). The Phase-0 question: does the OPTIMAL
quadrature profile (a) exist, (b) beat the uniform pair meaningfully in the
validated in-cone kernel, (c) stay binary-etch realizable (|s_j| <= ~60 nm)?

NUMERICS: constrained least-squares in the per-tooth TRANSLATION basis, using
the SAME kernel + calibration + constraints validated in
derive_boundary_profile_stack.py (mode B: real stack field, job 118462;
calibration on the measured [20,20]->[30,30] dose step; sign-validated on
[25,25]). Constraints: resonance moment, 2*beta cos/sin (kappa / mode width),
x^2 curvature (fwhm guard).

GATE: quadrature profile exists, predicted in-cone gain beyond the uniform
pair, orthogonal content vs the uniform-pair pattern, realizable amplitudes.

Run:  python python_tools/phase0_counterdiabatic.py
"""

import os

import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")

PITCH = 0.51683
HALF = PITCH / 2
S_UM = 0.020
CAV_LEN = HALF + 2 * S_UM
N_CLAD = 1.444
W_CAV, W_T1, W_T2, W_T, W_GAP = 1.050, 1.040, 0.980, 1.000, 0.600

# ------------------------------------------------------------------- kernel
# (verbatim structure from derive_boundary_profile_stack.py — validated)
m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6
y = np.asarray(m["y"], float) * 1e6
Ez2d = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam_res = float(m["resonance_wavelength_nm"])

xu = np.linspace(x.min(), x.max(), 1 << 12)
dx = xu[1] - xu[0]


def resample_row(y_target):
    j = int(np.argmin(np.abs(y - y_target)))
    row = Ez2d[:, j]
    return np.interp(xu, x, row.real) + 1j * np.interp(xu, x, row.imag), y[j]


def tooth_edges():
    """[(inner, outer, width), ...] per tooth outward, symmetrized layout."""
    c2 = CAV_LEN / 2
    out = [(c2, c2 + HALF, W_T1)]
    g1b = c2 + HALF + HALF - S_UM
    out.append((g1b, g1b + HALF, W_T2))
    a = g1b + HALF + HALF
    for j in range(12):
        out.append((a, a + HALF, W_T))
        a += 2 * HALF
    return out


EDGES = tooth_edges()


def width_profile(xs):
    W = np.full_like(xs, W_GAP)
    ax = np.abs(xs)
    W[ax <= CAV_LEN / 2] = W_CAV
    for (a, b, w) in EDGES:
        W[(ax > a) & (ax <= b)] = w
    return W


Wx = width_profile(xu)
E_wall = np.empty_like(xu, dtype=complex)
for w in np.unique(Wx):
    sel = Wx == w
    row, _ = resample_row(w / 2 - 0.05)
    E_wall[sel] = row[sel]

E_rad, y0 = resample_row(1.6)
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(xu), dx))
kc = 2 * np.pi / (lam_res * 1e-3) * N_CLAD
IN = np.abs(k) < kc
F = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))
A0 = F(E_rad)[IN]
P0 = float(np.sum(np.abs(A0) ** 2))
ky = np.sqrt(np.maximum(kc ** 2 - k[IN] ** 2, 0.0))
prop = np.exp(1j * ky * (y0 - 0.6))


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
    """calibration pattern: BOTH inner gap shifts +s on top of the stack's
    +20 (verbatim from mode B)."""
    d = np.zeros_like(xu)
    s = s_extra_nm * 1e-3
    c2 = CAV_LEN / 2
    deposit(d, c2 - s, s, +(W_T1 - W_CAV) / 2)
    deposit(d, c2 + HALF - s, s, -(W_T1 - W_GAP) / 2)
    g1b = c2 + HALF + HALF - S_UM
    deposit(d, g1b - s, s, +(W_T2 - W_GAP) / 2)
    deposit(d, g1b + HALF - s, s, -(W_T2 - W_GAP) / 2)
    return d


def tooth_translation(j, s_nm):
    """delta(x) of translating tooth j toward the cavity by s (nm),
    x-symmetric. Inner edge: tooth width replaces the medium inside it;
    outer edge: gap replaces tooth."""
    d = np.zeros_like(xu)
    s = abs(s_nm) * 1e-3
    sg = np.sign(s_nm)
    a, b, w = EDGES[j]
    w_in = W_CAV if j == 0 else W_GAP
    deposit(d, a - s, s, sg * (w - w_in) / 2)
    deposit(d, b - s, s, -sg * (w - W_GAP) / 2)
    return d


# ------------------------------------------------------------- calibration
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
      f"pred [30,30] step {best[2]*100:+.1f}% vs meas {MEAS_STEP*100:+.1f}%")
v = rel_dloss(extra_pair_shift(5.0), C)
print(f"VALIDATION [25,25] step: pred {v*100:+.1f}% vs meas -6.9%  "
      f"{'OK' if np.sign(v) < 0 else 'SIGN-FAIL'}")

# --------------------------------------------- constrained quadrature solve
NT = 14                                   # teeth per side in the basis
S_REF = 10.0                              # nm reference amplitude per pattern
B = np.array([tooth_translation(j, S_REF) for j in range(NT)])
M_resp = np.array([C * dA_of(b) for b in B]).T
w_res = np.abs(E_wall) ** 2
beta = np.pi / PITCH
Cons = np.vstack([
    B @ w_res,
    B @ (w_res * np.cos(2 * beta * xu)),
    B @ (w_res * np.sin(2 * beta * xu)),
    B @ (w_res * xu ** 2 / max(np.max(xu ** 2), 1e-30)),
])
Mr = np.vstack([M_resp.real, M_resp.imag])
A0r = np.concatenate([A0.real, A0.imag])
ridge = 1e-3 * np.linalg.norm(Mr, ord=2) ** 2
H_ = Mr.T @ Mr + ridge * np.eye(NT)
g = -Mr.T @ A0r
nc = Cons.shape[0]
KKT = np.block([[H_, Cons.T], [Cons, np.zeros((nc, nc))]])
sol = np.linalg.solve(KKT, np.concatenate([g, np.zeros(nc)]))
c_opt = sol[:NT]                          # in units of S_REF nm
s_opt = c_opt * S_REF                     # nm per tooth

delta_opt = B.T @ c_opt
pred_opt = rel_dloss(delta_opt, C)
print("\nOPTIMAL QUADRATURE (per-tooth translation) PROFILE, nm toward cavity:")
print("  tooth 1..14:", "  ".join(f"{v:+.1f}" for v in s_opt))
print(f"predicted in-cone change: {pred_opt*100:+.1f}%")

# linearity check: the kernel is linear in delta but the patterns were built
# at S_REF — rebuild at the actual amplitudes and re-evaluate
delta_exact = np.sum([tooth_translation(j, s_opt[j]) for j in range(NT)], axis=0)
print(f"re-evaluated with exact per-tooth strips: "
      f"{rel_dloss(delta_exact, C)*100:+.1f}%")

# comparison anchors
d_pair = extra_pair_shift(10.0)
pred_pair = rel_dloss(d_pair, C)
print(f"\nanchor: uniform extra pair [+10,+10] on the stack: {pred_pair*100:+.1f}%")
# cos-similarity of the optimal profile to the uniform-pair direction
pair_vec = np.zeros(NT)
pair_vec[:2] = 1.0
cs = float(np.dot(c_opt, pair_vec) / (np.linalg.norm(c_opt) * np.linalg.norm(pair_vec) + 1e-30))
print(f"cos-similarity(optimal, uniform inner pair) = {cs:+.2f}  "
      f"-> orthogonal content {100*(1-cs**2):.0f}%")

# scale ladder: predicted gain vs overall amplitude scale (dose curve)
print("\ndose curve of the optimal profile (scale x profile):")
for sc in (0.5, 1.0, 1.5, 2.0, 3.0):
    de = np.sum([tooth_translation(j, sc * s_opt[j]) for j in range(NT)], axis=0)
    mx = np.max(np.abs(sc * s_opt))
    print(f"  scale {sc:3.1f} (max |s| {mx:5.1f} nm): {rel_dloss(de, C)*100:+7.1f}%")

print(f"""
GATE CHECKS:
 * distinct channel from the derived width profile (route 2A)?  YES by
   construction — translations are the phi'(x) quadrature; the width solve
   held tooth positions fixed.
 * realizable? max |s_j| above; binary etch trivially (tooth positions move,
   widths unchanged); acceptance needs |s_j| <= ~60 nm and min gap > ~100 nm.
 * beats the uniform pair? compare the predicted numbers above; the uniform
   pair is the cs-parallel part — the orthogonal remainder is the NEW claim.
CAVEATS (honest): same kernel limits as mode B (single-wall symmetrized
layout, in-cone weight proxy, C calibrated on one dose step); the x^2 fwhm
guard is a proxy — width is re-measured at every FDTD point regardless.
""")
