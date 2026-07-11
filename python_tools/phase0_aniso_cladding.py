"""
PHASE-0 THEORY GATE (zero-GPU, 2026-07-07) — ANISOTROPIC / low-index CLADDING.

Question (user): could an anisotropic cladding do anything, and is it a *new*
lever or the same thing as the air trench?

Mechanism. The residual radiation of the stack escapes LATERALLY (in +/-y) as a
near-grazing wave: measured |ux| ~ 0.98 => the in-plane wavevector kx ~ 0.98*kc,
so the escaping lateral waves sit right at the cladding light-line edge kc =
n_clad*k0. A cladding whose EFFECTIVE lateral index is lower than n_clad (an
anisotropic form-birefringent SWG metamaterial, or simply a lower-index lateral
region = the air trench) moves the lateral light-line to kc' = n_eff*k0. Every
radiated component with |kx| > kc' can no longer propagate laterally -> it is
evanescent in y -> trapped (TIR). Because the radiation is PILED UP at the edge
(it is the band-edge envelope tail), even a small index drop cuts a big fraction.

This computes, from the MEASURED mode field, the ceiling:
   cut(n_eff) = (lateral flux with kc' < |kx| < kc) / (total lateral flux in cone)
i.e. the maximum fraction of the in-plane radiated power an anisotropic cladding
of lateral index n_eff can remove. It is a CEILING (assumes perfect TIR, ignores
back-scatter into the guided mode and the vertical channel).

HONEST FRAMING: this is the SAME light-cone lever as the air trench, just applied
as a homogeneous surrounding layer instead of two discrete walls. It is NOT a new
physical mechanism; it is cladding engineering. Its only advantages over the
trench are (a) it can be stronger (surrounds the mode on the light-line) and
(b) an SWG metamaterial cladding is a standard planar SOI fab step. If the goal
is "novel physics beyond cladding engineering", this does not qualify.

Run:  python python_tools/phase0_aniso_cladding.py
"""

import os
import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")
N_CLAD = 1.444
NX = 1 << 13

m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6          # um
y = np.asarray(m["y"], float) * 1e6          # um
E = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam = float(m["lam_used_nm"]) * 1e-3         # um
k0 = 2 * np.pi / lam
kc = N_CLAD * k0

xu = np.linspace(x.min(), x.max(), NX)
dx = xu[1] - xu[0]
kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, dx))
FT = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dx
win = np.hanning(NX)


def cladding_line_spectrum(y0):
    """|A(kx)|^2 lateral-flux spectrum of the field sampled on a cladding y-line."""
    j = int(np.argmin(np.abs(y - y0)))
    r = E[:, j]
    f = np.interp(xu, x, r.real) + 1j * np.interp(xu, x, r.imag)
    A = FT(win * f)
    inc = np.abs(kx) < kc
    ky = np.sqrt(np.maximum(kc**2 - kx[inc]**2, 0.0))
    flux = (ky / kc) * np.abs(A[inc])**2      # lateral (y) power flux weight
    return kx[inc], flux, y[j]


# choose a cladding line well outside the guided mode (near the box edge, before PML)
y_edge = min(2.6, 0.85 * y.max())
kin, flux, yline = cladding_line_spectrum(+y_edge)
kin2, flux2, _ = cladding_line_spectrum(-y_edge)
flux = 0.5 * (flux + flux2)                   # average both cladding lines
tot = float(np.sum(flux))
u = np.abs(kin) / kc                           # |kx|/kc in [0,1]

print(f"lam_used = {lam*1e3:.2f} nm   n_clad = {N_CLAD}   cladding line y = {yline:+.2f} um")
print(f"total in-cone lateral flux (arb) = {tot:.3e}\n")

# angular / edge concentration of the lateral radiation
print("Cumulative lateral radiated power vs |kx|/kc  (how edge-piled it is):")
for thr in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
    frac = float(np.sum(flux[u >= thr])) / tot
    print(f"   fraction with |kx|/kc >= {thr:.2f}:  {frac*100:5.1f}%   "
          f"(grazing angle from y-axis >= {np.degrees(np.arcsin(min(thr,1))):.1f} deg)")

mean_u = float(np.sum(flux * u) / tot)
print(f"\n   flux-weighted mean |kx|/kc = {mean_u:.3f}  (near-axial if ->1)")

# ceiling: cut fraction vs anisotropic lateral index n_eff
print("\nCEILING: max in-plane loss reduction from a lateral index n_eff < n_clad")
print("   (cuts every component with kc' < |kx| < kc; kc' = n_eff*k0):")
loss0_inplane = 0.0545 * 0.58        # stack loss * measured in-plane share (~58%)
for n_eff in (1.40, 1.35, 1.30, 1.25, 1.20, 1.10, 1.00):
    kcp = n_eff * k0
    up = kcp / kc
    cut = float(np.sum(flux[np.abs(kin) >= kcp])) / tot
    dloss = loss0_inplane * cut
    print(f"   n_eff={n_eff:.2f} (kc'/kc={up:.3f}): cuts {cut*100:5.1f}% of lateral "
          f"flux  ->  dLoss up to +{dloss:.4f}  (loss 0.0545 -> ~{0.0545-dloss:.4f})")

print(f"""
READING:
 * If the flux is strongly edge-piled (>50% beyond |kx|/kc=0.90), a modest
   anisotropy (n_eff ~ 1.30-1.35, a routine SWG form-birefringence) already cuts
   a large share -> anisotropic cladding is a REAL lever, ceiling comparable to or
   above the air trench's measured -0.012.
 * The air trench (n_eff -> 1.0 locally, but only at two discrete y-walls, not
   surrounding) measured dLoss ~ +0.012. A surrounding n_eff~1.3 layer should meet
   or beat that IF the vertical channel (the other ~42%) is left alone.
 * MECHANISM CLASS: identical to the trench (light-cone shrink / TIR). This is the
   refined form of 'cladding engineering', not a new mechanism. Verdict feeds the
   honest bottom line: works, but does not answer 'novel physics beyond cladding'.
""")
