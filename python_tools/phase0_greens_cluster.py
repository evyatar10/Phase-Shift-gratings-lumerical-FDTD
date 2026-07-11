"""
PHASE-0 refinement (Green's function, user-requested 2026-07-07): batch-1 placed
passive scatterers at the OPTIMAL-COMPLEX-alpha map sites and they ADDED loss.
A passive SiN cylinder has alpha ≈ REAL positive (below its Mie resonance), not
the complex optimum. So: were they just at the WRONG sites for a real alpha?

This recomputes the placement map for FIXED-SIGN REAL alpha (the only thing a
passive low-contrast cylinder provides) and finds where a real-alpha secondary
source actually CANCELS the residual in-cone radiation:

  residual(r_s) = min_{alpha real>0} |A0 + alpha*S(r_s)|^2 / |A0|^2
                = 1 - max(0, Re<A0,S>)^2 / (|A0|^2 |S|^2)   [flux-weighted]
  S(r_s) = propagated field of a unit source at r_s on the cladding line.

Reports: the best real-alpha SINGLE site (and its cancelled %), the worst site,
and a greedy real-alpha 2-source combo — the concrete positions phase-2 tests in
FDTD. HONESTY: this in-plane overlap model ignores (i) vertical re-radiation and
(ii) guided-mode back-reflection — the parasitics that made batch-1's singles
net-negative. So a positive here is necessary, not sufficient; the FDTD row is
the judge. If even the real-alpha-optimal site adds loss in FDTD, the passive
external-scatterer route is dead (parasitics dominate), full stop.

Run:  python python_tools/phase0_greens_cluster.py
"""

import os
import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                   "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")
N_CLAD = 1.444
Y_LINE = 2.8
NX = 1 << 12

m = loadmat(MAT, squeeze_me=True)
x = np.asarray(m["x"], float) * 1e6
y = np.asarray(m["y"], float) * 1e6
E = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])
lam = float(m["lam_used_nm"]) * 1e-3
kc = 2 * np.pi * N_CLAD / lam
xu = np.linspace(x.min(), x.max(), NX)
dx = xu[1] - xu[0]
k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, dx))
F = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dx
win = np.hanning(NX)
IN = np.abs(k) < kc
ky = np.sqrt(np.maximum(kc**2 - k[IN]**2, 0.0))
w = ky / kc
kyc = np.maximum(ky, 0.05 * kc)


def row(y0):
    j = int(np.argmin(np.abs(y - y0)))
    r = E[:, j]
    return np.interp(xu, x, r.real) + 1j * np.interp(xu, x, r.imag), y[j]


Ep, yline = row(+Y_LINE)
A0 = F(win * Ep)[IN]
P0 = float(np.sum(w * np.abs(A0)**2))

# field grid for source strength E_mode(x0,y0)
y0s = np.arange(0.65, 2.45 + 1e-9, 0.075)
x0s = np.arange(-11.5, 11.5 + 1e-9, 0.125)
Egrid = np.array([np.interp(y0s, y, E[i, :].real) for i in range(len(x))]) \
      + 1j * np.array([np.interp(y0s, y, E[i, :].imag) for i in range(len(x))])
kin = k[IN]


def Svec(x0, y0, iy):
    prop = np.exp(1j * kyc * (yline - y0)) / kyc
    Ex = np.interp([x0], x, Egrid[:, iy].real)[0] + 1j * np.interp([x0], x, Egrid[:, iy].imag)[0]
    return Ex * np.exp(-1j * kin * x0) * prop


# real-alpha residual map
res_real = np.ones((len(x0s), len(y0s)))
res_cplx = np.ones((len(x0s), len(y0s)))
best_alpha = np.zeros((len(x0s), len(y0s)))
for iy, y0 in enumerate(y0s):
    for ix, x0 in enumerate(x0s):
        S = Svec(x0, y0, iy)
        num = np.sum(w * np.conj(A0) * S)
        den = float(np.sum(w * np.abs(S)**2))
        if den <= 0:
            continue
        # complex-alpha ceiling
        res_cplx[ix, iy] = 1.0 - abs(num)**2 / (den * P0)
        # real-alpha: alpha = -Re(num)/den, valid only if that has the right sign
        a = -np.real(num) / den
        res_real[ix, iy] = float(np.sum(w * np.abs(A0 + a * S)**2)) / P0
        best_alpha[ix, iy] = a

ir, jr = np.unravel_index(np.argmin(res_real), res_real.shape)
iw, jw = np.unravel_index(np.argmax(res_real), res_real.shape)
print("REAL-alpha single-scatterer map (passive cylinder, sign-correct only):")
print(f"  BEST site: x0={x0s[ir]:+.2f} y0={y0s[jr]:.2f} um -> real-a residual "
      f"{res_real[ir,jr]*100:.1f}% (cancels {100-res_real[ir,jr]*100:.1f}%), "
      f"alpha sign {'+' if best_alpha[ir,jr]>0 else '-'}")
print(f"        complex ceiling there: cancels {100-res_cplx[ir,jr]*100:.1f}%")
print(f"  WORST site: x0={x0s[iw]:+.2f} y0={y0s[jw]:.2f} -> adds "
      f"{(res_real[iw,jw]-1)*100:.1f}% (this is the phase-wrong regime batch-1 hit)")

# where batch-1 placed them (x=0.38, 0.62, 0.81) — check real-alpha residual
print("\n  batch-1 sites (real-alpha residual; >100% = ADDS loss even at best real a):")
for xb in (0.38, 0.62, 0.81):
    ib = int(np.argmin(np.abs(x0s - xb)))
    jbest = int(np.argmin(res_real[ib, :]))
    print(f"    x0={xb}: best-over-y real-a residual {res_real[ib,jbest]*100:.1f}% "
          f"@ y0={y0s[jbest]:.2f}")

# top-8 real-alpha sites (spaced), sign-consistent (alpha>0 = SiN post below resonance)
print("\n  TOP real-alpha sites with alpha>0 (SiN post) — candidates to TEST:")
order = np.argsort(res_real, axis=None)
picked = []
for idx in order:
    i, j = np.unravel_index(idx, res_real.shape)
    if best_alpha[i, j] <= 0:      # alpha<0 needs an oxide hole (index<clad) — note separately
        continue
    if any(abs(x0s[i]-a) < 0.6 and abs(y0s[j]-b) < 0.25 for a, b in picked):
        continue
    picked.append((x0s[i], y0s[j]))
    print(f"    x0={x0s[i]:+.2f} y0={y0s[j]:.2f}  cancels {100-res_real[i,j]*100:.1f}%")
    if len(picked) >= 8:
        break

# greedy real-alpha 2-source (independent real strengths)
i0, j0 = ir, jr
S1 = Svec(x0s[i0], y0s[j0], j0)
best2 = (res_real[ir, jr], None)
for jy in range(0, len(y0s), 2):
    for ix in range(0, len(x0s), 2):
        S2 = Svec(x0s[ix], y0s[jy], jy)
        M = np.array([S1, S2]).T
        G = (M.conj().T * w) @ M
        b = (M.conj().T * w) @ A0
        try:
            al = np.linalg.solve(G, -b)
        except np.linalg.LinAlgError:
            continue
        if np.any(np.real(al) <= 0) or np.max(np.abs(np.imag(al))) > 0.4 * np.max(np.abs(np.real(al))):
            continue    # require both real-positive (passive posts)
        r = float(np.sum(w * np.abs(A0 + M @ al)**2)) / P0
        if r < best2[0]:
            best2 = (r, (x0s[ix], y0s[jy]))
if best2[1]:
    print(f"\n  real-alpha 2-post combo: site1 {x0s[i0]:+.2f}/{y0s[j0]:.2f} + "
          f"site2 {best2[1][0]:+.2f}/{best2[1][1]:.2f} -> cancels {100-best2[0]*100:.1f}%")

print(f"""
CEILING ARITHMETIC (stack loss 0.0545; in-plane share 0.58, x-window 0.77):
  best real-a single post: max dT ~ +{0.0545*0.58*0.77*max(0,1-res_real[ir,jr]):.4f}
  (IF the FDTD parasitics don't eat it — batch-1 says they well might; the TEST decides)
VERDICT INPUT for phase-2: test the best real-alpha site(s) above. If a
sign-correct SiN post at the real-alpha-optimal site STILL adds loss in FDTD,
the passive external-scatterer route is closed by parasitics, not by siting.
""")
