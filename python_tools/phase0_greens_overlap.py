"""
PHASE-0 (4.4) — Green's-function anti-radiation overlap analysis. Zero GPU.

Organizing principle for the BIC (4.1) and Kerker (4.2) routes: radiated power
= overlap of the effective source with the radiation-continuum Green's
function; loss -> 0 iff that overlap -> 0 (a photonic dark state). This script
measures, from the EXISTING accurate-mesh 2D field export of the stack
(tm_field_export, job 118462), everything the two routes need to aim:

  1. Residual in-cone radiating amplitude A0(kx) on cladding lines y=+-2.8 um
     (power-metric weighted by ky/kc = flux through the side plane), with a
     guided-peak leakage audit (the +-12 um window truncates the guided mode;
     its Hann sidelobe at kx=beta must stay well below the in-cone signal or
     the numbers are window artifacts).
  2. SYMMETRY table inputs for the BIC route: in-cone weight of the y-odd and
     x-odd parts of the radiating field. If the radiating continuum coupling
     is ~pure-even, only an odd operating mode is symmetry-protected — and an
     odd mode also decouples from the (even) input port, which is the honest
     killer to quantify, not assume.
  3. Single secondary-source (scatterer) placement map: for a monopole-like
     scatterer at (x0, y0) driven by the local resonant field E(x0,y0), the
     best achievable in-cone power residual after optimizing its complex
     strength alpha:  residual = 1 - |<A0, S>|^2 / (|A0|^2 |S|^2)  in the
     flux metric. This is the HARD CEILING for one scatterer pair placed
     anywhere in the exported window — directionality (Kerker) can approach
     it, never beat it. Reports the best sites, the fringe spacing (should
     match the lambda/(2 n_clad) phase arcs), and the value at the previously
     validated x=0.81 um site.

Assumptions (stated): symmetric +-y pair handles the -y side by mirror
symmetry (checked in 2.); scatterer re-radiation into the vertical channel is
NOT modeled (in-plane-only ceiling, consistent with the measured 62% in-plane
split); scatterer treated as point-like (fine for placement/ceiling; the
Kerker pattern only narrows the achievable subset).

Run:  python python_tools/phase0_greens_overlap.py
"""

import os

import numpy as np
from scipy.io import loadmat

ROOT = r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes"
MAT_STACK = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                         "result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox6p8_Zbox8p8_EZSLICE.mat")
MAT_W800 = os.path.join(ROOT, "results_from_athena", "tm_field_export", "results",
                        "result_N80_TM_avg_Ybox6p8_Zbox8p8_EZSLICE.mat")

N_CLAD = 1.444
PITCH = 0.51683           # um
Y_LINE = 2.8              # um, cladding sampling line
NX = 1 << 12


def load(mat):
    m = loadmat(mat, squeeze_me=True)
    x = np.asarray(m["x"], float) * 1e6
    y = np.asarray(m["y"], float) * 1e6
    E = np.asarray(m["Ez_re"]) + 1j * np.asarray(m["Ez_im"])   # (x, y)
    lam = float(m["lam_used_nm"]) * 1e-3                        # um
    return x, y, E, lam


def analyze(name, mat):
    x, y, E, lam = load(mat)
    kc = 2 * np.pi * N_CLAD / lam
    beta = np.pi / PITCH
    xu = np.linspace(x.min(), x.max(), NX)
    dx = xu[1] - xu[0]
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, dx))
    F = lambda f: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f))) * dx
    win = np.hanning(NX)

    def row(y0):
        j = int(np.argmin(np.abs(y - y0)))
        r = E[:, j]
        return (np.interp(xu, x, r.real) + 1j * np.interp(xu, x, r.imag)), y[j]

    Ep, yp = row(+Y_LINE)
    Em, ym = row(-Y_LINE)
    Ap, Am = F(win * Ep), F(win * Em)
    IN = np.abs(k) < kc
    ky = np.sqrt(np.maximum(kc**2 - k[IN]**2, 0.0))
    w = ky / kc                                   # flux metric weight
    Pp = float(np.sum(w * np.abs(Ap[IN])**2))
    Pm = float(np.sum(w * np.abs(Am[IN])**2))

    # guided-peak leakage audit: window sidelobe level at the in-cone edge
    guided = np.abs(k - beta) < 0.4
    pk_guided = np.abs(Ap[guided]).max()
    pk_incone = np.abs(Ap[IN]).max()
    # Hann sidelobe of the guided peak evaluated at distance (beta - kc)
    Lw = xu[-1] - xu[0]
    u = (beta - kc) * Lw / (2 * np.pi)            # bins from the peak
    sidelobe = pk_guided * max(1e-12, np.abs(np.sinc(u) / (1 - u**2))) if u > 1.5 else pk_guided

    print(f"\n--- {name}: lam {lam*1e3:.2f} nm, kc {kc:.3f}, beta {beta:.3f} rad/um")
    print(f"  in-cone flux-weighted power: +y {Pp:.4g}  -y {Pm:.4g}  (ratio {Pm/Pp:.2f})")
    ux = k[IN] / kc
    prof = w * np.abs(Ap[IN])**2
    print(f"  E^2-weighted mean |ux| of in-cone amplitude: "
          f"{float(np.sum(prof*np.abs(ux))/np.sum(prof)):.2f} "
          f"(polarimetry: 0.77 for rect-1050 family)")
    print(f"  guided peak |A|: {pk_guided:.3g}; in-cone peak |A|: {pk_incone:.3g}; "
          f"Hann-sidelobe of guided peak at cone edge ~{sidelobe:.2g} "
          f"({'OK, well below in-cone signal' if sidelobe < 0.2*pk_incone else 'WARNING: window leakage comparable to signal'})")

    # ---- symmetry decomposition (feeds 4.1a) --------------------------------
    # y-parity: even/odd combinations of the two lines
    A_even, A_odd = (Ap + Am) / 2, (Ap - Am) / 2
    Pe = float(np.sum(w * np.abs(A_even[IN])**2))
    Po = float(np.sum(w * np.abs(A_odd[IN])**2))
    print(f"  y-parity of in-cone radiation: even {Pe/(Pe+Po)*100:.1f}%  "
          f"odd {Po/(Pe+Po)*100:.1f}%")
    # x-parity about the defect: E(x) vs E(-x) on the +y line
    Ex_e = (Ep + Ep[::-1]) / 2
    Ex_o = (Ep - Ep[::-1]) / 2
    Axe, Axo = F(win * Ex_e), F(win * Ex_o)
    Pxe = float(np.sum(w * np.abs(Axe[IN])**2))
    Pxo = float(np.sum(w * np.abs(Axo[IN])**2))
    print(f"  x-parity of in-cone radiation: even {Pxe/(Pxe+Pxo)*100:.1f}%  "
          f"odd {Pxo/(Pxe+Pxo)*100:.1f}%")
    return dict(x=x, y=y, E=E, lam=lam, kc=kc, xu=xu, F=F, win=win, k=k, IN=IN,
                ky=ky, w=w, Ap=Ap, yp=yp, P0=Pp)


res_stack = analyze("THE STACK (W1050+pair[+20,+20]+see-saw)", MAT_STACK)
res_w800 = analyze("W800 baseline", MAT_W800)

# ---------------------------------------------------------------- placement map
print("\n" + "=" * 74)
print("SINGLE-SCATTERER (pair) PLACEMENT MAP — in-cone cancellation ceiling")
print("=" * 74)
d = res_stack
kin, ky, w, IN = d["k"][d["IN"]], d["ky"], d["w"], d["IN"]
A0 = d["Ap"][IN]
P0 = float(np.sum(w * np.abs(A0)**2))
x, y, E = d["x"], d["y"], d["E"]
yline = d["yp"]

x0s = np.arange(-11.5, 11.5 + 1e-9, 0.125)
y0s = np.arange(0.65, 2.45 + 1e-9, 0.075)
kyc = np.maximum(ky, 0.05 * d["kc"])            # cap the 1/ky edge singularity

E_interp_re = [np.interp(y0s, y, E[i, :].real) for i in range(len(x))]
E_interp_im = [np.interp(y0s, y, E[i, :].imag) for i in range(len(x))]
Egrid = np.array(E_interp_re) + 1j * np.array(E_interp_im)   # (nx_data, ny0)

resid = np.ones((len(x0s), len(y0s)))
for iy, y0 in enumerate(y0s):
    prop = np.exp(1j * kyc * (yline - y0)) / kyc
    Erow = np.interp(x0s, x, Egrid[:, iy].real) + 1j * np.interp(x0s, x, Egrid[:, iy].imag)
    for ix, x0 in enumerate(x0s):
        S = Erow[ix] * np.exp(-1j * kin * x0) * prop
        num = np.abs(np.sum(w * np.conj(A0) * S))**2
        den = float(np.sum(w * np.abs(S)**2)) * P0
        resid[ix, iy] = 1.0 - num / max(den, 1e-300)

best = np.unravel_index(np.argmin(resid), resid.shape)
print(f"best single-pair site: x0 = {x0s[best[0]]:+.2f} um, y0 = {y0s[best[1]]:.2f} um"
      f" -> in-cone residual {resid[best]*100:.1f}% (cancels {100-resid[best]*100:.1f}%)")
order = np.argsort(resid, axis=None)
print("top-10 sites (x0, y0, cancelled %):")
seen = []
for idx in order:
    i, j = np.unravel_index(idx, resid.shape)
    if any(abs(x0s[i] - a) < 0.5 and abs(y0s[j] - b) < 0.3 for a, b in seen):
        continue
    seen.append((x0s[i], y0s[j]))
    print(f"   x0 {x0s[i]:+7.2f}  y0 {y0s[j]:5.2f}   {100*(1-resid[i,j]):5.1f}%")
    if len(seen) >= 10:
        break

# fringe spacing along x at fixed y0 (should be ~lam/(2 n_clad) if near-axial)
j_mid = int(np.argmin(np.abs(y0s - 1.2)))
line = 1 - resid[:, j_mid]
pk = [i for i in range(1, len(line) - 1) if line[i] > line[i-1] and line[i] > line[i+1]
      and line[i] > 0.5 * line.max()]
if len(pk) > 2:
    sp = np.diff(x0s[np.array(pk)])
    print(f"fringe spacing along x @ y0=1.2: median {np.median(sp):.3f} um "
          f"(lam/(2 n_clad) = {d['lam']/(2*N_CLAD):.3f} um; pitch = {PITCH:.3f} um)")

# the previously validated site x = 0.81 um: best y0 there
i81 = int(np.argmin(np.abs(x0s - 0.81)))
jb = int(np.argmin(resid[i81, :]))
print(f"legacy validated site x0=0.81: best y0 {y0s[jb]:.2f} -> cancels "
      f"{100*(1-resid[i81, jb]):.1f}% (vs global best {100-resid[best]*100:.1f}%)")

# two-source joint optimum: best site + best partner (greedy, exact 2x2 solve)
i0, j0 = best
prop0 = np.exp(1j * kyc * (yline - y0s[j0])) / kyc
E00 = np.interp([x0s[i0]], x, Egrid[:, j0].real)[0] + 1j * np.interp([x0s[i0]], x, Egrid[:, j0].imag)[0]
S1 = E00 * np.exp(-1j * kin * x0s[i0]) * prop0
best2 = (1.0, None)
for iy, y0 in enumerate(y0s[::3]):
    propn = np.exp(1j * kyc * (yline - y0)) / kyc
    Erow = np.interp(x0s, x, Egrid[:, iy*3].real) + 1j * np.interp(x0s, x, Egrid[:, iy*3].imag)
    for ix in range(0, len(x0s), 2):
        S2 = Erow[ix] * np.exp(-1j * kin * x0s[ix]) * propn
        M = np.array([S1, S2]).T
        G = (M.conj().T * w) @ M
        b = (M.conj().T * w) @ A0
        try:
            al = np.linalg.solve(G, -b)
        except np.linalg.LinAlgError:
            continue
        r = float(np.sum(w * np.abs(A0 + M @ al)**2)) / P0
        if r < best2[0]:
            best2 = (r, (x0s[ix], y0))
print(f"best PAIR-of-pairs (site1 fixed at global best): residual {best2[0]*100:.1f}% "
      f"(partner at x0 {best2[1][0]:+.2f}, y0 {best2[1][1]:.2f})")

# ---------------------------------------------------------------- ceiling math
print("\nCEILING ARITHMETIC (stack, loss 0.0545):")
in_plane = 0.58
addressable = 0.77
for canc, tag in ((1 - resid[best], "1 pair @ map best"), ((1 - best2[0]), "2 pairs")):
    dT = 0.0545 * in_plane * addressable * float(canc)
    print(f"  {tag:20s}: cancels {float(canc)*100:4.1f}% of in-window in-plane "
          f"-> max dT ~ +{dT:.4f}")
print("(in-plane share 0.58, x-window share 0.77; vertical channel untouched;")
print(" point-source ceiling — a Kerker pattern can only approach these numbers)")
