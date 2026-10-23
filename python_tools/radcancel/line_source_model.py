# scratch 2026-10-20: does a LINE SOURCE on the waveguide axis explain the measured far field?
# Model: E(u) = sum_p  p_hat_tangential(u) * J_p(k u_x), p in {y} or {x,y,z}; J_p fitted per u_x ring (least squares).
# Explained fraction = ||E_model||^2/||E||^2 (residual orthogonal). Then the multipole table of E_model vs measured.
import sys, numpy as np, scipy.io as sio
sys.path.insert(0, r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/python_tools")
from farfield_surface import sphere_field
from farfield_multipole import multipoles
LMAX = 150
m = sio.loadmat(sys.argv[1], squeeze_me=True, struct_as_record=False)
x, w, phi, E_th, E_ph, info = sphere_field(m, LMAX)
cth = x[:, None] * np.ones_like(phi)[None, :]; sth = np.sqrt(1 - cth**2)
cph = np.cos(phi)[None, :] * np.ones_like(x)[:, None]; sph = np.sin(phi)[None, :] * np.ones_like(x)[:, None]
ux = sth * cph
dW = (w[:, None] * (2 * np.pi / len(phi))) * np.ones_like(cth)
# tangential components of the unit polarization vectors: (theta_hat . e, phi_hat . e)
P = {"x": (cth * cph, -sph), "y": (cth * sph, cph), "z": (-sth, np.zeros_like(sth))}
E = np.stack([E_th, E_ph])
Ptot = np.sum(np.abs(E)**2 * dW)
nb = 100; edges = np.linspace(-1, 1, nb + 1); idx = np.clip(np.digitize(ux, edges) - 1, 0, nb - 1)
def fit(pols):
    Em = np.zeros_like(E)
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 4: continue
        A = np.stack([np.concatenate([P[p][0][sel], P[p][1][sel]]) for p in pols], 1)   # (2n, npol)
        y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
        coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)
        fitv = A @ coef; n = sel.sum()
        Em[0][sel] = fitv[:n]; Em[1][sel] = fitv[n:]
    return Em
def table(Ef, label):
    aE, aM = multipoles(x, w, phi, Ef[0], Ef[1], LMAX)
    Pw = np.abs(aE)**2 + np.abs(aM)**2; tot = Pw.sum(); Pl = Pw.sum(1); ls = np.arange(LMAX + 1)
    order = np.argsort(Pw.ravel())[::-1][:6]
    tops = []
    for k in order:
        l, mi = divmod(k, 2 * LMAX + 1); mm = mi - LMAX
        t = "E" if abs(aE[l, mi]) >= abs(aM[l, mi]) else "M"
        tops.append(f"{t}({l},{mm:+d}) {100*Pw[l,mi]/tot:.1f}")
    print(f"{label:14s} <l> {np.sum(ls*Pl)/tot:5.1f}  l<=5 {100*Pl[:6].sum()/tot:5.1f}%  l<=10 {100*Pl[:11].sum()/tot:5.1f}%  E {100*(np.abs(aE)**2).sum()/tot:4.1f}%  top: " + ", ".join(tops))
    return aE, aM
print("file:", sys.argv[1].split("/")[-1][:60])
aE0, aM0 = table(E, "measured")
for pols in (["y"], ["x", "y", "z"]):
    Em = fit(pols)
    frac = np.sum(np.abs(Em)**2 * dW) / Ptot
    print(f"line source pol {''.join(pols)}: explained power fraction {frac:.3f}")
    aE1, aM1 = table(Em, f"model {''.join(pols)}")
    # complex overlap of the two multipole vectors (phase-sensitive agreement of the harmonic amplitudes)
    v0 = np.concatenate([aE0.ravel(), aM0.ravel()]); v1 = np.concatenate([aE1.ravel(), aM1.ravel()])
    print(f"   |<model,measured>|^2/(|m|^2|M|^2) = {abs(np.vdot(v1, v0))**2/(np.vdot(v1,v1).real*np.vdot(v0,v0).real):.3f}")

# ---- transverse FORM FACTOR models: the source is not a point in (y,z) ----
# sidewall pair at y=+-W/2 (y-even): cos(k u_y W/2); top/bottom pair at z=+-h/2: cos(k u_z h/2); slab-sinc for the other dim
k = 2 * np.pi * 1.444 / (float(np.ravel(m["surface_top"].lam)[0]))
uy = sth * sph; uz = cth
def sinc(a): return np.where(np.abs(a) < 1e-9, 1.0, np.sin(a) / np.where(np.abs(a) < 1e-9, 1, a))
def fit_ff(ff):
    Em = np.zeros_like(E); A0 = P["y"]
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 4: continue
        A = np.concatenate([A0[0][sel] * ff[sel], A0[1][sel] * ff[sel]])[:, None]
        y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
        coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)
        n = sel.sum(); fitv = (A @ coef); Em[0][sel] = fitv[:n]; Em[1][sel] = fitv[n:]
    return np.sum(np.abs(Em)**2 * dW) / Ptot, Em
print("form-factor scan (explained fraction):")
rows = []
for Wum in (0.6, 0.8, 1.0, 1.2, 1.4):
    for hum in (0.0, 0.35):
        ff = np.cos(k * uy * Wum * 1e-6 / 2) * (sinc(k * uz * hum * 1e-6 / 2) if hum else 1.0)
        fr, _ = fit_ff(ff); rows.append((fr, f"sidewalls W={Wum} um, slab h={hum}"))
for hum in (0.35, 0.5):
    ff = np.cos(k * uz * hum * 1e-6 / 2) * sinc(k * uy * 1.0e-6 / 2)
    fr, _ = fit_ff(ff); rows.append((fr, f"top/bottom h={hum} um, slab W=1.0"))
for fr, lab in sorted(rows, reverse=True)[:6]: print(f"  {fr:.3f}  {lab}")
# free azimuthal fit: per u_x ring, fit E as p_y * sum_n a_n cos(n*psi) with psi the angle around the x axis (even in y,z) -> ceiling
psi = np.arctan2(uz, uy)
def fit_free(nmax):
    Em = np.zeros_like(E); A0 = P["y"]
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 4: continue
        cols = [np.concatenate([A0[0][sel] * np.cos(2*n*psi[sel]), A0[1][sel] * np.cos(2*n*psi[sel])]) for n in range(nmax + 1)]
        A = np.stack(cols, 1); y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
        coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)
        n = sel.sum(); fitv = A @ coef; Em[0][sel] = fitv[:n]; Em[1][sel] = fitv[n:]
    return np.sum(np.abs(Em)**2 * dW) / Ptot
print("free even-azimuth fit (p_y x sum cos(2n psi)): n<=0 %.3f  n<=1 %.3f  n<=2 %.3f  n<=4 %.3f" % tuple(fit_free(n) for n in (0, 1, 2, 4)))
# full TE symmetry class per ring: p_y*cos(2n psi) [P_y even/even] + p_x*cos((2n+1)psi) [P_x odd-y even-z] + p_z*sin((2n+1)psi) [P_z even-y odd-z]
def fit_class(nmax, use=("y", "x", "z")):
    Em = np.zeros_like(E)
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 4: continue
        cols = []
        for n in range(nmax + 1):
            if "y" in use: cols.append(np.concatenate([P["y"][0][sel] * np.cos(2*n*psi[sel]), P["y"][1][sel] * np.cos(2*n*psi[sel])]))
            if "x" in use: cols.append(np.concatenate([P["x"][0][sel] * np.cos((2*n+1)*psi[sel]), P["x"][1][sel] * np.cos((2*n+1)*psi[sel])]))
            if "z" in use: cols.append(np.concatenate([P["z"][0][sel] * np.sin((2*n+1)*psi[sel]), P["z"][1][sel] * np.sin((2*n+1)*psi[sel])]))
        A = np.stack(cols, 1); y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
        coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)
        n_ = sel.sum(); fitv = A @ coef; Em[0][sel] = fitv[:n_]; Em[1][sel] = fitv[n_:]
    return np.sum(np.abs(Em)**2 * dW) / Ptot
print("TE-class fit: y only n<=2 %.3f | y+x n<=2 %.3f | y+z n<=2 %.3f | y+x+z n<=1 %.3f | n<=2 %.3f | n<=3 %.3f" % (
    fit_class(2, ("y",)), fit_class(2, ("y", "x")), fit_class(2, ("y", "z")), fit_class(1), fit_class(2), fit_class(3)))
# odd-class check (should be ~0 if the parity is clean): p_y*sin((2n+1)psi)-type terms
def fit_odd(nmax):
    Em = np.zeros_like(E)
    for b in range(nb):
        sel = idx == b
        if sel.sum() < 4: continue
        cols = [np.concatenate([P["y"][0][sel] * np.sin((2*n+1)*psi[sel]), P["y"][1][sel] * np.sin((2*n+1)*psi[sel])]) for n in range(nmax + 1)]
        A = np.stack(cols, 1); y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
        coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None)
        n_ = sel.sum(); fitv = A @ coef; Em[0][sel] = fitv[:n_]; Em[1][sel] = fitv[n_:]
    return np.sum(np.abs(Em)**2 * dW) / Ptot
print("wrong-parity class (p_y * sin((2n+1)psi)), n<=2: %.3f" % fit_odd(2))
# ---- WHERE along x does the leak come from: band-limited inverse FT of the per-ring source coefficients (TE class, n=0 terms) ----
nb2 = 300; edges2 = np.linspace(-1, 1, nb2 + 1); idx2 = np.clip(np.digitize(ux, edges2) - 1, 0, nb2 - 1); uc = 0.5 * (edges2[1:] + edges2[:-1])
Jy = np.zeros(nb2, complex); Jx = np.zeros(nb2, complex); Py = np.zeros(nb2)
for b in range(nb2):
    sel = idx2 == b
    if sel.sum() < 3: continue
    cols = [np.concatenate([P["y"][0][sel], P["y"][1][sel]]), np.concatenate([P["x"][0][sel] * np.cos(psi[sel]), P["x"][1][sel] * np.cos(psi[sel])]),
            np.concatenate([P["y"][0][sel] * np.cos(2*psi[sel]), P["y"][1][sel] * np.cos(2*psi[sel])])]
    A = np.stack(cols, 1); y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
    coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None); Jy[b], Jx[b] = coef[0], coef[1]; Py[b] = np.sum(np.abs(E[:, sel])**2 * dW[sel])
xs = np.linspace(-45, 45, 901)
for lab, J in (("P_y", Jy), ("P_x", Jx)):
    S = np.array([np.sum(J * np.exp(1j * k * uc * 1e-6 * xx) ) for xx in xs]); S2 = np.abs(S)**2; S2 /= S2.sum()
    cum = np.cumsum(S2); 
    def within(a): return 100 * S2[np.abs(xs) <= a].sum()
    peaks = xs[np.argsort(S2)[::-1][:12]]
    print(f"{lab}: source power within |x|<=5: {within(5):.0f}%  <=10: {within(10):.0f}%  <=20: {within(20):.0f}%  <=30: {within(30):.0f}%  <=42: {within(42):.0f}% ; strongest x (um): " + " ".join(f"{p:+.1f}" for p in sorted(peaks)))
# ---- localization with the FULL TE class (GPT review: p_y-only term carries 19-29 % of the power). Columns per ring:
# p_y cos(2n psi) n=0..2, p_x cos((2n+1) psi) n=0..1  (p_z class is redundant with these). Source image per column, power-weighted sum.
cols_def = [("y", 0), ("y", 1), ("y", 2), ("x", 0), ("x", 1)]
J = np.zeros((len(cols_def), nb2), complex); Pexp = 0.0
for b in range(nb2):
    sel = idx2 == b
    if sel.sum() < 6: continue
    cols = []
    for p, n in cols_def:
        f = np.cos(2 * n * psi[sel]) if p == "y" else np.cos((2 * n + 1) * psi[sel])
        cols.append(np.concatenate([P[p][0][sel] * f, P[p][1][sel] * f]))
    A = np.stack(cols, 1); y = np.concatenate([E[0][sel], E[1][sel]]); ww = np.sqrt(np.concatenate([dW[sel], dW[sel]]))
    coef, *_ = np.linalg.lstsq(A * ww[:, None], y * ww, rcond=None); J[:, b] = coef
    fitv = A @ coef; Pexp += np.sum(np.abs(fitv) ** 2 * np.concatenate([dW[sel], dW[sel]]))
print(f"full-class ring fit explains {Pexp/Ptot:.3f} of the power")
S2tot = np.zeros_like(xs); 
for kcol, (p, n) in enumerate(cols_def):
    # weight each column's image by that column's share of the explained power (ring-integrated |J|^2 x pattern norm ~ const)
    S = np.array([np.sum(J[kcol] * np.exp(1j * k * uc * 1e-6 * xx)) for xx in xs]); S2 = np.abs(S) ** 2
    wcol = np.sum(np.abs(J[kcol]) ** 2); S2tot += S2 / S2.sum() * wcol
S2tot /= S2tot.sum()
def within(a): return 100 * S2tot[np.abs(xs) <= a].sum()
print(f"ALL-CLASS source image: within |x|<=2: {within(2):.0f}%  <=5: {within(5):.0f}%  <=10: {within(10):.0f}%  <=20: {within(20):.0f}%  <=30: {within(30):.0f}%; "
      f"strongest x: " + " ".join(f"{v:+.1f}" for v in sorted(xs[np.argsort(S2tot)[::-1][:10]])))
