# postarray_ceiling.py (study 2026-10-23, doc docs/postarray_radiation_cancellation_2026-10-23.md): Born ceiling for ANY set of mirrored SiN posts on candidate sites: min ||C + F a||^2, a >= 0 (a = (r/63 nm)^2).
# Streams over far-field directions (RAM ~3 GB free). Sites: rows Y_ROWS, x in [-X_MAX, X_MAX] step DX.
import os, sys, numpy as np
from scipy.optimize import nnls, lsq_linear
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postarray_born as dp
S = dp.WORK
Y_ROWS = [float(v) * 1e-6 for v in os.environ.get("ROWS_UM", "1.3,1.6,1.9").split(",")]; TAG = os.environ.get("TAG", ""); X_MAX, DX, R_REF = float(os.environ.get("X_MAX_UM", "24")) * 1e-6, 50e-9, 63e-9
C, dW = dp.control_C(); cc = np.sum(np.abs(C)**2 * dW)
xs_site = np.arange(-X_MAX, X_MAX + DX / 2, DX)
sites = [(x, y) for y in Y_ROWS for x in xs_site]; n = len(sites)
pos, pvec = [], []
for x, y in sites:
    Ed = dp.drive(y); e = np.array([np.interp(x, dp.xs, Ed[:, c].real) + 1j * np.interp(x, dp.xs, Ed[:, c].imag) for c in range(3)])
    for yy, ee in ((y, e), (-y, dp.s_y * e * np.array([1, -1, 1]))):
        pos.append([x, yy, 0.0]); pvec.append(dp.k**2 / (4 * np.pi) * np.pi * R_REF**2 * dp.H_POST * dp.GAM * ee)
pos = np.array(pos); pvec = np.array(pvec)                          # (2n, 3)
Cf = np.concatenate([C[0].ravel(), C[1].ravel()]); wf = np.concatenate([dW[0].ravel(), dW[1].ravel()])
nd = dp.nhat.shape[0]; G = np.zeros((n, n)); b = np.zeros(n); CH = 3000
for lo in range(0, nd, CH):
    sl = slice(lo, min(lo + CH, nd)); nh = dp.nhat[sl]
    E = np.exp(-1j * dp.k * (nh @ pos.T)).astype(np.complex64)     # (ch, 2n)
    Fth = (E * (dp.th[sl] @ pvec.T)); Fph = (E * (dp.ph[sl] @ pvec.T))
    Fth = (Fth[:, 0::2] + Fth[:, 1::2]) * dp.NORM; Fph = (Fph[:, 0::2] + Fph[:, 1::2]) * dp.NORM    # mirror pairs
    for F, c0, wv in ((Fth, Cf[:nd][sl], wf[:nd][sl]), (Fph, Cf[nd:][sl], wf[nd:][sl])):
        Fw = F * np.sqrt(wv)[:, None]
        G += np.real(Fw.conj().T @ Fw); b += np.real(Fw.conj().T @ (c0 * np.sqrt(wv)))
np.savez(os.path.join(S, f"ceiling_G{TAG}.npz"), G=G, b=b, cc=cc, xs=xs_site, ys=np.array(Y_ROWS))
R = np.linalg.cholesky(G + 1e-12 * np.trace(G) / n * np.eye(n)).T; t = -np.linalg.solve(R.T, b)
const = cc - t @ t
for tag, amax in (("a >= 0 (any size)", None),):
    a = nnls(R, t, maxiter=20 * n)[0] if amax is None else lsq_linear(R, t, bounds=(0, amax)).x
    P = (const + np.sum((R @ a - t)**2)) / cc
    nz = a > 1e-3
    print(f"{tag:30s} P/Pc {P:.3f} (removed {1-P:.3f}) | sites used {nz.sum()} | sum a {a.sum():.1f} | per row {[int(nz[i*len(xs_site):(i+1)*len(xs_site)].sum()) for i in range(len(Y_ROWS))]}")
    np.save(os.path.join(S, f"ceiling_a{TAG}_{'inf' if amax is None else amax}.npy"), a)
