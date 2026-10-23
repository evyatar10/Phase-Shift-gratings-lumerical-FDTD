# postarray_design.py (study 2026-10-23, doc docs/postarray_radiation_cancellation_2026-10-23.md): discrete post-array design on the Born Gram system (ceiling_G{TAG}.npz).
# Posts: free x (continuous, linear-hat interpolation onto the 50 nm site grid), amplitude a = (r/63 nm)^2 in [0, A_MAX]; rows fixed.
# Model response scaled by SC (= 0.70, measured dy29 decomposition). Constraints: same-row gap >= GAP between post edges (penalty, then checked).
import os, sys, numpy as np
from scipy.optimize import minimize
S = os.environ.get("PA_WORK", r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/comb_overshoot/postarray_work"); TAG = os.environ.get("TAG", "_r4")
SC, R_REF, R_MAX, R_MIN, GAP = 0.70, 63.0, float(os.environ.get("R_MAX", "100")), 50.0, 100.0
d = np.load(os.path.join(S, f"ceiling_G{TAG}.npz")); G, b, cc, xs, ys = d["G"], d["b"], float(d["cc"]), d["xs"] * 1e9, d["ys"] * 1e9
nx = len(xs); DX = xs[1] - xs[0]; A_MAX = (R_MAX / R_REF) ** 2; A_MIN = (R_MIN / R_REF) ** 2
a0 = np.load(os.path.join(S, f"ceiling_a{TAG}_inf.npy"))
rows_allowed = [int(v) for v in os.environ.get("ROW_IDX", ",".join(str(i) for i in range(len(ys)))).split(",")]
if len(rows_allowed) < len(ys):   # re-seed from the NNLS optimum of the allowed rows only (the full-row seed is a different basin)
    from scipy.optimize import nnls
    sel = np.concatenate([np.arange(i * nx, (i + 1) * nx) for i in rows_allowed]); Gs, bs = G[np.ix_(sel, sel)], b[sel]
    Rc = np.linalg.cholesky(Gs + 1e-12 * np.trace(Gs) / len(sel) * np.eye(len(sel))).T; t = -np.linalg.solve(Rc.T, bs)
    a0 = np.zeros(len(ys) * nx); a0[sel] = nnls(Rc, t, maxiter=20 * len(sel))[0]
# init: one post per contiguous NNLS cluster, at its centroid, amplitude clipped
px, prow, pa = [], [], []
for i in rows_allowed:
    ai = a0[i * nx:(i + 1) * nx]; k = np.flatnonzero(ai > 1e-3)
    if not k.size: continue
    groups = np.split(k, np.flatnonzero(np.diff(k) > 1) + 1)
    for gk in groups:
        px.append(np.sum(xs[gk] * ai[gk]) / ai[gk].sum()); prow.append(i); pa.append(min(ai[gk].sum(), A_MAX))
px, prow, pa = np.array(px), np.array(prow), np.array(pa); N = len(px)
def grid_amp(x, a):
    u = (x - xs[0]) / DX; k = np.clip(np.floor(u).astype(int), 0, nx - 2); f = u - k
    A = np.zeros(len(ys) * nx); idx = prow * nx + k
    np.add.at(A, idx, a * (1 - f)); np.add.at(A, idx + 1, a * f); return A, idx, f
def radius(a): return R_REF * np.sqrt(np.maximum(a, 0))
def obj(v, lam):
    x, a = v[:N], v[N:]
    A, idx, f = grid_amp(x, a); gA = 2 * SC * b + 2 * SC**2 * (G @ A)
    P = (cc + 2 * SC * b @ A + SC**2 * A @ G @ A) / cc
    ga = (gA[idx] * (1 - f) + gA[idx + 1] * f) / cc; gx = a * (gA[idx + 1] - gA[idx]) / DX / cc
    pen = 0.0; gpx = np.zeros(N); gpa = np.zeros(N)
    for i in set(prow):
        j = np.flatnonzero(prow == i); j = j[np.argsort(x[j])]
        for p, q in zip(j[:-1], j[1:]):
            need = radius(a[p]) + radius(a[q]) + GAP; viol = need - (x[q] - x[p])
            if viol > 0:
                pen += lam * viol**2; gpx[q] -= 2 * lam * viol; gpx[p] += 2 * lam * viol
                for s in (p, q): gpa[s] += 2 * lam * viol * R_REF / (2 * np.sqrt(max(a[s], 1e-9)))
    return P + pen, np.concatenate([gx + gpx, ga + gpa])
v = np.concatenate([px, pa]); bnds = [(xs[0], xs[-2])] * N + [(0, A_MAX)] * N
for lam in (1e-6, 1e-5, 1e-4, 1e-3):
    v = minimize(obj, v, args=(lam,), jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 3000}).x
for it in range(3):   # drop sub-minimum posts and re-optimise
    keep = v[N:] >= A_MIN; px, pa, prow = v[:N][keep], v[N:][keep], prow[keep]; N = len(px); v = np.concatenate([px, pa])
    bnds = [(xs[0], xs[-2])] * N + [(A_MIN, A_MAX)] * N
    v = minimize(obj, v, args=(1e-3,), jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": 3000}).x
x, a = v[:N], v[N:]; A, _, _ = grid_amp(x, a); P = (cc + 2 * SC * b @ A + SC**2 * A @ G @ A) / cc
viol = 0
for i in set(prow):
    j = np.flatnonzero(prow == i); j = j[np.argsort(x[j])]
    viol += sum((radius(a[p]) + radius(a[q]) + GAP) - (x[q] - x[p]) > 1.0 for p, q in zip(j[:-1], j[1:]))
print(f"{TAG} rows {[int(ys[i]) for i in sorted(set(prow))]} R_MAX {R_MAX:.0f}: posts {N} (per row {[int((prow==i).sum()) for i in sorted(set(prow))]}), "
      f"P/Pc {P:.3f} (removed {1-P:.3f}), spacing violations {viol}, r {radius(a).min():.0f}-{radius(a).max():.0f} nm, x {x.min()/1e3:.1f}..{x.max()/1e3:.1f} um")
np.savez(os.path.join(S, f"design{TAG}_R{int(R_MAX)}_rows{''.join(str(i) for i in rows_allowed)}.npz"), x=x, a=a, r=radius(a), y=ys[prow], P=P)
