# postarray_place2d.py (study: runners/sweeps/farfield_sph_15um_postarray_overshoot; copied unchanged from the Fable 2-D agent scratch pa2d.py, 2026-10-23):
# Leak P(a) = (cc + 2 s b.A + s^2 A.G.A)/cc with A = T a (bilinear spreading of each post's amplitude onto the 4 neighbouring
# sites = interpolation of the response COLUMNS, so G_eff = T^T G T, b_eff = T^T b). a = (r/63 nm)^2.
# Constraints: r in [R_MIN, R_MAX]; all-pairs edge gap >= GAP (strict, repaired after every polish); tooth clearance y - r >= Y_EDGE;
# y within the Gram's row range. Mirror posts (-y) are already inside each Gram column.
import numpy as np, time
from scipy.optimize import minimize
R_REF, S = 63.0, 0.70
R_MIN, R_MAX = 50.0, 100.0
A_MIN, A_MAX = (R_MIN / R_REF) ** 2, (R_MAX / R_REF) ** 2
Y_EDGE = 1000.0          # nm: post edge must stay >= 100 nm outside the widest tooth (half-width 900 nm)
rad = lambda a: R_REF * np.sqrt(np.maximum(a, 0.0))
amp = lambda r: (np.asarray(r, float) / R_REF) ** 2


class Gram:
    def __init__(self, path, s=S):
        d = np.load(path)
        self.G, self.b, self.cc = d["G"], d["b"], float(d["cc"])   # float64 (770 MB); matvec via mv()
        self.xs, self.ys = d["xs"] * 1e9, d["ys"] * 1e9
        self.nx, self.ny = len(self.xs), len(self.ys); self.n = self.nx * self.ny
        self.dx = self.xs[1] - self.xs[0]; self.dy = (self.ys[1] - self.ys[0]) if self.ny > 1 else 1.0
        self.X = np.tile(self.xs, self.ny); self.Y = np.repeat(self.ys, self.nx)
        self.diag = np.diag(self.G).astype(float); self.s = s
    def mv(self, A): return self.G @ A
    def col(self, i): return self.G[:, i]

    # ---- bilinear spreading T(x, y): idx (N,4), weights (N,4), d/dx, d/dy ----
    def T(self, x, y):
        u = (x - self.xs[0]) / self.dx; i = np.clip(np.floor(u).astype(int), 0, self.nx - 2); f = u - i
        if self.ny > 1:
            v = (y - self.ys[0]) / self.dy; j = np.clip(np.floor(v).astype(int), 0, self.ny - 2); g = v - j
        else:
            j = np.zeros_like(i); g = np.zeros_like(f)
        j1 = np.minimum(j + 1, self.ny - 1)
        idx = np.stack([j * self.nx + i, j * self.nx + i + 1, j1 * self.nx + i, j1 * self.nx + i + 1], 1)
        w = np.stack([(1 - f) * (1 - g), f * (1 - g), (1 - f) * g, f * g], 1)
        dwx = np.stack([-(1 - g), (1 - g), -g, g], 1) / self.dx
        dwy = np.stack([-(1 - f), -f, (1 - f), f], 1) / self.dy
        return idx, w, dwx, dwy

    def spread(self, x, y, a):
        idx, w, _, _ = self.T(x, y); A = np.zeros(self.n); np.add.at(A, idx.ravel(), (a[:, None] * w).ravel()); return A

    def leak(self, x, y, a):
        if len(x) == 0: return 1.0
        A = self.spread(x, y, a); s = self.s
        return (self.cc + 2 * s * self.b @ A + s * s * A @ self.mv(A)) / self.cc

    def resid(self, x, y, a):
        """g = b + s G A (site residual drive) for the current design."""
        A = self.spread(x, y, a) if len(x) else np.zeros(self.n)
        return self.b + self.s * self.mv(A)

    def leak_and_grad(self, x, y, a):
        idx, w, dwx, dwy = self.T(x, y); A = np.zeros(self.n); np.add.at(A, idx.ravel(), (a[:, None] * w).ravel()); s = self.s
        GA = self.mv(A); P = (self.cc + 2 * s * self.b @ A + s * s * A @ GA) / self.cc
        gA = (2 * s * self.b + 2 * s * s * GA) / self.cc           # dP/dA
        gk = gA[idx]                                                # (N,4)
        return P, np.sum(gk * dwx, 1) * a, np.sum(gk * dwy, 1) * a, np.sum(gk * w, 1)


# ---------------- geometry helpers ----------------
def pair_gap(x, y, r):
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None]) - r[:, None] - r[None]
    np.fill_diagonal(d, np.inf); return d

def min_gap(x, y, r):
    return pair_gap(x, y, r).min() if len(x) > 1 else np.inf

def feasible(x, y, r, gap, y_lo, y_hi, x_lim):
    ok = (r >= R_MIN - 1e-9).all() and (r <= R_MAX + 1e-9).all() and (y - r >= Y_EDGE - 1e-9).all()
    ok &= (y >= y_lo - 1e-9).all() and (y <= y_hi + 1e-9).all() and (np.abs(x) <= x_lim + 1e-9).all()
    return bool(ok and min_gap(x, y, r) >= gap - 1e-9)

def repair(x, y, r, gap):
    """Strict gap feasibility: shrink the radii of the worst violating pair (never below R_MIN; else drop the smaller post);
    tooth clearance by shrinking r. Returns new arrays (posts may be removed)."""
    x, y, r = x.copy(), y.copy(), r.copy()
    r = np.minimum(r, y - Y_EDGE)                                   # tooth clearance
    keep = r >= R_MIN - 1e-9; x, y, r = x[keep], y[keep], r[keep]
    while len(x) > 1:
        d = pair_gap(x, y, r); p, q = np.unravel_index(np.argmin(d), d.shape)
        deficit = gap - d[p, q]
        if deficit <= 0: break
        sp, sq = min(deficit / 2 + 1e-6, r[p] - R_MIN), min(deficit / 2 + 1e-6, r[q] - R_MIN)
        if sp + sq + 1e-9 < deficit:                                # cannot shrink enough: drop the smaller post
            drop = p if r[p] <= r[q] else q
            keep = np.ones(len(x), bool); keep[drop] = False; x, y, r = x[keep], y[keep], r[keep]; continue
        # split the deficit proportionally to what each can give
        tot = sp + sq; r[p] -= deficit * sp / tot + 5e-7; r[q] -= deficit * sq / tot + 5e-7
        r = np.maximum(r, R_MIN)
    return x, y, r


# ---------------- discrete moves ----------------
def insert_greedy(gm, x, y, a, gap, mask, max_posts=10**9, tol=1e-7, rows=None):
    """Exact-score matching pursuit on grid sites: for each allowed site, best feasible amplitude
    a* = clip(-g_i/(s G_ii), A_MIN, min(A_MAX, a_gapmax)); gain = 2 s a g_i + s^2 a^2 G_ii (<0 = improvement)."""
    s = gm.s; x, y, a = list(x), list(y), list(a)
    g = gm.resid(np.array(x), np.array(y), np.array(a))
    # clearance of every site to the existing posts (distance minus their radius)
    clr = np.full(gm.n, np.inf)
    for xp, yp, ap in zip(x, y, a):
        clr = np.minimum(clr, np.hypot(gm.X - xp, gm.Y - yp) - rad(ap))
    clr_tooth = gm.Y - Y_EDGE
    while len(x) < max_posts:
        rmax = np.minimum(np.minimum(clr - gap, clr_tooth), R_MAX)
        amax = amp(np.maximum(rmax, 0.0))
        ok = mask & (amax >= A_MIN)
        astar = np.clip(-g / (s * gm.diag), A_MIN, np.maximum(amax, A_MIN))
        gain = 2 * s * astar * g + s * s * astar ** 2 * gm.diag
        gain[~ok] = np.inf
        i = int(np.argmin(gain))
        if gain[i] >= -tol * gm.cc: break
        ai = astar[i]; x.append(gm.X[i]); y.append(gm.Y[i]); a.append(ai)
        g += s * ai * gm.col(i)
        clr = np.minimum(clr, np.hypot(gm.X - gm.X[i], gm.Y - gm.Y[i]) - rad(ai))
    return np.array(x), np.array(y), np.array(a)

def delete_greedy(gm, x, y, a):
    """Remove posts whose exact removal lowers P (one at a time, residual updated)."""
    x, y, a = x.copy(), y.copy(), a.copy(); s = gm.s
    while len(x):
        idx, w, _, _ = gm.T(x, y); g = gm.resid(x, y, a)
        # cc*dP(remove p) = -2 s d.g + s^2 d.G.d with d = a_p w_p on idx_p
        dG = np.array([w[p] @ gm.G[np.ix_(idx[p], idx[p])].astype(float) @ w[p] for p in range(len(x))])
        dP = -2 * s * a * np.sum(w * g[idx], 1) + s * s * a * a * dG
        p = int(np.argmin(dP))
        if dP[p] >= 0: break
        keep = np.ones(len(x), bool); keep[p] = False; x, y, a = x[keep], y[keep], a[keep]
    return x, y, a


def trim(gm, x, y, a, n):
    """Reduce to n posts by repeatedly removing the post whose exact removal hurts P least."""
    x, y, a = x.copy(), y.copy(), a.copy(); s = gm.s
    while len(x) > n:
        idx, w, _, _ = gm.T(x, y); g = gm.resid(x, y, a)
        dG = np.array([w[p] @ gm.G[np.ix_(idx[p], idx[p])] @ w[p] for p in range(len(x))])
        dP = -2 * s * a * np.sum(w * g[idx], 1) + s * s * a * a * dG
        p = int(np.argmin(dP)); keep = np.ones(len(x), bool); keep[p] = False; x, y, a = x[keep], y[keep], a[keep]
    return x, y, a


# ---------------- continuous polish ----------------
def polish(gm, x, y, a, gap, y_lo, y_hi, x_lim, free_y=True, maxiter=250):
    N = len(x)
    if N == 0: return x, y, a
    def obj(v, lam):
        xx, yy, aa = v[:N], v[N:2 * N], v[2 * N:]
        P, gx, gy, ga = gm.leak_and_grad(xx, yy, aa)
        rr = rad(aa); dr_da = R_REF / (2 * np.sqrt(np.maximum(aa, 1e-9)))
        dxm = xx[:, None] - xx[None]; dym = yy[:, None] - yy[None]; dist = np.hypot(dxm, dym) + 1e-9
        viol = np.triu(np.maximum(rr[:, None] + rr[None] + gap - dist, 0.0), 1)
        W = 2 * lam * (viol + viol.T)
        pen = lam * np.sum(viol ** 2)
        gx = gx - np.sum(W * dxm / dist, 1); gy = gy - np.sum(W * dym / dist, 1); ga = ga + np.sum(W, 1) * dr_da
        vt = np.maximum(Y_EDGE - (yy - rr), 0.0)                   # tooth clearance
        pen += lam * np.sum(vt ** 2); gy = gy - 2 * lam * vt; ga = ga + 2 * lam * vt * dr_da
        return P + pen, np.concatenate([gx, gy, ga])
    v = np.concatenate([x, y, a])
    yb = [(y_lo, y_hi)] * N if free_y else [(yy, yy) for yy in y]
    bnds = [(-x_lim, x_lim)] * N + yb + [(A_MIN, A_MAX)] * N
    for lam in (1e-6, 1e-5, 1e-4, 1e-3):
        v = minimize(obj, v, args=(lam,), jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": maxiter}).x
    return v[:N], v[N:2 * N], v[2 * N:]


# ---------------- full design loop ----------------
class Design:
    def __init__(self, x, y, r, P): self.x, self.y, self.r, self.P = np.asarray(x, float), np.asarray(y, float), np.asarray(r, float), float(P)
    @property
    def N(self): return len(self.x)

def evaluate(gm, x, y, r): return gm.leak(np.asarray(x), np.asarray(y), amp(r))

def design(gm, gap, x_lim, y_lo, y_hi, seeds, rows=None, max_posts=10**9, rounds=2, hops=2, rng=0, verbose=True, label=""):
    """Incumbent-preserving search. seeds: list of (x, y, r) feasible-or-not starting designs ([] = empty).
    rows: list of fixed row y's (nm) -> straight-row design (y frozen); None -> free 2D in [y_lo, y_hi].
    Returns the best strictly feasible Design (seeds are repaired and included in the comparison)."""
    rng = np.random.default_rng(rng); t0 = time.time(); free_y = rows is None
    mask = (np.abs(gm.X) <= x_lim + 1e-6) & (gm.Y >= y_lo - 1e-6) & (gm.Y <= y_hi + 1e-6)
    if rows is not None:                             # row design: only sites on the allowed rows
        mask &= np.isin(np.round(gm.Y), np.round(np.asarray(rows)))
    best = None
    def consider(x, y, a, tag):
        nonlocal best
        x, y, r = repair(np.asarray(x), np.asarray(y), rad(np.asarray(a)), gap)
        if len(x) > max_posts or not feasible(x, y, r, gap, y_lo, y_hi, x_lim): return None
        P = evaluate(gm, x, y, r); d = Design(x, y, r, P)
        if best is None or P < best.P - 1e-12:
            best = d
            if verbose: print(f"  [{label}] {tag:14s} P {P:.4f} N {d.N} min gap {min_gap(x, y, r):.0f} t {time.time()-t0:.0f}s", flush=True)
        return d
    def one_pass(x, y, a, tag):
        x, y, a = insert_greedy(gm, x, y, a, gap, mask, max_posts); consider(x, y, a, tag + "+ins")
        for _ in range(rounds):
            x, y, a = polish(gm, x, y, a, gap, y_lo, y_hi, x_lim, free_y=free_y)
            x, y, r = repair(x, y, rad(a), gap); a = amp(r); consider(x, y, a, tag + "+pol")
            x, y, a = delete_greedy(gm, x, y, a); x, y, a = insert_greedy(gm, x, y, a, gap, mask, max_posts); consider(x, y, a, tag + "+del/ins")
        return x, y, a
    for k, sd in enumerate(seeds):
        x, y, r = (np.asarray(v, float) for v in sd) if len(sd[0]) else (np.zeros(0),) * 3
        if len(x):
            x, y, r = repair(x[np.abs(x) <= x_lim], y[np.abs(x) <= x_lim], r[np.abs(x) <= x_lim], gap)   # cut to the region first
            kk = (y >= y_lo) & (y <= y_hi)
            if rows is not None: kk &= np.isin(np.round(y), np.round(np.asarray(rows)))
            x, y, r = x[kk], y[kk], r[kk]
            if len(x) > max_posts: x, y, a_ = trim(gm, x, y, amp(r), max_posts); r = rad(a_)
            consider(x, y, amp(r), f"seed{k}")
        one_pass(x, y, amp(r), f"seed{k}")
    # basin hopping around the incumbent: remove a random cluster, re-insert, polish
    for h in range(hops):
        x, y, a = best.x.copy(), best.y.copy(), amp(best.r)
        if len(x) > 5:
            c = rng.integers(len(x)); dcl = np.hypot(x - x[c], y - y[c]); keep = dcl > rng.uniform(300, 1500)
            x, y, a = x[keep], y[keep], a[keep]
        one_pass(x, y, a, f"hop{h}")
    return best


def ceiling(gm, mask, iters=3000):
    """Non-negative any-size ceiling on the masked sites (FISTA, s = 1): upper bound on P for this region."""
    G, b = gm.G, gm.b; m = mask.astype(float)
    v = np.random.default_rng(0).standard_normal(gm.n) * m
    for _ in range(30): v = gm.mv(v) * m; v /= np.linalg.norm(v)
    L = 2 * float(v @ gm.mv(v)); A = np.zeros(gm.n); Z = A.copy(); t = 1.0; Pold = np.inf
    for it in range(iters):
        An = np.maximum(Z - (2 * gm.mv(Z) + 2 * b) * m / L, 0.0) * m
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); Z = An + (t - 1) / tn * (An - A); A, t = An, tn
        if it % 200 == 0:
            P = (gm.cc + 2 * b @ A + A @ gm.mv(A)) / gm.cc
            if Pold - P < 1e-6: break
            Pold = P
    P = (gm.cc + 2 * b @ A + A @ gm.mv(A)) / gm.cc
    return P, A
