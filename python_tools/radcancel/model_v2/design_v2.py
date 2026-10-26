# design_v2.py (2026-10-25): FAST design with model v2 (Gram form). At fixed candidate sites the model far field is AFFINE in the post areas
# a_j = (r_j / 75 nm)^2:  E(a) = C + sum_j a_j g_j, with g_j = the post's direct + strip re-radiation field (patch weights W of the chosen
# model_v2 config) [+ its mode-reshaping share w_j h(|x_j|) C], drive = control drive x the coupled-dipole (DDA) local-field factor of the
# CURRENT design (fixed point: re-linearised every round). P(a) = 1 + 2 b.a + a.M.a; gain = 1 / (f P + c) (config map).
# Constraints: |x| <= XMAX, r in [50, 100] nm, edge gaps >= 100 nm (also between +y posts and their -y mirrors), local-wall clearance >= CLR.
#   python design_v2.py scratch <tag>        free 2-D cloud from scratch (candidate grid)
#   python design_v2.py seed_ext42 <tag>     ext42 x positions, free y per post
import os, sys, json, time, numpy as np
from scipy.special import hankel1
from scipy.optimize import minimize
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import model_v2 as mm
XMAX, CLR, RMIN, RMAX, GAP, R0 = float(os.environ.get("XMAX_NM", 42000)), 200.0, float(os.environ.get("RMIN_NM", 50)), 100.0, float(os.environ.get("GAP_NM", 100)), 75.0   # fab variants via env (A: 50/100, A-cons: 75/150, B: 35/70, 25/50)
YMAX = float(os.environ.get("YMAX_NM", 1950))
T = json.load(open(os.path.join(mm.DATA, "teeth.json"))); EDGES = T["CAV"] / 2 + T["PITCH"] * np.arange(len(T["NAR"]) + 1)
TCAD = json.load(open(os.path.join(mm.DATA, "teeth_cad.json")))["rects"]
def wall(x, reach):
    """local built wall half-width (nm) within +-reach of x: max over the per-tooth arrays (CLAUDE.md sec 5: never the avg+corr/2 formula)."""
    # 2026-10-26 GPT audit: the formula (EDGES from CAV = cavity WIDTH, abs(x) = mirrored arms) misplaced teeth; real clearances were 160-180 nm.
    # Use the BUILT rectangles of the Athena scene (data/teeth_cad.json: teeth + cavity, per arm, exact x extents).
    return max(hw for x0, x1, hw in TCAD if x0 - reach <= x <= x1 + reach)
W, PINV = mm._W["W"], mm._W["pinv"]; F_MAP, C_MAP = mm.FINAL["map"]; NB = mm.FINAL["NB"]; VAR = mm.FINAL["variant"]
NA = mm.NHAT.shape[0]; Cv = mm.C.reshape(2, -1); dWv = mm.dW.reshape(2, -1); CC = mm.CC
KT = mm.K * np.sqrt(np.maximum(1 - mm.UX ** 2, 0.05 ** 2))
Wd = [[W[pol][PINV][:, c] for c in range(NB)] for pol in range(2)]       # per pol, per basis k: (NA,) weights
if mm.FINAL.get("resh"): KN, HK, AN = mm._W["KNOTS"], mm._W["hk"], float(mm._W["An"])
def local_factor(xc, yc, des, e_des):
    """DDA local/incident field factor (Nc, 2) at candidate sites for the current design des (x, y, r nm) with its self-consistent dipoles."""
    if des is None or len(des[0]) == 0 or not VAR.get("dda"): return np.ones((len(xc), 2), complex)
    x, y, r = des; NZ = 3; H = mm.H0_NM * 1e-9; zs = (np.arange(NZ) + 0.5) / NZ * H - H / 2
    ratio = mm._dda(x, y, r, e_des, mm.H0_NM)                                                    # design's own local field / incident
    El = e_des[:, :2] * ratio                                                                     # (N, 2) slice-averaged local field
    q = np.pi * (r * 1e-9) ** 2 * H / NZ
    P = np.array([[a * 1e-9, b * 1e-9, z] for a, b in zip(x, y) for z in zs]); pz = np.repeat(np.arange(len(x)), NZ)
    p = np.repeat(El, NZ, 0) * (np.repeat(q, NZ)[:, None] * mm.GAM[None, :2])                    # slice dipoles (2 comps)
    M = np.diag([1.0, -1.0, 1.0]) * mm.S_Y; e_inc, _ = mm.drive(xc, yc); out = np.ones((len(xc), 2), complex)
    for j, (a, b) in enumerate(zip(xc, yc)):
        Ef = np.zeros(2, complex)
        for z in zs:
            Rp = np.array([a * 1e-9, b * 1e-9, z]); d1 = Rp - P; d2 = Rp - P * np.array([1, -1, 1])
            own = (np.abs(d1[:, 0]) < 1e-12) & (np.abs(d1[:, 1]) < 1e-12)                          # exclude the candidate's own column
            K1 = kern(np.where(own[:, None], 1.0, d1)); K1[own] = 0; K2 = kern(d2) @ M
            Ef += np.einsum("nab,nb->a", (K1 + K2)[:, :2, :2], p) / NZ
        out[j] = 1 + Ef / np.where(np.abs(e_inc[j, :2]) > 0, e_inc[j, :2], 1)
    return out
def kern(R):
    rr = np.linalg.norm(R, axis=1); n = R / rr[:, None]; nn = np.einsum("ni,nj->nij", n, n); I = np.eye(3)[None]
    return (np.exp(1j * mm.K * rr) / (4 * np.pi))[:, None, None] * (mm.K ** 2 * (I - nn) / rr[:, None, None] + (3 * nn - I) * (1 / rr ** 3 - 1j * mm.K / rr ** 2)[:, None, None])
def gram(xc, yc, fac, chunk=2048):
    """M (Nc, Nc), b (Nc,) for unit-area (r = R0) posts at (xc, yc) with drive factor fac (Nc, 2)."""
    e0, _ = mm.drive(xc, yc); pref = mm.K ** 2 / (4 * np.pi) * np.pi * (R0 * 1e-9) ** 2 * (mm.H0_NM * 1e-9)
    p = pref * mm.GAM[None, :2] * e0[:, :2] * fac * mm.NORM                                         # (Nc, 2)
    ares = np.zeros(len(xc), complex)
    if mm.FINAL.get("resh"):
        hat = np.array([np.interp(np.abs(xc) / 1e3, KN, np.eye(KN.size)[m]) for m in range(KN.size)]).T
        ares = (R0 * 1e-3) ** 2 * np.sum(np.abs(e0[:, :2]) ** 2, 1) / AN * (hat @ HK)              # alpha per unit area
    Nc = len(xc); M = np.zeros((Nc, Nc)); b = np.zeros(Nc)
    for i0 in range(0, NA, chunk):
        sl = slice(i0, min(NA, i0 + chunk)); n = mm.NHAT[sl]
        php = np.exp(-1j * mm.K * (np.outer(xc * 1e-9, n[:, 0]) + np.outer(yc * 1e-9, n[:, 1])))
        phm = np.exp(-1j * mm.K * (np.outer(xc * 1e-9, n[:, 0]) - np.outer(yc * 1e-9, n[:, 1])))
        img = hankel1(0, np.outer(yc * 1e-9, KT[sl])) * np.exp(-1j * mm.K * np.outer(xc * 1e-9, mm.UX[sl]))
        for pol, V in ((0, mm.TH[sl]), (1, mm.PH[sl])):
            G = np.zeros((Nc, sl.stop - sl.start), complex)
            for c in (0, 1):
                mir = mm.S_Y * (1 if c == 0 else -1)
                G += p[:, c:c + 1] * ((php + mir * phm) * V[None, :, c]) * Wd[pol][c][sl][None]
                if NB == 4: G += p[:, c:c + 1] * img * Wd[pol][2 + c][sl][None]
            G += ares[:, None] * Cv[pol, sl][None]
            Gw = (G * dWv[pol, sl][None]).astype(np.complex64)
            M += np.real(np.conj(G.astype(np.complex64)) @ Gw.T) / CC; b += np.real(Gw @ np.conj(Cv[pol, sl])) / CC
    return M, b
def candidates(mode, seed=None):
    if mode == "scratch":
        xs = np.arange(-XMAX, XMAX + 1, 300.0); ys = np.arange(800.0, 1951.0, 100.0)
        X, Y = np.meshgrid(xs, ys, indexing="ij"); X, Y = X.ravel(), Y.ravel(); grp = np.arange(X.size)
    else:   # seed: each seed post keeps x, free y on a 50 nm ladder; one post per seed (group)
        sx = np.asarray(seed["x_nm"], float); Y0 = np.arange(800.0, 1951.0, 100.0)
        X = np.repeat(sx, Y0.size); Y = np.tile(Y0, sx.size); grp = np.repeat(np.arange(sx.size), Y0.size)
    rmax = np.array([min(RMAX, y - CLR - wall(x, RMAX + 100)) for x, y in zip(X, Y)])
    ok = (rmax >= RMIN) & (np.abs(X) <= XMAX)
    return X[ok], Y[ok], rmax[ok], grp[ok]
def P_of(a, M, b): return 1 + 2 * b @ a + a @ M @ a
def solve_box(M, b, lo, hi, a0):
    f = lambda a: (P_of(a, M, b), 2 * b + 2 * M @ a)
    return minimize(f, np.clip(a0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi))).x
def ok_gap(x, y, r, xs, ys, rs):
    if len(xs) == 0: return True
    d = np.hypot(np.asarray(xs) - x, np.asarray(ys) - y); dm = np.hypot(np.asarray(xs) - x, np.asarray(ys) + y)
    return bool(np.all(d >= r + np.asarray(rs) + GAP) and np.all(dm >= r + np.asarray(rs) + GAP) and 2 * y >= 2 * r + GAP)
def rlim(j, sel, r, X, Y, RM):
    """largest feasible radius of candidate j given the selected posts (gaps incl. mirrors, wall via RM)."""
    lim = min(RM[j], Y[j] - GAP / 2)
    if len(sel):
        s = np.array(sel); others = s[s != j]
        if others.size:
            d = np.hypot(X[others] - X[j], Y[others] - Y[j]); dm = np.hypot(X[others] - X[j], Y[others] + Y[j])
            lim = min(lim, np.min(d - r[others] - GAP), np.min(dm - r[others] - GAP))
    return lim
def polish(M, b, X, Y, RM, GRP, sel, r, maxit=600):
    """exact local search on the quadratic: best of {add, remove, resize} per step, always feasible; r (all candidates, 0 = absent)."""
    amin = (RMIN / R0) ** 2; a = (r / R0) ** 2; sel = list(sel); n0 = len(sel)
    for it in range(maxit):
        g = b + M @ a; best = (0.0, None)
        used = {GRP[i] for i in sel}
        for i in sel:                                                   # remove
            dP = -2 * a[i] * g[i] + a[i] ** 2 * M[i, i]
            if dP < best[0]: best = (dP, ("rm", i))
            lim = rlim(i, sel, r, X, Y, RM)                             # resize
            ai = np.clip(a[i] - g[i] / M[i, i], amin, (max(lim, RMIN) / R0) ** 2); da = ai - a[i]
            dP = 2 * da * g[i] + da ** 2 * M[i, i]
            if dP < best[0] and lim >= RMIN: best = (dP, ("rs", i, ai))
        cand = np.where((r == 0) & (g < 0))[0]
        cand = cand[np.argsort(g[cand] / np.sqrt(M.diagonal()[cand]))][:200]          # most promising adds
        for j in cand:
            if GRP[j] in used: continue
            lim = rlim(j, sel, r, X, Y, RM)
            if lim < RMIN: continue
            aj = np.clip(-g[j] / M[j, j], amin, (lim / R0) ** 2); dP = 2 * aj * g[j] + aj ** 2 * M[j, j]
            if dP < best[0]: best = (dP, ("add", j, aj))
        if best[0] > -1e-6: break
        mv = best[1]
        if mv[0] == "rm": sel.remove(mv[1]); a[mv[1]] = 0
        elif mv[0] == "rs": a[mv[1]] = mv[2]
        else: sel.append(mv[1]); a[mv[1]] = mv[2]
        r = np.where(a > 0, R0 * np.sqrt(a), 0.0)
    print(f"    polish: {it} moves, posts {n0} -> {len(sel)}, P {P_of(a, M, b):.4f}", flush=True)
    return sel, r
def design(mode, tag, rounds=3, seed=None):
    X, Y, RM, GRP = candidates(mode, seed); amax = (RM / R0) ** 2; amin = (RMIN / R0) ** 2
    print(f"{tag}: {X.size} candidates, |x| <= {XMAX/1e3:.0f} um, y {Y.min():.0f}-{Y.max():.0f} nm", flush=True)
    des = None; e_des = None; a = np.zeros(X.size); hist = []
    for rd in range(rounds):
        t0 = time.time(); fg = os.path.join(D, "cache", f"gram0_{mode}_{mm.FINAL['tag']}_{X.size}.npz")
        if rd == 0 and os.path.exists(fg): z = np.load(fg); M, b = z["M"], z["b"]
        else:
            fac = local_factor(X, Y, des, e_des); M, b = gram(X, Y, fac)
            if rd == 0: np.savez(fg, M=M, b=b)
        a = solve_box(M, b, np.zeros(X.size), amax, a)                               # continuous relaxation (0 = no post)
        # greedy discretisation: by marginal value a_j * (-grad_j at 0) proxy = a_j |b_j + (M a)_j|, gaps + one-per-group
        order = np.argsort(-(a * np.abs(b + M @ a - M.diagonal() * a)))
        keep, kx, ky, kr, used = [], [], [], [], set()
        for j in order:
            if a[j] < 0.5 * amin or GRP[j] in used: continue
            r = float(np.clip(R0 * np.sqrt(a[j]), RMIN, RM[j]))
            if ok_gap(X[j], Y[j], r, kx, ky, kr): keep.append(j); kx.append(X[j]); ky.append(Y[j]); kr.append(r); used.add(GRP[j])
        rr_all = np.zeros(X.size); rr_all[keep] = kr
        if des is not None:                                              # also offer the previous design as a starting point: keep the better
            r_prev = np.zeros(X.size); idx0 = {(x_, y_): i for i, (x_, y_) in enumerate(zip(X, Y))}
            for x_, y_, r_ in zip(*des): r_prev[idx0[(x_, y_)]] = r_
            if P_of((r_prev / R0) ** 2, M, b) < P_of((rr_all / R0) ** 2, M, b): keep, rr_all = list(np.where(r_prev > 0)[0]), r_prev
        sel, rr_all = polish(M, b, X, Y, RM, GRP, list(keep), rr_all)
        sel = sorted(sel); fx, fy, fr = list(X[sel]), list(Y[sel]), list(rr_all[sel])
        des = (np.array(fx), np.array(fy), np.array(fr)); e_des, _ = mm.drive(des[0], des[1])
        a = np.zeros(X.size); idx = {(x_, y_): i for i, (x_, y_) in enumerate(zip(X, Y))}
        for x_, y_, r_ in zip(*des): a[idx[(x_, y_)]] = (r_ / R0) ** 2
        Pl = P_of(a, M, b); g_lin = 1 / (F_MAP * Pl + C_MAP)
        hist.append(dict(round=rd, N=len(fx), P_lin=float(Pl), gain_lin=float(g_lin)))
        print(f"  round {rd}: N {len(fx)} posts/side, linearised P {Pl:.4f} gain x{g_lin:.3f} ({time.time()-t0:.0f} s)", flush=True)
    full = mm.predict(dict(x_nm=des[0], y_nm=des[1], r_nm=des[2]))
    print(f"  FULL model_v2 ({mm.FINAL['tag']}): gain x{full['Qi_gain']:.3f}, P_tube {full['P_tube']:.4f}", flush=True)
    return des, full, hist
def check(des):
    x, y, r = des; n = len(x); bad = []
    for i in range(n):
        if not (RMIN - 1e-6 <= r[i] <= RMAX + 1e-6): bad.append(("r", i))
        if abs(x[i]) > XMAX + 1e-6: bad.append(("x", i))
        if y[i] - r[i] - wall(x[i], r[i] + 100) < CLR - 1e-6: bad.append(("clr", i))
        if 2 * y[i] - 2 * r[i] < GAP - 1e-6: bad.append(("mirror", i))
        for j in range(i + 1, n):
            if np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j] < GAP - 1e-6: bad.append(("gap", i, j))
    return bad
if __name__ == "__main__":
    mode, tag = sys.argv[1], sys.argv[2]
    seed = json.load(open(r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/runners/sweeps/postarray_overshoot_designs.json"))["nw_r2pack_ext42"] if mode == "seed_ext42" else None
    des, full, hist = design(mode, tag, seed=seed)
    bad = check(des); assert not bad, bad[:10]
    x, y, r = des
    print(f"  checks PASS: {len(x)} posts/side, r {r.min():.0f}-{r.max():.0f}, min edge gap OK, min wall clearance "
          f"{min(yy - rr - wall(xx, rr + 100) for xx, yy, rr in zip(x, y, r)):.0f} nm, |x|max {np.abs(x).max():.0f}; x>0 {np.sum(x>0)} / x<0 {np.sum(x<0)}")
    json.dump(dict(x_nm=[round(float(v), 1) for v in x], y_nm=[round(float(v), 1) for v in y], r_nm=[round(float(v), 1) for v in r],
                   model=mm.FINAL["tag"], gain_pred=full["Qi_gain"], P_tube=full["P_tube"], history=hist,
                   source=f"model v2 design ({mode}), {time.strftime('%Y-%m-%d %H:%M')}, |x|<={XMAX/1e3:.0f} um, r 50-100, gaps>=100, wall clr>={CLR:.0f}"),
              open(os.path.join(D, f"design_{tag}.json"), "w"), indent=1)
