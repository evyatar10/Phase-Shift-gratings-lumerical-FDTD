"""v3-step local gate (zero GPU, pure math): peak3, cw_from_widths, qp_step, radius_update.

Study dir: runners/lumopt2_design/gates/ | Created 2026-10-05 | zero GPU
Usage (repo root):  python runners/lumopt2_design/gates/gate_v3_local.py
Every property that can be faked gets a must-fail tooth: the checker is run
on a known-bad step (old projection-then-clip, unnormalised D·g, raw sampled
max) and must reject it.
"""
import os
import sys
import time

import autograd
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from runners.lumopt2_design import v3_step as v3                # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ================================================================ peak3
def lorentz(f, f0, gam, T0=0.9):
    return T0 / (1.0 + ((f - f0) / gam) ** 2)


gam = 1.0                       # HWHM; 50 samples per FWHM ⇒ step 2γ/50
h = 2 * gam / 50
f = np.arange(-60, 61) * h
T = lorentz(f, 0.3 * h, gam)
i = int(np.argmax(T))
r = v3.peak3_r(T, i)
w_an = np.array([r * (r - 1) / 2, 1 - r * r, r * (r + 1) / 2])
g_ad = autograd.grad(lambda x: v3.peak3(x, i))(T)
g_fd = np.zeros_like(T)
for j in range(i - 2, i + 3):
    e = np.zeros_like(T)
    e[j] = 1e-7
    g_fd[j] = (v3.peak3(T + e, i) - v3.peak3(T - e, i)) / 2e-7
w_full = np.zeros_like(T)
w_full[i - 1:i + 2] = w_an
check("peak3 weights: analytic vs autograd vs FD",
      np.abs(g_ad - w_full).max() < 1e-8 and np.abs(g_fd - w_full).max() < 1e-8,
      f"r={r:+.3f} |ad-an|={np.abs(g_ad - w_full).max():.1e} |fd-an|={np.abs(g_fd - w_full).max():.1e}")

bias3, bias_raw = 0.0, 0.0
for off in np.linspace(-0.5, 0.5, 41) * h:
    T = lorentz(f, off, gam)
    i = int(np.argmax(T))
    bias3 = max(bias3, abs(v3.peak3(T, i) - 0.9))
    bias_raw = max(bias_raw, abs(T[i] - 0.9))
    assert abs(v3.peak3(T[::-1], T.size - 1 - i) - v3.peak3(T, i)) < 1e-15   # order-free
check("peak3 Lorentzian bias (50 pts/FWHM, offsets ±½ step) < 5e-6", bias3 < 5e-6, f"worst {bias3:.2e}")
check("  tooth: raw sampled max FAILS the same bar", bias_raw > 5e-6, f"worst {bias_raw:.2e}")

Tc = np.array([0.5, 0.6, 0.8, 0.7, 0.9])         # i=2: neighbours 0.6/0.7 ⇒ B<0 ok; i=3: B>0 (a dip)
gd = autograd.grad(lambda x: v3.peak3(x, 3))(Tc)
check("peak3 fallback B≥0 → T[i], one-hot grad",
      v3.peak3(Tc, 3) == 0.7 and np.array_equal(gd, np.eye(5)[3]) and v3.peak3_r(Tc, 3) == 0.0)
check("peak3 edge index → T[i]", v3.peak3(Tc, 0) == 0.5 and v3.peak3(Tc, 4) == 0.9)

# ================================================================ cw_from_widths
lam = 1560 + np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
s, c = v3.cw_from_widths(lam, 19.0 + 0.8 * (lam - 1560))
check("cw linear 5-pt: slope exact, not curved", abs(s - 0.8) < 1e-9 and not c, f"slope {s:.12f}")
s3, c3 = v3.cw_from_widths(lam[1:4], 19.0 + 0.8 * (lam[1:4] - 1560))
check("cw linear 3-pt: slope exact, not curved", abs(s3 - 0.8) < 1e-9 and not c3)
s, c = v3.cw_from_widths(lam, 19.0 + 0.8 * (lam - 1560) + 1.0 * (lam - 1560) ** 2)
check("cw mild curvature not flagged", not c and abs(s - 0.8) < 1e-9, f"2c2h/s={2*1*0.04/0.8:.2f}")
s, c = v3.cw_from_widths(lam, 19.0 + 0.8 * (lam - 1560) + 400.0 * (lam - 1560) ** 2)
check("cw strong curvature flagged (tooth)", c, f"2c2h/s={2*400*0.04/0.8:.1f}")


# ================================================================ qp_step helpers
def boxes(lo, hi, cap):
    return np.maximum(lo, -cap), np.minimum(hi, cap)


def instance(rng, n, m, family):
    if family == "engine":            # D = (bound half-range)², 10 % at a bound, 10 % frozen
        hr = 10 ** rng.uniform(-0.5, 2, n)
        p = rng.uniform(-1, 1, n) * hr
        atb = rng.random(n) < 0.1
        p[atb] = np.where(rng.random(atb.sum()) < 0.5, -1.0, 1.0) * hr[atb]
        fz = rng.random(n) < 0.1
        hr[fz], p[fz] = 1e-3, 0.0
        D, lo, hi = hr ** 2, -hr - p, hr - p
    else:                             # stress: D independent of the bounds
        D = 10 ** rng.uniform(-6, 4, n)
        lo, hi = -rng.uniform(0, 30, n), rng.uniform(0, 30, n)
        lo[rng.random(n) < 0.1] = 0.0
        hi[rng.random(n) < 0.1] = 0.0
    cap = rng.uniform(2, 30)
    g = rng.standard_normal(n) * 10 ** rng.uniform(-3, 3)
    a1 = rng.standard_normal(n) * 10 ** rng.uniform(-3, 1)
    A = [a1] if m == 1 else [a1, rng.uniform(-1, 1) * a1 * 10 ** rng.uniform(-1, 1)
                             + rng.standard_normal(n) * 10 ** rng.uniform(-2, 1)]
    lb, ub = boxes(lo, hi, cap)
    d_f = rng.uniform(lb, ub)                      # a feasible point ⇒ every band below holds there
    tau = cap / np.abs(D * g).max()
    d0 = np.clip(tau * D * g, lb, ub)
    bands = []
    for a in A:
        yf, y0, R = a @ d_f, a @ d0, np.abs(a) @ (ub - lb)
        kind = rng.integers(4)
        w = R * rng.uniform(0.001, 0.1)
        bands.append((min(yf, y0) - w, max(yf, y0) + w) if kind == 0 else
                     (yf - w, yf + w) if kind == 1 else (yf, yf) if kind == 2 else (yf - w, yf + R))
    return g, A, bands, D, lo, hi, cap


def objective(g, D, tau, d):
    return g @ d - (d * d / D).sum() / (2 * tau)


def row_viol(A, bands, d, cap):
    """Max band violation, scaled by cap·‖a‖₁ (the row's natural reach)."""
    return max([max(0.0, L - a @ d, a @ d - U) / (cap * np.abs(a).sum())
                for a, (L, U) in zip(A, bands)] + [0.0])


def ref_solve(g, A, bands, D, lo, hi, cap):
    """SLSQP reference in z = d/√D (removes the 1e10 Hessian spread; same problem)."""
    sq = np.sqrt(D)
    lb, ub = boxes(lo, hi, cap)
    tau = cap / np.abs(D * g).max()
    c = sq * g
    fs = tau * (c @ c) / 2                         # unconstrained optimum value: objective scale
    cons = []
    for a, (L, U) in zip(A, bands):
        t = cap * np.abs(a).sum()
        az = a * sq / t
        if L == U:
            cons.append({"type": "eq", "fun": lambda z, az=az, L=L / t: az @ z - L, "jac": lambda z, az=az: az})
        else:
            cons += [{"type": "ineq", "fun": lambda z, az=az, L=L / t: az @ z - L, "jac": lambda z, az=az: az},
                     {"type": "ineq", "fun": lambda z, az=az, U=U / t: U - az @ z, "jac": lambda z, az=az: -az}]
    res = minimize(lambda z: (z @ z / (2 * tau) - c @ z) / fs, np.clip(0 * c, lb / sq, ub / sq),
                   jac=lambda z: (z / tau - c) / fs, method="SLSQP",
                   bounds=list(zip(lb / sq, ub / sq)), constraints=cons,
                   options={"ftol": 1e-15, "maxiter": 3000})
    return res.x * sq, fs


def in_box(d, lo, hi, cap):
    return np.abs(d).max() <= cap * (1 + 1e-15) and np.all(d >= lo) and np.all(d <= hi)


ALL_STEPS = []                                     # (d, lo, hi, cap, frozen mask) for (d)/(g)

# ================================================================ (a) reference solver
rng = np.random.default_rng(20261005)
worst_rel, worst_v, worst_ref_v, n_ok, n_tot, iters, n_act = 0.0, 0.0, 0.0, 0, 0, [], 0
t0 = time.time()
for k in range(216):
    n = 296 if k >= 200 else 40
    g, A, bands, D, lo, hi, cap = instance(rng, n, 1 + k % 2, "engine" if k % 4 < 2 else "stress")
    out = v3.qp_step(g, A, bands, D, lo, hi, cap)
    d, tau = out["d"], out["tau"]
    ALL_STEPS.append((d, lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
    iters.append(out["iters"])
    n_act += any(out["row_active"])
    dr, fs = ref_solve(g, A, bands, D, lo, hi, cap)
    vm, vr = row_viol(A, bands, d, cap), row_viol(A, bands, dr, cap)
    fm, fr = objective(g, D, tau, d), objective(g, D, tau, dr)
    rel = (fr - fm) / max(abs(fr), 1e-3 * fs)      # >0 ⇒ reference better
    n_tot += 1
    worst_v = max(worst_v, vm)
    if out["mode"] != "ascent":
        print(f"   instance {k}: unexpected mode {out['mode']}")
        continue
    if vr <= 1e-7:
        n_ok += 1
        worst_rel = max(worst_rel, rel)
    worst_ref_v = max(worst_ref_v, vr)
check(f"(a) vs SLSQP on {n_tot} instances (200×n40 + 16×n296, 1-2 rows)",
      worst_rel <= 1e-6 and worst_v <= 1e-7 and n_ok >= 0.9 * n_tot,
      f"worst (f_ref−f_v3)/|f| {worst_rel:.1e}; worst v3 viol {worst_v:.1e}; "
      f"ref feasible {n_ok}/{n_tot} (worst ref viol {worst_ref_v:.1e}); "
      f"rows active in {n_act}; Newton iters median {int(np.median(iters))} max {max(iters)}; "
      f"{time.time() - t0:.0f}s")
# tooth: the bare clip(τDg) step (rows ignored) must violate an active band
g, A, bands, D, lo, hi, cap = instance(np.random.default_rng(7), 40, 1, "engine")
lb, ub = boxes(lo, hi, cap)
d_naive = np.clip(cap / np.abs(D * g).max() * D * g, lb, ub)
y0 = A[0] @ d_naive
bands = [(y0 + 0.2 * np.abs(A[0]) @ (ub - lb) * 0.1, y0 + 0.2 * np.abs(A[0]) @ (ub - lb) * 0.2)]
out = v3.qp_step(g, A, bands, D, lo, hi, cap)
check("  tooth: rows-ignored step FAILS the violation check; v3 passes",
      row_viol(A, bands, d_naive, cap) > 1e-7 and row_viol(A, bands, out["d"], cap) <= 1e-7)

# ================================================================ (b) no active rows, (h) τ rule
g, A, _, D, lo, hi, cap = instance(np.random.default_rng(11), 296, 2, "engine")
j = int(np.argmax(np.abs(D * g)))
lo[j], hi[j] = -100.0, 100.0                       # the τ-defining param is not bound-limited
lb, ub = boxes(lo, hi, cap)
tau = cap / np.abs(D * g).max()
d0 = np.clip(tau * D * g, lb, ub)
wide = [(a @ d0 - 1e3, a @ d0 + 1e3) for a in A]
out = v3.qp_step(g, A, wide, D, lo, hi, cap)
ALL_STEPS.append((out["d"], lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
check("(b) no rows active ⇒ d = clip(τ·D·gT)", np.abs(out["d"] - d0).max() < 1e-12
      and not any(out["row_active"]), f"|Δ|={np.abs(out['d'] - d0).max():.1e}")
check("(h) τ rule: slack bands ⇒ max|d| = cap exactly", abs(np.abs(out["d"]).max() - cap) < 1e-12 * cap,
      f"max|d|={np.abs(out['d']).max():.6f} cap={cap:.6f}")
d_old = np.clip(D * g, lb, ub)                     # unnormalised: step length tied to |gT|
check("  tooth: τ=1 (D·g) step does NOT reach the cap", abs(np.abs(d_old).max() - cap) > 1e-6 * cap)
tight = [(a @ d0 + 0.05 * cap * np.abs(a).sum(), a @ d0 + 0.06 * cap * np.abs(a).sum()) for a in A[:1]]
out_t = v3.qp_step(g, A[:1], tight, D, lo, hi, cap)
check("  tooth: with an active row the (b) identity FAILS", np.abs(out_t["d"] - d0).max() > 1e-6
      and out_t["row_active"][0])

# ================================================================ (c) invariance
rng = np.random.default_rng(3)
worst_g, worst_row, n_active = 0.0, 0.0, 0
for k in range(30):
    g, A, bands, D, lo, hi, cap = instance(rng, 40, 2, "engine" if k % 2 else "stress")
    base = v3.qp_step(g, A, bands, D, lo, hi, cap)
    n_active += any(base["row_active"])
    big = v3.qp_step(1e3 * g, A, bands, D, lo, hi, cap)
    sc = v3.qp_step(g, [1e-5 * A[0], A[1]], [(1e-5 * bands[0][0], 1e-5 * bands[0][1]), bands[1]], D, lo, hi, cap)
    worst_g = max(worst_g, np.abs(big["d"] - base["d"]).max())
    worst_row = max(worst_row, np.abs(sc["d"] - base["d"]).max())
check("(c) gT×1e3 ⇒ same d (1e-9)", worst_g < 1e-9, f"worst {worst_g:.1e}  ({n_active}/30 with active rows)")
check("(c) row+band ×1e-5 ⇒ same d (1e-7)", worst_row < 1e-7, f"worst {worst_row:.1e}")
d1 = np.clip(D * g, *boxes(lo, hi, cap))
d2 = np.clip(D * 1e3 * g, *boxes(lo, hi, cap))
check("  tooth: unnormalised D·g step is NOT gT-scale invariant", np.abs(d1 - d2).max() > 1e-6)

# ================================================================ (e) width band above 0
g, A, _, D, lo, hi, cap = instance(np.random.default_rng(5), 296, 1, "engine")
lb, ub = boxes(lo, hi, cap)
a = A[0]
y_max = np.maximum(a * lb, a * ub).sum()          # LP bound: best reachable a·d inside the box
L = 0.3 * y_max
out = v3.qp_step(g, [a], [(L, L + 0.05 * y_max)], D, lo, hi, cap)
ALL_STEPS.append((out["d"], lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
check("(e) reachable L>0 ⇒ ascent and a·d ≥ L", out["mode"] == "ascent"
      and a @ out["d"] >= L - 1e-9 * cap * np.abs(a).sum(), f"a·d={a @ out['d']:.6g} L={L:.6g}")
out = v3.qp_step(g, [a], [(1.5 * y_max, 2 * y_max)], D, lo, hi, cap)
ALL_STEPS.append((out["d"], lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
short = (y_max - a @ out["d"]) / y_max
eps0, v3.RESTORE_EPS = v3.RESTORE_EPS, 1e-10        # the shortfall must be the ε-prox alone
short0 = (y_max - a @ v3.qp_step(g, [a], [(1.5 * y_max, 2 * y_max)], D, lo, hi, cap)["d"]) / y_max
v3.RESTORE_EPS = eps0
check("(e) unreachable ⇒ restore, a·d = LP max (up to the ε-prox)",
      out["mode"] == "restore" and -1e-12 <= short < 1e-4 and abs(short0) < 1e-12,
      f"relative shortfall {short:.1e} at ε=1e-6, {short0:.1e} at ε=1e-10")
xiJ = D * g - D * a * (a @ (D * g)) / (a @ (D * a))   # old engine tangent step: a·ξ_J = 0
check("  tooth: tangent (old null-space) step has a·d=0 < L", a @ xiJ < L)

# ================================================================ (f) infeasible only via row 2
g, A, _, D, lo, hi, cap = instance(np.random.default_rng(9), 40, 2, "engine")
lb, ub = boxes(lo, hi, cap)
y2max = np.maximum(A[1] * lb, A[1] * ub).sum()
W_band = (-0.01 * np.abs(A[0]) @ (ub - lb), 0.01 * np.abs(A[0]) @ (ub - lb))
lam_band = (1.2 * y2max, 1.3 * y2max)             # beyond the LP bound ⇒ pair infeasible
out = v3.qp_step(g, A, [W_band, lam_band], D, lo, hi, cap)
ALL_STEPS.append((out["d"], lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
yW = A[0] @ out["d"]
check("(f) λ row unreachable ⇒ ascent_dropped_row, width row held",
      out["mode"] == "ascent_dropped_row" and W_band[0] - 1e-9 <= yW <= W_band[1] + 1e-9
      and out["row_active"][1] is False and y2max < lam_band[0],
      f"mode {out['mode']}, a_W·d={yW:.3g} in [{W_band[0]:.3g},{W_band[1]:.3g}]")

# ================================================================ (d) cap + box always, (g) frozen
rng = np.random.default_rng(13)
for k in range(40):                                # extra restore / dropped-row cases
    g, A, bands, D, lo, hi, cap = instance(rng, 40, 1 + k % 2, "engine")
    bands = [(L + 3 * cap * np.abs(a).sum(), U + 4 * cap * np.abs(a).sum()) for a, (L, U) in zip(A, bands)]
    out = v3.qp_step(g, A, bands, D, lo, hi, cap)
    assert out["mode"] == "restore", out["mode"]  # bands sit beyond every row's reach
    ALL_STEPS.append((out["d"], lo, hi, cap, (hi - lo) < v3.FROZEN_SPAN_NM))
ok_box = all(in_box(d, lo, hi, cap) for d, lo, hi, cap, _ in ALL_STEPS)
check(f"(d) max|d| ≤ cap and lo ≤ d ≤ hi on all {len(ALL_STEPS)} steps (incl. 40 restore)", ok_box)
worst_cap, worst_drift = 0.0, 0.0
for seed in range(17, 37):                         # old engine step (review A2) on 20 instances
    g, A, _, D, lo, hi, cap = instance(np.random.default_rng(seed), 40, 2, "engine")
    Am = np.array(A)
    M = Am @ (D[:, None] * Am.T)
    hres = np.array([0.5, -0.5]) * cap * np.abs(Am).sum(1)
    xiJ = D * g - D * (Am.T @ np.linalg.solve(M, Am @ (D * g)))
    xiJ *= cap / np.abs(xiJ).max()                 # null part scaled to the cap ...
    xiC = -D * (Am.T @ np.linalg.solve(M, hres))
    xiC *= min(1.0, cap / np.abs(xiC).max())       # ... range part clamped to the cap, then added
    worst_cap = max(worst_cap, np.abs(xiJ + xiC).max() / cap)
    xiJc = np.clip(xiJ, *boxes(lo, hi, cap))       # ... then clipped to the bounds
    worst_drift = max(worst_drift, np.abs(Am @ xiJc).max() / (cap * np.abs(Am).sum(1)).min())
check("  tooth: old projection+clip step breaks the cap and the tangency",
      worst_cap > 1.01 and worst_drift > 1e-3,
      f"worst max|d_old|/cap={worst_cap:.2f}; clipped-tangent row drift {worst_drift:.1e}")
mv = max(np.abs(d[fz]).max() if fz.any() else 0.0 for d, lo, hi, cap, fz in ALL_STEPS)
check("(g) frozen params (D=1e-6, ±1e-3 nm) move < 1e-3 nm", mv <= 1e-3, f"max {mv:.1e} (holds by construction: box)")

# ================================================================ zero gradient
g, A, _, D, lo, hi, cap = instance(np.random.default_rng(19), 40, 1, "engine")
out = v3.qp_step(0 * g, A, [(-1.0, 1.0)], D, lo, hi, cap)
check("gT=0, band contains 0 ⇒ d=0 (minimum-norm feasible step)", np.abs(out["d"]).max() == 0 and out["tau"] == 1.0)

# ================================================================ radius_update table
TABLE = [  # cap, pred, meas, noise, active, expect_cap, expect_tag
    (10, 1e-5, 5e-5, 1e-5, 0.3, 10, "unresolved"),
    (10, 1e-3, -1e-3, 1e-5, 0.3, 5, "shrink"),
    (10, 1e-3, 1e-4, 1e-5, 0.3, 5, "shrink"),
    (3, 1e-3, 1e-4, 1e-5, 0.3, 2, "shrink"),        # clamped at cap_min
    (10, 1e-3, 2.5e-4, 1e-5, 0.3, 10, "keep"),      # ratio 0.25: not < 0.25, not ≥ 0.5
    (10, 1e-3, 4e-4, 1e-5, 0.3, 10, "keep"),
    (10, 1e-3, 5e-4, 1e-5, 0.3, 15, "grow"),        # ratio 0.5 edge
    (10, 1e-3, 2e-3, 1e-5, 0.3, 15, "grow"),        # ratio 2.0 edge
    (25, 1e-3, 1e-3, 1e-5, 0.3, 30, "grow"),        # clamped at cap_max
    (10, 1e-3, 1e-3, 1e-5, 0.0, 10, "keep"),        # not radius-limited ⇒ no growth
    (10, 1e-3, 3e-3, 1e-5, 0.3, 10, "keep"),        # ratio > 2
]
bad = [row for row in TABLE if v3.radius_update(*row[:5]) != (float(row[5]), row[6])]
check(f"radius_update: all {len(TABLE)} table rows (every branch + clamps + edges)", not bad, str(bad))

# ================================================================ hardening (GPT v3 review G1/G3/G5)
# status: a dual failure on a REACHABLE row set must not drop a row or restore
for seed in range(41, 141):                        # first instance with an ACTIVE row
    g, A, bands, D, lo, hi, cap = instance(np.random.default_rng(seed), 40, 2, "engine")
    ref = v3.qp_step(g, A, bands, D, lo, hi, cap)
    if ref["mode"] == "ascent" and any(ref["row_active"]):
        break
lb, ub = boxes(lo, hi, cap)
dual0, slsqp0 = v3._dual_solve, v3._slsqp_ascent
v3._dual_solve = lambda gg, *a: None if a[-1] == 0.0 else dual0(gg, *a)   # ascent dual "fails"
fb = v3.qp_step(g, A, bands, D, lo, hi, cap)
v3._slsqp_ascent = lambda *a: None                                          # ... and SLSQP too
ff = v3.qp_step(g, A, bands, D, lo, hi, cap)
yf0 = A[0] @ rng.uniform(lb, ub)
band_no0 = [(yf0 + 0.01 * abs(yf0), yf0 + 0.02 * abs(yf0)) if yf0 > 0 else
            (yf0 - 0.02 * abs(yf0), yf0 - 0.01 * abs(yf0))]                # excludes d = 0
ff1 = v3.qp_step(g, A[:1], band_no0, D, lo, hi, cap)
v3._dual_solve, v3._slsqp_ascent = dual0, slsqp0
tauR = ref["tau"]
check("status ok/dual on a normal instance; result carries viol per row",
      ref["status"] == "ok" and ref["solver"] == "dual" and len(ref["viol"]) == 2
      and row_viol(A, bands, ref["d"], cap) <= 1e-12
      and np.allclose(ref["viol"], [max(0.0, L - a @ ref["d"], a @ ref["d"] - U)
                                    for a, (L, U) in zip(A, bands)], rtol=0, atol=0),
      f"{ref['status']}/{ref['solver']} viol {ref['viol']}")
check("dual fails on reachable rows ⇒ SLSQP fallback, mode ascent, status ok, same optimum",
      fb["mode"] == "ascent" and fb["status"] == "ok" and fb["solver"] == "slsqp"
      and row_viol(A, bands, fb["d"], cap) <= 1e-7
      and abs(objective(g, D, tauR, fb["d"]) - objective(g, D, tauR, ref["d"]))
      <= 1e-6 * abs(objective(g, D, tauR, ref["d"])),
      f"Δobj rel {abs(objective(g, D, tauR, fb['d']) - objective(g, D, tauR, ref['d'])) / abs(objective(g, D, tauR, ref['d'])):.1e}")
check("both solvers fail ⇒ solver_failed, mode NOT downgraded, d feasible (0 in bands: scaled step)",
      ff["status"] == "solver_failed" and ff["mode"] == "ascent" and in_box(ff["d"], lo, hi, cap)
      and row_viol(A, bands, ff["d"], cap) <= 1e-9 and g @ ff["d"] >= 0, f"mode {ff['mode']}")
check("both solvers fail, band excludes 0 ⇒ LP feasible vertex, solver_failed, feasible",
      ff1["status"] == "solver_failed" and ff1["mode"] == "ascent" and in_box(ff1["d"], lo, hi, cap)
      and row_viol(A[:1], band_no0, ff1["d"], cap) <= 1e-9, f"viol {ff1['viol']}")
Ah_ = np.array(A) / (cap * np.abs(np.array(A)).sum(1))[:, None]
check("  tooth: those rows ARE LP-reachable (the old code would have dropped/restored)",
      v3._min_violation(Ah_, np.array([b[0] for b in bands]) / (cap * np.abs(np.array(A)).sum(1)),
                        np.array([b[1] for b in bands]) / (cap * np.abs(np.array(A)).sum(1)),
                        lb, ub) <= v3.FEAS_TOL)
# restoration: every returned quantity from the FINAL multipliers, incl. at the iteration limit
g, A, bands, D, lo, hi, cap = instance(np.random.default_rng(43), 40, 1, "engine")
lb, ub = boxes(lo, hi, cap)
far = [(bands[0][0] + 3 * cap * np.abs(A[0]).sum(), bands[0][1] + 4 * cap * np.abs(A[0]).sum())]
rs = v3.qp_step(g, A, far, D, lo, hi, cap)
mx0, v3.MAX_NEWTON = v3.MAX_NEWTON, 1
rt = v3.qp_step(g, A, far, D, lo, hi, cap)
v3.MAX_NEWTON = mx0
y_hi = np.maximum(A[0] * lb, A[0] * ub).sum()
cons_ok = all(np.allclose(r_["d"], np.clip(-D * (A[0] * r_["mu"][0]) / v3.RESTORE_EPS, lb, ub),
                          rtol=0, atol=1e-12 * cap)
              and abs(r_["viol"][0] - max(0.0, far[0][0] - A[0] @ r_["d"])) < 1e-12 * cap * np.abs(A[0]).sum()
              for r_ in (rs, rt))
check("restore: d / viol consistent with the FINAL μ (converged and MAX_NEWTON=1)", cons_ok,
      f"status {rs['status']}/{rt['status']}, viol {rs['viol'][0]:.4g} vs LP min {far[0][0] - y_hi:.4g}")
check("restore converged: status infeasible, viol = LP min violation (ε-prox)",
      rs["mode"] == "restore" and rs["status"] == "infeasible"
      and abs(rs["viol"][0] - (far[0][0] - y_hi)) <= 1e-3 * cap * np.abs(A[0]).sum())
check("  tooth: at the limit the stale (μ=0) d = 0 differs from the returned d",
      np.abs(rt["d"]).max() > 0 and rt["iters"] == 1)
# guards: every non-finite input ⇒ ValueError
g, A, bands, D, lo, hi, cap = instance(np.random.default_rng(47), 40, 2, "engine")
bad_args = {"gT": 0, "rows": 1, "bands": 2, "D": 3, "lo_step": 4, "hi_step": 5, "cap_nm": 6}
raised = []
for nm_, k in bad_args.items():
    args = [g.copy(), [a.copy() for a in A], [list(b) for b in bands], D.copy(), lo.copy(), hi.copy(), cap]
    if nm_ == "rows":
        args[1][0][3] = np.nan
    elif nm_ == "bands":
        args[2][1][1] = np.inf
    elif nm_ == "cap_nm":
        args[6] = np.nan
    else:
        args[k] = args[k].copy()
        args[k][5] = np.nan if nm_ != "lo_step" else -np.inf
    try:
        v3.qp_step(*args)
    except ValueError as e:
        raised.append(nm_ in str(e))
check("non-finite gT/rows/bands/D/lo/hi/cap ⇒ ValueError naming the input",
      raised == [True] * len(bad_args), str(raised))
# radius_update: never grow on a non-positive prediction
check("radius_update pred = meas = −0.01, active cap ⇒ keep (not grow)",
      v3.radius_update(10, -0.01, -0.01, 1e-5, 0.3) == (10.0, "keep")
      and v3.radius_update(10, -1e-3, 1e-3, 1e-5, 0.3) == (10.0, "keep"))
check("  tooth: the ratio test alone would have grown it (ratio 1, cap active)",
      0.5 <= (-0.01) / (-0.01) <= 2.0)
# peak3: tie jump documented; vertex outside the half-cell ⇒ sample fallback
Tt = np.array([0.0, 0.9, 0.9, 0.8])
check("peak3 tie (0,.9,.9,.8): left 1.0125 vs right 0.9125 (documented jump)",
      abs(v3.peak3(Tt, 1) - 1.0125) < 1e-12 and abs(v3.peak3(Tt, 2) - 0.9125) < 1e-12,
      f"{v3.peak3(Tt, 1):.4f} / {v3.peak3(Tt, 2):.4f}")
To = np.array([0.0, 0.8, 0.85])
go = autograd.grad(lambda x: v3.peak3(x, 1))(To)
check("peak3 |r| > ½ (i not nearest the max) ⇒ plain sample, one-hot grad, r = 0",
      v3.peak3(To, 1) == 0.8 and np.array_equal(go, np.eye(3)[1]) and v3.peak3_r(To, 1) == 0.0)
Bo, Do_ = To[2] - 2 * To[1] + To[0], To[2] - To[0]
check("  tooth: the raw parabola there extrapolates above every sample",
      To[1] - Do_ ** 2 / (8 * Bo) > To.max(), f"{To[1] - Do_ ** 2 / (8 * Bo):.4f}")


# ================================================================ DRIVER (V1-V7)
# 2026-10-05. The REAL eng.run_projected / make_fct_v2 / make_log_callback on
# campaign_te_s1.SPEC_V3 (296 params, ns2 plumbing + v3 + peak objective).
# Mocked: project.compute_fom/compute_gradient, the fields→gradient conversion
# (fixed gW / gTlo / gThi; gλ = gThi − gTlo at dTp = −1), and the log callback
# (writes the evals.jsonl rows + profile npz the real one writes). V7 runs the
# REAL callback against a fake fdtd.getresult (lumopt2 stubbed: only its
# BaseCallback import is needed). Same harness as gate_projection_local §11.
import contextlib, dataclasses, inspect, io, json, tempfile, types   # noqa: E401,E402
from runners.lumopt2_design import lumopt2_design as eng    # noqa: E402
from runners.lumopt2_design.campaign_te_s1 import SPEC_V3   # noqa: E402

SV = dataclasses.replace(SPEC_V3, label="gatev3", fwhm0_um=19.121,
                         wgp_target_um=19.121, max_iter=10)
bV = np.asarray(eng.param_bounds(SV), dtype=float)
DV = ((bV[:, 1] - bV[:, 0]) / 2.0) ** 2
P0 = np.asarray(eng.seed_params(SV), dtype=float)
NP_, W0, LAM0 = len(P0), 19.121, float(SV.scan_center_nm)
F0, SLV, CAPV = 0.70, float(SV.wgp_fom_slack), float(SV.wgp_step_max_nm)
W_LO, W_HI = eng.RHO_DN * W0 + SV.wgp_margin_um, eng.RHO_UP * W0 - SV.wgp_margin_um
DLAM = float(SV.wgp_v3_dlam_nm)
rv = np.random.default_rng(31)
gWv, gLov, gHiv, gTd = (rv.standard_normal(NP_) for _ in range(4))
gLamv = gHiv - gLov                                  # = −(gHi − gLo)/dTp, dTp = −1
q0 = v3.qp_step(gTd, [gWv, gLamv], [(W_LO - W0, W_HI - W0), (-DLAM, DLAM)],
                DV, bV[:, 0] - P0, bV[:, 1] - P0, CAPV)


def gT_pred(k_slack):
    """gT scaled so the it-0 v3 step predicts k_slack × slack (d is gT-scale free)."""
    return gTd * (k_slack * SLV / float(gTd @ q0["d"]))


XG = np.linspace(-40.0, 40.0, 801)
GAUSS = np.exp(-XG ** 2 / 32.0)


class FakeProject:
    def __init__(self, script, gT):
        self.script, self.n, self.gT = script, 0, gT
        self.fom = types.SimpleNamespace(gfields_W="W", gfields_Tlo="Tlo", gfields_Thi="Thi")
        self.fdtd_session = None
        g = {"W": gWv, "Tlo": gLov, "Thi": gHiv}
        self.parametrization = types.SimpleNamespace(
            compute_gradient_from_fields=lambda f, sess, p: g[f].copy())

    def compute_fom(self, p):
        self.n += 1
        f = self.script[self.n - 1]["fom"]
        return float(f(p) if callable(f) else f)

    def compute_gradient(self, p):
        return self.gT.copy()


class FakeLog:
    def __init__(self, spec, out_dir, script):
        self.spec, self.out_dir, self.script = spec, out_dir, script

    def on_function_eval(self, project, it, p, fom):
        e, lab = self.script[it], self.spec.label
        stale = bool(e.get("stale"))            # ineligible spectrum: no λ stencil
        self.spec._wg_dTp, self.spec._wg_lam_idx = (None, None) if stale else (-1.0, (0, 1))
        row = {"eval": int(it), "fom": float(fom), "params": [float(v) for v in p],
               "fwhm_env_um": e["W"], "lam_pk_nm": e.get("lam", LAM0),
               "softw_um": e["W"] - 0.4, "softw_adj_um": e.get("sw_adj", e["W"] - 0.3)}
        for k_src, k_row in (("cw", "cw_um_per_nm"), ("curved", "cw_curved"), ("twin", "twin_lam_nm")):
            if e.get(k_src) is not None:
                row[k_row] = e[k_src]
        with open(os.path.join(self.out_dir, f"{lab}_evals.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        pd = os.path.join(self.out_dir, "profiles")
        os.makedirs(pd, exist_ok=True)
        np.savez_compressed(os.path.join(pd, f"{lab}_ev{int(it):04d}.npz"),
                            x_um=XG, I=GAUSS, lam_pk_nm=LAM0)


def drive(script, gT, fn=None, **kw):
    """Run the real (or a patched) run_projected; artefacts + any exception."""
    spec = dataclasses.replace(SV, **kw)
    buf, exc = io.StringIO(), None
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(buf):
            try:
                (fn or eng.run_projected)(spec, FakeProject(script, gT),
                                          FakeLog(spec, td, script), td, P0)
            except Exception as e:                  # noqa: BLE001 - returned to the caller
                exc = e
        rd = lambda n: [json.loads(l) for l in open(os.path.join(td, n))]
        return {"out": buf.getvalue(), "ev": rd(f"{spec.label}_evals.jsonl"), "exc": exc,
                "pr": rd(f"{spec.label}_proj.jsonl"), "ost": eng._load_opt_state(td, spec.label)}


SRC_RP = inspect.getsource(eng.run_projected)


def patched(old, new):
    """run_projected with ONE source line replaced (the pre-fix behaviour)."""
    assert SRC_RP.count(old) == 1, old
    ns = dict(vars(eng))
    exec(compile(SRC_RP.replace(old, new), eng.__file__, "exec"), ns)
    return ns["run_projected"]


@contextlib.contextmanager
def grads(gW=None, gLo=None, gHi=None):
    """Temporarily swap the fake project's gW / gTlo / gThi CONTENTS in place."""
    saved = [v.copy() for v in (gWv, gLov, gHiv)]
    for v, new in zip((gWv, gLov, gHiv), (gW, gLo, gHi)):
        if new is not None:
            v[:] = new
    try:
        yield
    finally:
        for v, old in zip((gWv, gLov, gHiv), saved):
            v[:] = old


def steps_vs_cap(r):
    """(worst max|Δp| − logged cap_nm, all params inside bounds) over every row."""
    ev = [np.asarray(e["params"]) for e in r["ev"]]
    acc, worst = ev[0], -np.inf
    for k in range(len(ev) - 1):
        if not r["pr"][k]["phase"].endswith("-retry"):
            acc = ev[k]
        worst = max(worst, float(np.max(np.abs(ev[k + 1] - acc))) - r["pr"][k]["cap_nm"])
    inb = all(np.all(v >= bV[:, 0]) and np.all(v <= bV[:, 1]) for v in ev)
    return worst, inb


# -- V1 PLUMBING: the real v3 fct on the FLAT x = [T(λ_0..n−1), softW]
wl_f = np.linspace(eng.C0 / ((LAM0 + 5.0) * 1e-9), eng.C0 / ((LAM0 - 5.0) * 1e-9), 501)
wl_nm = eng.C0 / wl_f * 1e9                          # frequency-uniform, λ-DESCENDING (Lumerical order)
f0 = wl_f[250] + 0.3 * (wl_f[1] - wl_f[0])
gam_f = eng.C0 / (LAM0 * 1e-9) ** 2 * 0.5e-9         # HWHM 0.5 nm ⇒ ~50 samples / FWHM
T_l = 0.9 / (1.0 + ((wl_f - f0) / gam_f) ** 2)
x1 = np.r_[T_l, 18.7]
fct3 = eng.make_fct_v2(list(wl_nm), SV)
j3 = np.asarray(autograd.grad(fct3)(x1))
i3 = int(np.argmin(np.abs(wl_nm - eng.measure_peak(wl_nm, T_l)[0])))
r3 = v3.peak3_r(T_l, i3)
w3 = np.zeros(502)
w3[i3 - 1:i3 + 2] = [r3 * (r3 - 1) / 2, 1 - r3 * r3, r3 * (r3 + 1) / 2]
check("V1 v3 fct: 3 nonzeros = parabola weights at peak±1, softW slot 0, value = peak3",
      np.count_nonzero(j3) == 3 and np.abs(j3 - w3).max() < 1e-12 and j3[501] == 0.0
      and abs(float(fct3(x1)) - 0.9) < 2e-6,
      f"i_pk {i3} r {r3:+.3f} value−T0 {float(fct3(x1)) - 0.9:+.1e}")
j_sm = np.asarray(autograd.grad(eng.make_fct_v2(list(wl_nm), dataclasses.replace(
    SV, wgp_v3_peak=False)))(x1))
check("  tooth: non-v3 fct = windowed softmax (many nonzeros)",
      np.count_nonzero(j_sm) > 100, f"{np.count_nonzero(j_sm)} nonzero entries")

# -- V2 STEP BOUNDS + v3 rec fields: 5 accepted evals, linear fom (meas = pred)
gT10 = gT_pred(10.0)
lin = lambda p: F0 + float(gT10 @ (np.asarray(p) - P0))
r2 = drive([dict(fom=lin, W=W0)] * 5, gT10, max_iter=5)
wv2, inb2 = steps_vs_cap(r2)
pr2 = r2["pr"]
f_ok = all(q.get("v3_mode") and len(q["v3_mu"]) == 2 and "v3_active_cap" in q
           and W_LO - q["W"] - 1e-7 <= q["v3_pred_rows"][0] <= W_HI - q["W"] + 1e-7
           and abs(q["v3_pred_rows"][1]) <= DLAM + 1e-7 for q in pr2)
check("V2 every step: max|dp| <= logged cap_nm, inside param_bounds (5 evals)",
      len(pr2) == 5 and wv2 <= 1e-9 and inb2,
      f"worst excess {wv2:.2e}; caps {[q['cap_nm'] for q in pr2]}")
check("V2 rows carry v3_mode / v3_mu(2) / v3_active_cap; v3_pred_rows inside the bands",
      f_ok, str([(q.get("v3_mode"), [round(v, 4) for v in q.get("v3_pred_rows", [])])
                 for q in pr2]))

# -- V3 TOTAL WIDTH ROW: cw written ⇒ the width prediction is (gW + cw·gλ)·d
r3a = drive([dict(fom=F0, W=W0, cw=0.3)] * 2, gT10, max_iter=2)
d3 = np.asarray(r3a["ev"][1]["params"]) - np.asarray(r3a["ev"][0]["params"])
p_tot, p_fix = float((gWv + 0.3 * gLamv) @ d3), float(gWv @ d3)
check("V3 cw=0.3 in the rows ⇒ v3_pred_rows[0] == (gW + 0.3·gλ)·d",
      abs(r3a["pr"][0]["v3_pred_rows"][0] - p_tot) < 1e-9 and r3a["pr"][0]["cw"] == 0.3
      and abs(p_tot - p_fix) > 1e-3,
      f"logged {r3a['pr'][0]['v3_pred_rows'][0]:+.6f} total {p_tot:+.6f} fixed {p_fix:+.6f}")
r3b = drive([dict(fom=F0, W=W0)] * 2, gT10, max_iter=2)
d3b = np.asarray(r3b["ev"][1]["params"]) - np.asarray(r3b["ev"][0]["params"])
check("  tooth: cw missing ⇒ v3_pred_rows[0] == gW·d (fixed-λ row)",
      abs(r3b["pr"][0]["v3_pred_rows"][0] - float(gWv @ d3b)) < 1e-9
      and r3b["pr"][0]["cw"] is None)

# -- V4 RESTORE: measured W 0.6 µm BELOW the band
W4 = W_LO - 0.6
r4 = drive([dict(fom=F0, W=W4)], gT10, max_iter=1)
q4 = r4["pr"][0]
check("V4 reachable: phase v3-*, width row reaches the band (pred >= W_lo − W)",
      q4["phase"].startswith("v3-") and q4["v3_pred_rows"][0] >= (W_LO - W4) - 1e-7,
      f"{q4['phase']} pred {q4['v3_pred_rows'][0]:+.4f} need {W_LO - W4:+.4f}")
gW_keep = gWv.copy()
gWv *= 1e-4                                          # weak width row: the band is out of reach
r4b = drive([dict(fom=F0, W=W4)], gT10, max_iter=1)
gWv[:] = gW_keep
q4b = r4b["pr"][0]
check("V4 unreachable: mode restore, model width moves UP (pred > 0)",
      q4b["v3_mode"] == "restore" and q4b["phase"] == "v3-restore"
      and q4b["v3_pred_rows"][0] > 0, f"{q4b['phase']} pred {q4b['v3_pred_rows'][0]:+.3e}")

# -- V5 RADIUS RULE on the accept after it-0 (cap 10)
# (a') full rows: both bands bind, NO component on the cap (active_cap 0), but
# τ ∝ cap, so a larger radius still changes d (GPT v3 review G5). The engine's
# 1.5×-radius probe (rec v3_gain_1p5) must then license "grow".
r5k = drive([dict(fom=lin, W=W0)] * 2, gT10, max_iter=2)
g15k, prk = r5k["pr"][1].get("v3_gain_1p5"), r5k["pr"][1].get("dT_pred_prev")
check("V5 (a') rows bind, cap inactive, probe gain > 5 % of pred, meas = pred ⇒ grow",
      r5k["pr"][0]["v3_active_cap"] == 0.0 and g15k > 0.05 * abs(prk)
      and (r5k["pr"][1].get("v3_radius"), r5k["ost"]["cap_nm"]) == ("grow", 15.0),
      f"gain_1p5 {g15k:.3e} vs pred {prk:.3e} -> {r5k['pr'][1].get('v3_radius')}")
# (a'') the step is fully determined (one-hot gT on corr[5]; width row = c·e_5
# caps d_5 at 3 nm < cap): the 1.5× probe gains nothing ⇒ "keep"
J5 = 5
e5 = np.zeros(NP_)
e5[J5] = 1.0
gT1h = e5 * (10.0 * SLV / 3.0)                       # pred = 10 slack at the 3 nm row limit
lin1h = lambda p: F0 + float(gT1h @ (np.asarray(p) - P0))
with grads(gW=e5 * ((W_HI - W0) / 3.0), gLo=np.zeros(NP_), gHi=np.zeros(NP_)):
    r5d = drive([dict(fom=lin1h, W=W0)] * 2, gT1h, max_iter=2)
g15d = r5d["pr"][1].get("v3_gain_1p5")
st5d = float(np.max(np.abs(np.asarray(r5d["ev"][1]["params"]) - P0)))
check("V5 (a'') row-determined step: probe gain ~0 ⇒ keep, cap 10",
      abs(g15d) <= 1e-12 and (r5d["pr"][1].get("v3_radius"), r5d["ost"]["cap_nm"]) == ("keep", 10.0)
      and abs(st5d - 3.0) < 1e-6, f"gain_1p5 {g15d:.1e}, step {st5d:.6f}")
g_keep = [g.copy() for g in (gWv, gLov, gHiv)]
for g in (gWv, gLov, gHiv):
    g *= 1e-3
r5a = drive([dict(fom=lin, W=W0)] * 2, gT10, max_iter=2)
for g, k in zip((gWv, gLov, gHiv), g_keep):
    g[:] = k
r5b = drive([dict(fom=F0, W=W0)] * 2, gT10, max_iter=2)
gT04 = gT_pred(0.4)
r5c = drive([dict(fom=F0, W=W0), dict(fom=F0 + 0.1 * SLV, W=W0)], gT04, max_iter=2)
tags5 = [(r["pr"][1].get("v3_radius"), r["ost"]["cap_nm"]) for r in (r5a, r5b, r5c)]
check("V5 (a) meas = pred, cap active ⇒ grow 10 → 15", tags5[0] == ("grow", 15.0)
      and r5a["pr"][0]["v3_active_cap"] > 0, str(tags5[0]))
check("V5 (b) meas 0 vs pred 10×slack, accepted ⇒ shrink 10 → 5", tags5[1] == ("shrink", 5.0)
      and not r5b["pr"][1]["phase"].endswith("-retry"), str(tags5[1]))
check("V5 (c) |pred| < 2×slack ⇒ unresolved, cap 10 kept", tags5[2] == ("unresolved", 10.0),
      str(tags5[2]))

# -- V6 RETRIES: noise reject ⇒ a DIFFERENT, smaller QP trial; stop classification
NOISE3 = [dict(fom=F0 - 1.2 * SLV)] * 3
r6 = drive([dict(fom=F0, W=W0)] + [dict(e, W=W0) for e in NOISE3], gT04, max_iter=4)
e6 = [np.asarray(e["params"]) for e in r6["ev"]]
check("V6 noise reject ⇒ retry is a different QP trial with step <= cap/2",
      r6["pr"][1].get("noise_reject") and float(np.max(np.abs(e6[2] - e6[1]))) > 1e-6
      and float(np.max(np.abs(e6[2] - e6[0]))) <= CAPV / 2 + 1e-9
      and steps_vs_cap(r6)[0] <= 1e-9,
      f"trial steps {[round(float(np.max(np.abs(v - e6[0]))), 6) for v in e6[1:]]}")
check("V6 W INSIDE the band, 3 noise rejects ⇒ CONVERGED WITHIN NOISE",
      len(r6["ev"]) == 4 and "CONVERGED WITHIN NOISE" in r6["out"]
      and "restoration unresolved" not in r6["out"])
W6o = W_LO - 0.05
r6o = drive([dict(fom=F0, W=W6o)] + [dict(e, W=W6o) for e in NOISE3], gT04, max_iter=4)
check("V6 W OUTSIDE the band (hv>0), 3 noise rejects ⇒ STOPPED restoration unresolved, NOT converged",
      len(r6o["ev"]) == 4 and "restoration unresolved, NOT converged" in r6o["out"]
      and "CONVERGED WITHIN NOISE" not in r6o["out"]
      and all(r6o["pr"][k].get("noise_reject") for k in (1, 2, 3)),
      f"evals {len(r6o['ev'])}")
# restoration-only (gT = 0, W just below the band): the QP step sits under every cap
z6 = np.zeros(NP_)


def dup_run(dW, n):
    return drive([dict(fom=F0, W=W_LO - dW)] + [dict(fom=F0 - 1.2 * SLV, W=W_LO - dW)] * (n - 1),
                 z6, max_iter=n)


r6d = dup_run(0.01, 8)
d6 = [float(np.max(np.abs(np.asarray(e["params"]) - e6[0]))) for e in r6d["ev"][1:]]
check("V6 dup guard under v3: trials s, s/2, s/4 distinct; out of band ⇒ STOPPED, not CONVERGED",
      len(d6) == 3 and np.allclose(d6, [d6[0], d6[0] / 2, d6[0] / 4], rtol=1e-6, atol=0)
      and d6[0] < 1.0 and all(r6d["pr"][k].get("dup_retry") for k in (1, 2))
      and "restoration unresolved, NOT converged" in r6d["out"]
      and "CONVERGED WITHIN NOISE" not in r6d["out"],
      f"deliv {[f'{v:.4g}' for v in d6]}")
r6s = dup_run(0.01 * 1e-9 / d6[0], 8)                # s ≈ 1e-9 nm ⇒ halving collapses
check("V6 stalled path under v3: STOPPED, no CONVERGED",
      any(q.get("stalled") for q in r6s["pr"]) and "[proj] STOPPED" in r6s["out"]
      and "CONVERGED WITHIN NOISE" not in r6s["out"], f"evals {len(r6s['ev'])}")

# -- V8 WIDTH REJECT: higher fom, but the measured width LEAVES the band
s8 = [dict(fom=F0, W=W0), dict(fom=F0 + 10 * SLV, W=W_HI + 0.05)]
r8 = drive(s8, gT10, max_iter=2)
q8 = r8["pr"][1]
check("V8 v3: width leaves the band ⇒ rejected (v3_width_reject), not noise, cap halved 10 → 5",
      "v3_width_reject" in q8 and q8["phase"].endswith("-retry") and "noise_reject" not in q8
      and r8["ost"]["cap_nm"] == 5.0,
      f"{q8['phase']} hv {q8.get('v3_width_reject')} cap {r8['ost']['cap_nm']}")
r8t = drive(s8, gT10, max_iter=2, wgp_v3=False, wgp_v3_peak=False)
check("  tooth: non-v3 spec ACCEPTS the same trial (FOM buys the width violation)",
      not r8t["pr"][1]["phase"].endswith("-retry"), r8t["pr"][1]["phase"])

# -- V9 DEGRADED WIDTH ROW: cw missing / curved / no gλ ⇒ λ bound halved, loud
cases9 = {"ok": dict(cw=0.3), "missing": dict(), "curved": dict(cw=0.3, curved=True),
          "no_glam": dict(cw=0.3, stale=True)}
res9 = {}
for st_, extra in cases9.items():
    r9 = drive([dict(fom=F0, W=W0, **extra)], gT10, max_iter=1)
    q9 = r9["pr"][0]
    lam_row = q9["v3_pred_rows"][1] if len(q9["v3_pred_rows"]) > 1 else None
    res9[st_] = (q9.get("v3_cw_state"), "DEGRADED WIDTH ROW" in r9["out"], lam_row)
ok9 = all(res9[k][0] == k and res9[k][1] for k in ("missing", "curved", "no_glam"))
check("V9 degraded states logged + '★v3 DEGRADED WIDTH ROW' printed (missing/curved/no_glam)",
      ok9 and res9["no_glam"][2] is None, str({k: v[:2] for k, v in res9.items()}))
check("V9 degraded λ-row prediction <= ½·dlam; ok state uses the FULL bound (and no warning)",
      all(abs(res9[k][2]) <= DLAM / 2 + 1e-7 for k in ("missing", "curved"))
      and res9["ok"][0] == "ok" and not res9["ok"][1] and abs(res9["ok"][2]) > DLAM / 2 + 1e-3,
      str({k: (round(v[2], 6) if v[2] is not None else None) for k, v in res9.items()}))

# -- V10 RECENTER: an ACCEPTED v3 point beyond recenter_nm from the scan centre
s10 = [dict(fom=F0, W=W0, lam=LAM0 + SV.recenter_nm + 0.1)]
r10 = drive(s10, gT10, max_iter=3)
check("V10 accepted peak > recenter_nm off centre ⇒ RecenterNeeded AFTER the proj row is logged",
      isinstance(r10["exc"], eng.RecenterNeeded) and len(r10["pr"]) == 1
      and not r10["pr"][0]["phase"].endswith("-retry"), repr(r10["exc"]))
r10t = drive(s10, gT10, max_iter=1, wgp_v3=False, wgp_v3_peak=False)
check("  tooth: non-v3 driver does NOT recenter (only the callback's best-FOM guard did)",
      r10t["exc"] is None and len(r10t["pr"]) == 1)

# -- V11 BROYDEN with the twin λ moving: secant on the FIXED-λ part only
TW0, DTW, CW11 = LAM0 - 0.02, 0.04, 0.3
s11 = [dict(fom=lin, W=W0, cw=CW11, twin=TW0, sw_adj=18.80),
       dict(fom=lin, W=W0, cw=CW11, twin=TW0 + DTW, sw_adj=18.83)]
r11 = drive(s11, gT10, max_iter=2)
q11, e11 = r11["pr"][1], r11["ev"]
dp11 = np.asarray(e11[1]["params"]) - np.asarray(e11[0]["params"])
want11 = (18.83 - 18.80 - CW11 * DTW) - float(gWv @ dp11)
res11 = q11.get("broyden_dW_resid", np.nan)
check("V11 reused row: broyden_dW_resid == (Δsoftw_adj − cw·Δtwin_λ) − gW·Δp",
      q11.get("gw_reused") == 1 and abs(res11 - want11) < 1e-9,
      f"logged {res11} want {want11:.9f}")
r11t = drive(s11, gT10, max_iter=2, fn=patched(
    'dW_ -= float(acc["cw"]) * (row["twin_lam_nm"]', 'dW_ -= 0.0 * (row["twin_lam_nm"]'))
res11t = r11t["pr"][1].get("broyden_dW_resid", np.nan)
check("  tooth: without the correction the residual differs by cw·Δtwin_λ = 0.012",
      abs((res11t - res11) - CW11 * DTW) < 1e-9, f"uncorrected {res11t:.9f}")

# -- V12 STALE QP DIAGNOSTICS after an accepted duplicate-halved retry. Cap at
# its 2 nm floor: an ordinary reject keeps 2 nm, the retry QP re-delivers the
# rejected trial ⇒ halved ⇒ accepted. Weak rows (×1e-3) so the cap, not the
# rows, limits the step (active_cap > 0 in the stale record — with the full
# rows every step here is row-limited and the stale record could not grow).
gT50 = gT_pred(50.0)
lin50 = lambda p: F0 + float(gT50 @ (np.asarray(p) - P0))
s12 = [dict(fom=lin50, W=W0), dict(fom=F0 - 20 * SLV, W=W0), dict(fom=lin50, W=W0)]
weak = dict(gW=gWv * 1e-3, gLo=gLov * 1e-3, gHi=gHiv * 1e-3)
with grads(**weak):
    r12 = drive(s12, gT50, max_iter=3, wgp_step_max_nm=2.0)
q12 = r12["pr"][2]
check("V12 accepted halved retry: radius rule does not grow from the stale QP record",
      r12["pr"][1].get("dup_retry") and not q12["phase"].endswith("-retry")
      and q12.get("v3_radius") in ("keep", "unresolved", "shrink"),
      f"dup {r12['pr'][1].get('dup_retry')} tag {q12.get('v3_radius')} "
      f"pred {q12.get('dT_pred_prev')} meas {q12.get('dT_meas')}")
with grads(**weak):
    r12t = drive(s12, gT50, max_iter=3, wgp_step_max_nm=2.0, fn=patched(
        "v3_last = dict(v3_last, active_cap=0.0, gain_1p5=0.0, halved=True)", "pass"))
check("  tooth: pre-fix (stale record kept) GROWS the radius after the halved accept",
      r12["pr"][0]["v3_active_cap"] > 0 and r12t["pr"][2].get("v3_radius") == "grow",
      f"{r12t['pr'][2].get('v3_radius')}, it-0 active_cap {r12['pr'][0]['v3_active_cap']:.3g}")

# -- V7 the REAL log callback's c_W block, fake fdtd.getresult
LAMPK7 = float(wl_nm[250])
RATE7 = 0.3                                          # µm/nm: profile FWHM grows linearly with λ
x7 = np.arange(-50.0, 50.0 + 1e-9, 0.1)
fw7 = W0 + RATE7 * (wl_nm - LAMPK7)
I7 = np.exp(-4.0 * np.log(2.0) * x7[:, None] ** 2 / fw7[None, :] ** 2)      # (x, λ)
E7 = np.zeros((x7.size, 2, wl_nm.size, 3))
E7[:, :, :, 0] = np.sqrt(I7)[:, None, :]
T7 = 0.9 / (1.0 + ((wl_f - eng.C0 / (LAMPK7 * 1e-9)) / gam_f) ** 2)
FIELD7 = {"lambda": wl_nm * 1e-9, "E": E7, "x": x7 * 1e-6, "y": np.array([-0.1e-6, 0.1e-6])}


class FakeFdtd:
    def __init__(self, fail_field_call=None):
        self.n_field, self.fail = 0, fail_field_call

    def getresult(self, mon, key):
        if mon == "FDTD::ports::Port_2":
            return {"S": np.sqrt(T7).astype(complex), "lambda": wl_nm * 1e-9}
        if mon == "FDTD::ports::Port_1":
            return {"S": np.full(wl_nm.size, 0.2 + 0j), "lambda": wl_nm * 1e-9}
        if mon == "field_profile":
            self.n_field += 1
            if self.n_field == self.fail:
                raise RuntimeError("synthetic getresult failure")
            return FIELD7
        raise KeyError(mon)


def real_cb_row(fdtd):
    stub = types.ModuleType("lumopt2.utils.callbacks")
    stub.BaseCallback = object
    saved = {k: sys.modules.get(k) for k in ("lumopt2", "lumopt2.utils", "lumopt2.utils.callbacks")}
    sys.modules.update({"lumopt2": types.ModuleType("lumopt2"),
                        "lumopt2.utils": types.ModuleType("lumopt2.utils"),
                        "lumopt2.utils.callbacks": stub})
    try:
        with tempfile.TemporaryDirectory() as td:
            cb = eng.make_log_callback(SV, td, lmpt=object())
            proj = types.SimpleNamespace(load_forward_results=lambda: None,
                                         fdtd_session=types.SimpleNamespace(fdtd=fdtd))
            with contextlib.redirect_stdout(io.StringIO()):
                cb.on_function_eval(proj, 0, P0, 0.9)
            return json.loads(open(os.path.join(td, f"{SV.label}_evals.jsonl")).readline())
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


row7 = real_cb_row(FakeFdtd())
cw7 = row7.get("cw_um_per_nm")
check("V7 REAL callback: cw_um_per_nm within 5 % of the profile's dFWHM/dλ = 0.3",
      cw7 is not None and abs(cw7 - RATE7) / RATE7 < 0.05 and "diag_error" not in row7,
      f"cw {cw7} curved {row7.get('cw_curved')} lam_pk {row7.get('lam_pk_nm')} "
      f"fwhm_nm {row7.get('fwhm_nm')} err {row7.get('diag_error') or row7.get('cw_error')}")
row7e = real_cb_row(FakeFdtd(fail_field_call=2))    # 1st read = profile_line, 2nd = the c_W block
check("V7 getresult raising in the c_W block ⇒ row cw_error, no exception, eval logged",
      "cw_error" in row7e and "cw_um_per_nm" not in row7e and "diag_error" not in row7e
      and row7e.get("lam_pk_nm") is not None, f"cw_error {row7e.get('cw_error')!r}")

print("V3 LOCAL GATE: ALL PASS" if not FAILS else f"V3 LOCAL GATE: FAIL ({len(FAILS)}): {FAILS}")
sys.exit(1 if FAILS else 0)
