"""v3 optimizer step: the pure-math core (numpy / autograd / scipy only).

Study dir: runners/lumopt2_design/ | Created 2026-10-05 | zero GPU, no lumapi.
Purpose: the four math pieces of the user-approved "v3" step. Design from the
GPT review docs/ask_gpt_followup_2026-10-05_answer.md (F2-F5) and our check
docs/fom_linewidth_bias_check_2026-10-05.txt:
  peak3 / peak3_r  3-point parabola peak on a frequency-uniform grid (F3).
                   Bias -1.2e-6 at 50 samples/FWHM, where the power-mean FOM
                   rewards broadening by +0.075 T0-equivalent.
  cw_from_widths   c_W = dW/dλ at fixed geometry, from neighbouring recorded λ
                   of ONE forward solve (F4 total width row g_W + c_W g_λ).
  qp_step          ONE bounded composite step: ascent, width/λ bands, bounds
                   and a single inf-norm radius, solved exactly through its
                   ≤2-dim dual. It replaces projection-then-clipping (review A2:
                   cap 10 delivered a 19.975 nm step). The output is never post-clipped.
  radius_update    trust-radius rule on an accepted step (F2).
Gate: runners/lumopt2_design/gates/gate_v3_local.py
"""
import numpy as np
from autograd.tracer import getval
from scipy.optimize import linprog, minimize

FROZEN_SPAN_NM = 0.01   # bound span below this = frozen param (engine freezes at ±1e-3 nm)
RESTORE_EPS = 1e-6      # weight of the proximal term that makes restoration unique
FEAS_TOL = 1e-9         # scaled min-max band violation still counted as reachable
KKT_TOL = 1e-6          # scaled dual residual above this after the solve ⇒ infeasible
MAX_NEWTON = 100


# ---------------------------------------------------------------- F3 peak value
def peak3(T, i_pk):
    """Peak value T* = T0 − D²/(8B) of the parabola through T[i-1], T[i], T[i+1].

    T must be sampled UNIFORMLY IN FREQUENCY (either order: reversing the grid
    flips D and r but leaves T* unchanged). i_pk is the caller's stop-gradient
    choice. autograd gives the weights r(r−1)/2, 1−r², r(r+1)/2 (r = −D/2B).
    The vertex slope is zero, so dr/dp does not enter.
    Fallback: i at an edge, or B ≥ 0 (not a local max) → plain T[i], gradient
    one-hot. A parabola that is not concave has no peak to interpolate.
    Fallback also when |r| > ½ (+1e-9): the vertex lies outside the half-cell
    around i, so i is NOT the sample nearest the true maximum (a neighbour is
    higher). The parabola would then EXTRAPOLATE (e.g. (0, .8, .85) at i=1
    gives 0.92 > every sample), so the plain sample is returned instead. At a
    sampled maximum |r| ≤ ½ always, so a well-resolved peak never falls back.
    Known discontinuity (GPT review G3): a TIE switches the index and the
    value jumps, e.g. (0, .9, .9, .8) gives 1.0125 at i=1 but 0.9125 at i=2
    (|r| = ½ exactly at both). Smooth resolved peaks jump far less (gate).
    """
    i = int(i_pk)
    if i <= 0 or i >= T.shape[0] - 1:
        return T[i]
    Tm, T0, Tp = T[i - 1], T[i], T[i + 1]
    B = Tp - 2.0 * T0 + Tm
    if getval(B) >= 0.0:
        return T0
    D = Tp - Tm
    if abs(getval(D) / (2.0 * getval(B))) > 0.5 + 1e-9:
        return T0
    return T0 - D * D / (8.0 * B)


def peak3_r(T, i_pk):
    """Vertex offset r = −D/(2B) in grid steps (|r| ≤ ½ at a sampled max); 0 on
    every peak3 fallback (edge, B ≥ 0, |r| > ½)."""
    T = np.asarray(T, dtype=float)
    i = int(i_pk)
    if i <= 0 or i >= T.size - 1:
        return 0.0
    B = T[i + 1] - 2.0 * T[i] + T[i - 1]
    if B >= 0.0:
        return 0.0
    r = float(-(T[i + 1] - T[i - 1]) / (2.0 * B))
    return 0.0 if abs(r) > 0.5 + 1e-9 else r


# ---------------------------------------------------------------- F4 c_W
def cw_from_widths(lam_nm, w_um):
    """Least-squares dW/dλ (µm/nm) from 3-7 (λ, softW) samples at FIXED geometry.

    Returns (slope, curved). curved=True when a quadratic fit's curvature
    changes the slope across the stencil by >30 % of the slope (|2·c2·h| >
    0.3|slope|, h = half-span), or moves the centre slope by >30 %. At that
    curvature a linear c_W is not trustworthy. With symmetric sampling the
    centre slope of the quadratic EQUALS the linear LS slope, so the
    across-stencil change is the test that can actually fire.
    """
    lam, w = np.asarray(lam_nm, dtype=float), np.asarray(w_um, dtype=float)
    assert lam.size == w.size and 3 <= lam.size <= 7, "need 3-7 samples"
    x = lam - lam.mean()
    slope = np.polyfit(x, w, 1)[0]
    c2, c1, _ = np.polyfit(x, w, 2)
    h = np.abs(x).max()
    resolvable = abs(c2) * h * h > 1e-12 * max(np.abs(w).max(), 1e-300)   # not round-off
    curved = resolvable and (abs(2.0 * c2 * h) > 0.3 * abs(slope)
                             or abs(c1 - slope) > 0.3 * abs(slope))
    return float(slope), bool(curved)


# ---------------------------------------------------------------- F2 bounded step
# Scaled rows â_k = a_k/t_k, bands [L̂,Û] = [L,U]/t_k. Primal (min form):
#   min_d (1/2τ)Σ d²/D − g·d   s.t. lo ≤ d ≤ hi,  L̂ ≤ Âd ≤ Û.
# Dual (μ ∈ R^m free; μ_k>0 prices Û_k, μ_k<0 prices L̂_k):
#   φ(μ) = max_{d∈box}[(g − Âᵀμ)·d − (1/2τ)Σd²/D] + Σ_k (Û_k μ_k⁺ − L̂_k μ_k⁻) + (κ/4)|μ|²
#   d(μ) = clip(τ·D·(g − Âᵀμ), lo, hi),   ∂φ/∂μ_k = s_k − â_k·d(μ) + κμ_k/2.
# κ=0 is the ascent problem. κ=1 with g=0 is restoration, since dist²(y,[L,U])
# has conjugate σ_[L,U](μ) + μ²/4, so the same solver minimises Σ viol² + prox.
def _pgrad(mu, y, Lh, Uh, kappa):
    """Pseudo-gradient of φ and the orthant sign per row (0 = row optimal at μ_k=0)."""
    G, xi = np.zeros_like(mu), np.zeros_like(mu)
    for k in range(mu.size):
        up = Uh[k] - y[k] + 0.5 * kappa * mu[k]
        dn = Lh[k] - y[k] + 0.5 * kappa * mu[k]
        if mu[k] > 0 or (mu[k] == 0 and up < 0):
            G[k], xi[k] = up, 1.0
        elif mu[k] < 0 or (mu[k] == 0 and dn > 0):
            G[k], xi[k] = dn, -1.0
    return G, xi


def _dual_solve(g, Ah, Lh, Uh, D, lo, hi, tau, kappa):
    """Orthant-wise semismooth Newton on φ with an EXACT line search.

    φ is convex and piecewise quadratic, so along a direction its slope is
    monotone and piecewise linear, and bisection finds the exact minimiser.
    With the right active set the full Newton step is exact, so the solve
    terminates finitely. Returns (μ, d, G, iters), or None if φ is unbounded
    (the bands cannot be met).
    """
    m = Lh.size
    mu = np.zeros(m)
    for it in range(1, MAX_NEWTON + 1):
        z = tau * D * (g - Ah.T @ mu)
        d = np.clip(z, lo, hi)
        G, xi = _pgrad(mu, Ah @ d, Lh, Uh, kappa)
        if m == 0 or np.abs(G).max() <= 1e-13:
            return mu, d, G, it
        F = (z > lo) & (z < hi)                       # unclipped params carry curvature
        H = tau * (Ah[:, F] * D[F]) @ Ah[:, F].T + 0.5 * kappa * np.eye(m)
        free = [k for k in range(m) if xi[k] != 0]
        while True:   # a row leaving μ=0 must move into its own orthant (≤1 drop for m≤2)
            p = np.zeros(m)
            sol = np.linalg.lstsq(H[np.ix_(free, free)], -G[free], rcond=1e-13)[0]
            if not np.all(np.isfinite(sol)) or sol @ G[free] >= 0:
                sol = -G[free]                        # singular curvature → steepest descent
            p[free] = sol
            bad = [k for k in free if mu[k] == 0 and p[k] * xi[k] <= 0]
            if not bad:
                break
            free = [k for k in free if k not in bad]
        s = np.where(xi > 0, Uh, Lh)

        def slope(t):
            mut = mu + t * p
            dt = np.clip(tau * D * (g - Ah.T @ mut), lo, hi)
            return p @ (s - Ah @ dt + 0.5 * kappa * mut)

        lim = [(-mu[k] / p[k], k) for k in range(m) if mu[k] * p[k] < 0]
        tmax, klim = min(lim) if lim else (np.inf, -1)
        a, b = 0.0, min(1.0, tmax)
        while slope(b) < 0 and b < tmax:              # bracket the 1-D minimiser
            a, b = b, min(2.0 * b, tmax)
            if b > 1e30:
                return None                           # dual unbounded ⇒ bands unreachable
        if slope(b) < 0:                              # minimiser past the μ_k=0 kink: stop on it
            mu = mu + tmax * p
            mu[klim] = 0.0
            continue
        for _ in range(200):
            c = 0.5 * (a + b)
            if b - a <= 1e-16 * b or c in (a, b):
                break
            a, b = (c, b) if slope(c) < 0 else (a, c)
        mu = mu + b * p
    # iteration limit: d and G above belong to the PREVIOUS μ — recompute both
    # from the final multipliers so every returned quantity is consistent
    d = np.clip(tau * D * (g - Ah.T @ mu), lo, hi)
    G, _ = _pgrad(mu, Ah @ d, Lh, Uh, kappa)
    return mu, d, G, MAX_NEWTON


def _min_violation(Ah, Lh, Uh, lo, hi):
    """Smallest achievable max scaled band violation over the box (0 ⇒ reachable)."""
    m = Lh.size
    if m == 0:
        return 0.0
    if m == 1:      # exact: the row's reach over the box is an interval
        y_lo, y_hi = np.minimum(Ah[0] * lo, Ah[0] * hi).sum(), np.maximum(Ah[0] * lo, Ah[0] * hi).sum()
        return max(0.0, Lh[0] - y_hi, y_lo - Uh[0])
    n = lo.size     # LP: min s  s.t.  L̂ − s ≤ Âd ≤ Û + s, box, s ≥ 0
    A_ub = np.block([[Ah, -np.ones((m, 1))], [-Ah, -np.ones((m, 1))]])
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=A_ub, b_ub=np.r_[Uh, -Lh],
                  bounds=list(zip(lo, hi)) + [(0.0, None)], method="highs")
    return float(res.fun) if res.status == 0 else np.inf


def _scaled_viol(Ah, Lh, Uh, d):
    """Max scaled band violation of d (0 = every row inside its band)."""
    if Lh.size == 0:
        return 0.0
    y = Ah @ d
    return float(np.max(np.maximum(0.0, np.maximum(Lh - y, y - Uh))))


def _slsqp_ascent(g, Ah, Lh, Uh, D, lo, hi, tau):
    """Robust fallback for the ascent problem when the dual Newton fails on a
    REACHABLE row set: SLSQP in z = d/√D (removes the D spread). d or None."""
    sq = np.sqrt(D)
    c = sq * g
    fs = tau * float(c @ c) / 2.0 or 1.0
    cons = []
    for k in range(Lh.size):
        az = Ah[k] * sq
        cons += [{"type": "ineq", "fun": lambda z, az=az, L=Lh[k]: az @ z - L, "jac": lambda z, az=az: az},
                 {"type": "ineq", "fun": lambda z, az=az, U=Uh[k]: U - az @ z, "jac": lambda z, az=az: -az}]
    try:
        res = minimize(lambda z: (z @ z / (2 * tau) - c @ z) / fs, np.zeros_like(c),
                       jac=lambda z: (z / tau - c) / fs, method="SLSQP",
                       bounds=list(zip(lo / sq, hi / sq)), constraints=cons,
                       options={"ftol": 1e-15, "maxiter": 1000})
    except (ValueError, np.linalg.LinAlgError):
        return None
    d = np.clip(res.x * sq, lo, hi)
    ok = np.all(np.isfinite(d)) and _scaled_viol(Ah, Lh, Uh, d) <= FEAS_TOL
    return d if ok else None


def _feasible_point(g, Ah, Lh, Uh, D, lo, hi, tau):
    """Last resort on a REACHABLE row set: a feasible (not optimal) d. If d=0
    meets every band, the clipped unconstrained step scaled toward 0 until the
    rows hold (its gain stays >= 0); else an LP vertex of the feasible set."""
    if np.all(Lh <= 0.0) and np.all(Uh >= 0.0):
        d0 = np.clip(tau * D * g, lo, hi)
        s = 1.0
        for y, L, U in zip(Ah @ d0, Lh, Uh):
            if y > U:
                s = min(s, U / y)
            elif y < L:
                s = min(s, L / y)
        return s * d0
    res = linprog(np.zeros(lo.size), A_ub=np.vstack([Ah, -Ah]), b_ub=np.r_[Uh, -Lh],
                  bounds=list(zip(lo, hi)), method="highs")
    return np.clip(res.x, lo, hi) if res.status == 0 else np.zeros(lo.size)


def _mu_estimate(g, Ah, Lh, Uh, D, lo, hi, tau, d):
    """Least-squares row multipliers for a fallback d: stationarity on the
    unclipped params, rows at a band edge only (others mu = 0)."""
    m = Lh.size
    mu = np.zeros(m)
    y = Ah @ d
    act = [k for k in range(m) if min(abs(y[k] - Lh[k]), abs(y[k] - Uh[k])) <= 10 * FEAS_TOL]
    F = (d > lo) & (d < hi)
    if act and F.any():
        rhs = g[F] - d[F] / (tau * D[F])
        mu[act] = np.linalg.lstsq(Ah[act][:, F].T, rhs, rcond=None)[0]
    return mu


def qp_step(gT, rows, bands, D, lo_step, hi_step, cap_nm, restore_tol=None):
    """Bounded composite step d (nm). Maximise gT·d − (1/2τ)Σd²/D subject to
    max(lo_step,−cap) ≤ d ≤ min(hi_step,cap) and L_k ≤ a_k·d ≤ U_k.

    τ = cap/max|D·gT| makes the unconstrained d = τ·D·gT just reach the
    radius, so the step length comes from the trust radius, not from |gT|
    (gT = 0 → τ = 1, minimum-norm feasible step). If the bands cannot be met
    inside the box, the LAST row (λ, an algorithmic aid) is dropped first.
    If the first row alone is unreachable, the step restores it:
    min (viol/t)² + ε/2·Σd²/D. That prox uses τ=1, not the ascent τ, so the
    restoration does not depend on |gT|.
    restore_tol: per-row tolerances t_k in caller units (e.g. [1e-3 µm,
    1e-3 nm]) for conditioning/restoration. None → t_k = cap·‖a_k‖₁ (scale
    free, so rescaling a row together with its band changes nothing).

    status (GPT v3 review G1): "ok" = every requested row met at the optimum;
    "infeasible" = the requested rows cannot all be met inside the box (mode
    says what was done: ascent_dropped_row or restore); "solver_failed" = the
    LP says the rows ARE reachable but the dual Newton AND the SLSQP fallback
    both failed. d is then a FEASIBLE (not optimal) point, never an
    infeasible one, and the mode is NOT downgraded (no row drop, no restore).
    A restoration left above the LP-minimum violation + 1e-3 (scaled) is also
    "solver_failed". solver: dual | slsqp | feasible_point | restore.
    viol: remaining violation per row in caller units. Non-finite inputs
    raise ValueError.
    """
    gT, D = np.asarray(gT, dtype=float), np.asarray(D, dtype=float)
    lo_step, hi_step = np.asarray(lo_step, dtype=float), np.asarray(hi_step, dtype=float)
    n, m = gT.size, len(rows)
    A = np.asarray(rows, dtype=float).reshape(m, n)
    LU = np.asarray(bands, dtype=float).reshape(m, 2)
    for name, v in (("gT", gT), ("rows", A), ("D", D), ("lo_step", lo_step),
                    ("hi_step", hi_step), ("bands", LU), ("cap_nm", np.asarray(cap_nm, float))):
        if not np.all(np.isfinite(v)):
            raise ValueError(f"qp_step: non-finite {name} (NaN/inf), refusing to build a step")
    if not np.all(D > 0):
        raise ValueError("qp_step: D must be > 0 everywhere")
    assert m <= 2 and np.all(LU[:, 0] <= LU[:, 1])
    assert np.all(lo_step <= 0) and np.all(hi_step >= 0) and cap_nm > 0
    lo, hi = np.maximum(lo_step, -cap_nm), np.minimum(hi_step, cap_nm)
    Dg = np.abs(D * gT).max()
    tau = cap_nm / Dg if Dg > 0 else 1.0
    t = cap_nm * np.abs(A).sum(axis=1) if restore_tol is None else np.asarray(restore_tol, float)[:m].copy()
    t[t <= 0] = 1.0                                   # zero row: a·d ≡ 0, only its band decides
    Ah, Lh, Uh = A / t[:, None], LU[:, 0] / t, LU[:, 1] / t

    def ascent(keep):
        """(result, status, solver); result None <=> status "infeasible"."""
        a_, l_, u_ = Ah[keep], Lh[keep], Uh[keep]
        if _min_violation(a_, l_, u_, lo, hi) > FEAS_TOL:
            return None, "infeasible", None
        r = _dual_solve(gT, a_, l_, u_, D, lo, hi, tau, 0.0)
        if (r is not None and not (r[2].size and np.abs(r[2]).max() > KKT_TOL)
                and _scaled_viol(a_, l_, u_, r[1]) <= FEAS_TOL):
            return r, "ok", "dual"
        it_ = r[3] if r is not None else MAX_NEWTON
        d = _slsqp_ascent(gT, a_, l_, u_, D, lo, hi, tau)
        status, solver = "ok", "slsqp"
        if d is None:
            d = _feasible_point(gT, a_, l_, u_, D, lo, hi, tau)
            status, solver = "solver_failed", "feasible_point"
        mu_ = _mu_estimate(gT, a_, l_, u_, D, lo, hi, tau, d)
        G_ = _pgrad(mu_, a_ @ d, l_, u_, 0.0)[0]
        return (mu_, d, G_, it_), status, solver

    keep, mode = list(range(m)), "ascent"
    r, status, solver = ascent(keep)
    if r is None and m == 2:
        keep, mode = [0], "ascent_dropped_row"
        r, st2, solver = ascent(keep)
        status = "infeasible" if st2 == "ok" else st2
    if r is None:
        keep, mode, solver = [0], "restore", "restore"
        r = _dual_solve(np.zeros(n), Ah[keep], Lh[keep], Uh[keep], D, lo, hi, 1.0 / RESTORE_EPS, 1.0)
        status = "infeasible"
        if r is None or not np.all(np.isfinite(r[1])) or (
                _scaled_viol(Ah[keep], Lh[keep], Uh[keep], r[1])
                > _min_violation(Ah[keep], Lh[keep], Uh[keep], lo, hi) + 1e-3):
            status = "solver_failed"
            if r is None:
                r = (np.zeros(1), np.zeros(n), np.zeros(1), MAX_NEWTON)
    mu_s, d, G, iters = r

    mu = np.zeros(m)
    mu[keep] = mu_s / t[keep]                         # back to caller units (per row unit)
    y = A @ d
    viol = [float(max(0.0, LU[k, 0] - y[k], y[k] - LU[k, 1])) for k in range(m)]
    nf = (hi_step - lo_step) >= FROZEN_SPAN_NM
    at_cap = nf & (np.abs(d) >= cap_nm * (1 - 1e-9))
    tiny = 1e-12 * cap_nm
    at_bnd = nf & (((d <= lo_step + tiny) & (lo_step > -cap_nm)) | ((d >= hi_step - tiny) & (hi_step < cap_nm)))
    return {"d": d, "mode": mode, "mu": mu, "status": status, "solver": solver, "viol": viol,
            "pred": {"dT": float(gT @ d), "rows": [float(v) for v in y]},
            "active_cap": float(at_cap.sum() / max(nf.sum(), 1)),
            "active_bounds": int(at_bnd.sum()),
            "row_active": [bool(mu[k] != 0) for k in range(m)],
            "kkt": float(np.abs(G).max()) if G.size else 0.0,
            "tau": tau, "iters": iters}


# ---------------------------------------------------------------- F2 radius
def radius_update(cap_nm, dT_pred, dT_meas, noise, active_cap, grow=1.5, cap_max=30.0, cap_min=2.0):
    """Trust-radius rule on an ACCEPTED step. Returns (new_cap, tag).

    A prediction inside the noise cannot be used for a ratio test, so the
    cap stays and the tag is "unresolved". A poor model (ratio < 0.25) shrinks
    it. Good agreement grows it, but only when the radius actually limited
    the step (active_cap > 0). Otherwise a larger radius would not have
    changed it.
    A non-positive predicted gain never grows the radius (GPT v3 review G5:
    pred = meas = −0.01 with an active cap used to grow 10→15): the ratio
    test is meaningless for a step the model expected to LOSE, so "keep".
    Restoration steps really need a violation-reduction ratio (measured vs
    predicted width-violation decrease); the driver does not use one yet.
    """
    if abs(dT_pred) < 2.0 * noise:
        return cap_nm, "unresolved"
    if dT_pred <= 0.0:
        return float(min(max(cap_nm, cap_min), cap_max)), "keep"
    ratio = dT_meas / dT_pred
    if ratio < 0.25:
        new, tag = 0.5 * cap_nm, "shrink"
    elif 0.5 <= ratio <= 2.0 and active_cap > 0:
        new, tag = grow * cap_nm, "grow"
    else:
        new, tag = cap_nm, "keep"
    return float(min(max(new, cap_min), cap_max)), tag
