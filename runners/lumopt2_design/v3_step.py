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
from scipy.optimize import linprog

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
    """
    i = int(i_pk)
    if i <= 0 or i >= T.shape[0] - 1:
        return T[i]
    Tm, T0, Tp = T[i - 1], T[i], T[i + 1]
    B = Tp - 2.0 * T0 + Tm
    if getval(B) >= 0.0:
        return T0
    D = Tp - Tm
    return T0 - D * D / (8.0 * B)


def peak3_r(T, i_pk):
    """Vertex offset r = −D/(2B) in grid steps (|r| ≤ ½ at a sampled max); 0 on fallback."""
    T = np.asarray(T, dtype=float)
    i = int(i_pk)
    if i <= 0 or i >= T.size - 1:
        return 0.0
    B = T[i + 1] - 2.0 * T[i] + T[i - 1]
    return 0.0 if B >= 0.0 else float(-(T[i + 1] - T[i - 1]) / (2.0 * B))


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
    """
    gT, D = np.asarray(gT, dtype=float), np.asarray(D, dtype=float)
    lo_step, hi_step = np.asarray(lo_step, dtype=float), np.asarray(hi_step, dtype=float)
    n, m = gT.size, len(rows)
    A = np.asarray(rows, dtype=float).reshape(m, n)
    LU = np.asarray(bands, dtype=float).reshape(m, 2)
    assert m <= 2 and np.all(D > 0) and np.all(np.isfinite(LU)) and np.all(LU[:, 0] <= LU[:, 1])
    assert np.all(lo_step <= 0) and np.all(hi_step >= 0) and cap_nm > 0
    lo, hi = np.maximum(lo_step, -cap_nm), np.minimum(hi_step, cap_nm)
    Dg = np.abs(D * gT).max()
    tau = cap_nm / Dg if Dg > 0 else 1.0
    t = cap_nm * np.abs(A).sum(axis=1) if restore_tol is None else np.asarray(restore_tol, float)[:m].copy()
    t[t <= 0] = 1.0                                   # zero row: a·d ≡ 0, only its band decides
    Ah, Lh, Uh = A / t[:, None], LU[:, 0] / t, LU[:, 1] / t

    def ascent(keep):
        if _min_violation(Ah[keep], Lh[keep], Uh[keep], lo, hi) > FEAS_TOL:
            return None
        r = _dual_solve(gT, Ah[keep], Lh[keep], Uh[keep], D, lo, hi, tau, 0.0)
        return None if r is None or (r[2].size and np.abs(r[2]).max() > KKT_TOL) else r

    keep, mode = list(range(m)), "ascent"
    r = ascent(keep)
    if r is None and m == 2:
        keep, mode = [0], "ascent_dropped_row"
        r = ascent(keep)
    if r is None:
        keep, mode = [0], "restore"
        r = _dual_solve(np.zeros(n), Ah[keep], Lh[keep], Uh[keep], D, lo, hi, 1.0 / RESTORE_EPS, 1.0)
    mu_s, d, G, iters = r

    mu = np.zeros(m)
    mu[keep] = mu_s / t[keep]                         # back to caller units (per row unit)
    nf = (hi_step - lo_step) >= FROZEN_SPAN_NM
    at_cap = nf & (np.abs(d) >= cap_nm * (1 - 1e-9))
    tiny = 1e-12 * cap_nm
    at_bnd = nf & (((d <= lo_step + tiny) & (lo_step > -cap_nm)) | ((d >= hi_step - tiny) & (hi_step < cap_nm)))
    return {"d": d, "mode": mode, "mu": mu,
            "pred": {"dT": float(gT @ d), "rows": [float(v) for v in A @ d]},
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
    """
    if abs(dT_pred) < 2.0 * noise:
        return cap_nm, "unresolved"
    ratio = dT_meas / dT_pred
    if ratio < 0.25:
        new, tag = 0.5 * cap_nm, "shrink"
    elif 0.5 <= ratio <= 2.0 and active_cap > 0:
        new, tag = grow * cap_nm, "grow"
    else:
        new, tag = cap_nm, "keep"
    return float(min(max(new, cap_min), cap_max)), tag
