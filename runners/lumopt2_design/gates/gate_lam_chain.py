"""ZERO-GPU GATE for defect #19's resonance-chain term (2026-08-25).

glam = dlam_pk/dp is what the projected optimizer was missing: gW from the
adjoint is dW/dp at FIXED lambda, but W is specced at the device's own MOVING
resonance, and W is near-linear in lam (MEASURED dW/dlam = 0.3655 um/nm).

Two estimators are checked against an analytic Lorentzian whose peak moves by a
KNOWN amount per unit parameter (and whose height ALSO drifts, so a pure
translation cannot make a wrong estimator look right):

  NAIVE   glam = -[(g_hi-g_lo)/(2h)] / [(T[hi]-2T[pk]+T[lo])/h^2]
          error factor 1/(1+x^2), x = h/g -- 49% low at x~1. This shipped
          first and the gate caught it.

  MATCHED glam = -(g_hi-g_lo) / (T'(lam_hi) - T'(lam_lo))
          For T = A(p)*S(lam-lam0(p)) with S EVEN, the amplitude part is even
          and cancels in both antisymmetric differences, so the stencil
          truncation cancels in the RATIO -- exact for any h, any symmetric
          lineshape. This is what the engine now uses.

Run: python gate_lam_chain.py
"""
import numpy as np

FWHM = 0.81          # nm, the device's measured spectral FWHM
LAM0 = 1564.61       # nm, resonance
SPAN, NPTS = 5.0, 251   # production scan window (251 pts / 5 nm = 40 per FWHM)
C_TRUE = 0.037       # nm of peak shift per unit param -- what we must recover
T_PK = 0.9012        # peak height, from job 137075_41 eval 0
DADP = 0.02          # amplitude drift per unit param (breaks pure translation)


def spectrum(wl, p):
    lam_pk = LAM0 + C_TRUE * p
    g = FWHM / 2.0
    return (T_PK + DADP * p) * g**2 / ((wl - lam_pk) ** 2 + g**2)


def _dTdp(wl, p, i, dp=1e-6):
    return (spectrum(wl, p + dp)[i] - spectrum(wl, p - dp)[i]) / (2 * dp)


def glam_naive(wl, p, k):
    T = spectrum(wl, p)
    i_pk = int(np.argmax(T))
    i_lo, i_hi = i_pk - k, i_pk + k
    h = float(wl[i_hi] - wl[i_lo]) / 2.0
    cross = (_dTdp(wl, p, i_hi) - _dTdp(wl, p, i_lo)) / (2.0 * h)
    curv = float(T[i_hi] - 2.0 * T[i_pk] + T[i_lo]) / (h * h)
    return (None if not curv < 0.0 else -cross / curv), curv


def glam_matched(wl, p, k):
    """EXACTLY the engine's recipe (incl. the λ-descending swap, 2026-08-28:
    Lumerical stores spectra freq-ascending = λ-DESCENDING; without the swap
    the dTp<0 guard refuses a genuine max — measured live on 137853_41)."""
    T = spectrum(wl, p)
    i_pk = int(np.argmax(T))
    k = max(1, min(k, i_pk - 1, len(wl) - 2 - i_pk))
    i_lo, i_hi = i_pk - k, i_pk + k
    if wl[i_hi] < wl[i_lo]:
        i_lo, i_hi = i_hi, i_lo
    tp_lo = float(T[i_lo + 1] - T[i_lo - 1]) / float(wl[i_lo + 1] - wl[i_lo - 1])
    tp_hi = float(T[i_hi + 1] - T[i_hi - 1]) / float(wl[i_hi + 1] - wl[i_hi - 1])
    dTp = tp_hi - tp_lo
    if not dTp < 0.0:
        return None, dTp
    return -(_dTdp(wl, p, i_hi) - _dTdp(wl, p, i_lo)) / dTp, dTp


wl = np.linspace(LAM0 - SPAN / 2, LAM0 + SPAN / 2, NPTS)
dl = wl[1] - wl[0]
K_ENGINE = max(1, int(round(0.5 * FWHM / dl)))
print(f"grid {NPTS} pts / {SPAN} nm = {dl*1000:.1f} pm/pt, "
      f"{FWHM/dl:.0f} pts per FWHM; engine k = {K_ENGINE}")
print(f"true dlam_pk/dp = {C_TRUE:+.6f} nm/param\n")
ok = True

print("  stencil sweep -- NAIVE vs MATCHED (rel. err of glam):")
print("    k    x=h/g    naive      matched")
for k in (1, 2, 4, 8, 16, 20, 40):
    if k > (NPTS - 2) // 2:
        continue
    gn, _ = glam_naive(wl, 0.0, k)
    gm, _ = glam_matched(wl, 0.0, k)
    en = abs(gn - C_TRUE) / C_TRUE if gn else float("nan")
    em = abs(gm - C_TRUE) / C_TRUE if gm else float("nan")
    print("   %2d   %6.3f   %6.2f%%    %7.4f%%"
          % (k, k * dl / (FWHM / 2), 100 * en, 100 * em))
    # the whole point: MATCHED must stay accurate where NAIVE collapses
    ok &= em < 0.01

print(f"\n  {'OK  ' if ok else 'FAIL'} matched estimator < 1% for every k")

# Fable's closed form for the naive error: exactly 1/(1+x^2)
gn20, _ = glam_naive(wl, 0.0, 20)
x = 20 * dl / (FWHM / 2)
pred = C_TRUE / (1.0 + x * x)
agree = abs(gn20 - pred) / pred < 2e-3
print(f"  {'OK  ' if agree else 'FAIL'} naive error matches closed form "
      f"C/(1+x^2): got {gn20:+.6f}, predicted {pred:+.6f}")
ok &= agree

# operating-point robustness of the estimator the engine actually uses
print()
for p in (0.0, 0.5, -0.5, 2.0):
    g, dTp = glam_matched(wl, p, K_ENGINE)
    err = abs(g - C_TRUE) / C_TRUE
    flag = "OK  " if err < 0.01 else "FAIL"
    ok &= err < 0.01
    print(f"  {flag} p={p:+.2f}  glam {g:+.6f}  err {err*100:6.3f}%  "
          f"(dTp={dTp:+.4g})")

# sign: a peak that RED-shifts under +p must give glam > 0, so that
# gW += 0.3655*glam PENALISES red drift (measured: red drift widens the mode)
g_sign, _ = glam_matched(wl, 0.0, K_ENGINE)
print(f"\n  {'OK  ' if g_sign > 0 else 'FAIL'} sign: red-shifting peak -> "
      f"glam {g_sign:+.6f} (must be > 0)")
ok &= g_sign > 0

# guard: a stencil straddling a MINIMUM must be rejected, not sign-flipped
T_inv = -spectrum(wl, 0.0)
i_c = int(np.argmax(T_inv))
tp_lo = float(T_inv[i_c - K_ENGINE + 1] - T_inv[i_c - K_ENGINE - 1])
tp_hi = float(T_inv[i_c + K_ENGINE + 1] - T_inv[i_c + K_ENGINE - 1])
print(f"  {'OK  ' if (tp_hi-tp_lo) < 0 else 'FAIL'} guard: dTp<0 at a maximum")
ok &= (tp_hi - tp_lo) < 0
print(f"  {'OK  ' if -(tp_hi-tp_lo) >= 0 else 'FAIL'} guard: dTp>=0 at a "
      f"minimum -> engine prints a loud skip")

# delta-leak (Fable): the argmax INDEX sits up to dl/2 off the true peak, which
# leaks the amplitude gradient at O(delta). h-independent; removed only by
# fitting lam0. Reported, not failed -- it is ~0.6% here.
worst = 0.0
for shift in np.linspace(-0.5, 0.5, 11):
    wl_s = wl + shift * dl          # slide the grid under the true peak
    g, _ = glam_matched(wl_s, 0.0, K_ENGINE)
    worst = max(worst, abs(g - C_TRUE) / C_TRUE)
print(f"\n  delta-leak (grid offset vs true peak): worst {worst*100:.2f}% "
      f"-- expected ~{100*DADP*(dl/2)/T_PK/C_TRUE:.2f}%, accepted")

# ★λ-DESCENDING order (the live 137853_41 it-0 skip): the engine recipe must
# give the SAME gLam on a reversed grid, and the OLD (unswapped) recipe must
# refuse it — proof this check has teeth.
g_asc, dTp_asc = glam_matched(wl, 0.0, K_ENGINE)
g_desc, dTp_desc = glam_matched(wl[::-1].copy(), 0.0, K_ENGINE)
inv_ok = g_desc is not None and abs(g_desc - g_asc) / abs(g_asc) < 1e-9
print(f"  {'OK  ' if inv_ok else 'FAIL'} λ-descending grid: gLam invariant "
      f"({g_asc:+.6f} vs {g_desc if g_desc is None else g_desc:+.6f}), "
      f"dTp {dTp_asc:+.3f}/{dTp_desc:+.3f}")
ok &= inv_ok


def _glam_unswapped(wl, p, k):
    T = spectrum(wl, p)
    i_pk = int(np.argmax(T))
    k = max(1, min(k, i_pk - 1, len(wl) - 2 - i_pk))
    i_lo, i_hi = i_pk - k, i_pk + k
    tp_lo = float(T[i_lo + 1] - T[i_lo - 1]) / float(wl[i_lo + 1] - wl[i_lo - 1])
    tp_hi = float(T[i_hi + 1] - T[i_hi - 1]) / float(wl[i_hi + 1] - wl[i_hi - 1])
    dTp = tp_hi - tp_lo
    return (None, dTp) if not dTp < 0.0 else (0.0, dTp)


g_old, dTp_old = _glam_unswapped(wl[::-1].copy(), 0.0, K_ENGINE)
teeth = g_old is None and dTp_old > 0
print(f"  {'OK  ' if teeth else 'FAIL'} old unswapped recipe still refuses the "
      f"descending grid (dTp {dTp_old:+.3f} > 0) — teeth confirmed")
ok &= teeth

print("\n" + ("ALL PASS" if ok else "*** GATE FAILED ***"))
raise SystemExit(0 if ok else 1)
