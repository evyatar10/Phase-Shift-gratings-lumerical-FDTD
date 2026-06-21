"""
Self-contained, unattended INTEGER-BISECTION search (one sequential Athena GPU
job) that finds the TM grating period count N whose resonance PEAK TRANSMISSION
matches the fixed TE@80 reference, then reports the Q-factor comparison
(TE@80 vs the period-matched TM).

Why this design: the user wants an iterative search that, after every run, checks
whether the TM peak transmission is above or below the TE target and keeps going
accordingly — i.e. a bracket-then-bisection on integer N — with the WHOLE process
uploaded once and running unattended (no per-run babysitting).

METHOD
  1. Run TE @ N=80, pitch 500 nm  -> target peak transmission  T* (and TE Q).
  2. Evaluate TM peak T at N=80 (pitch 518.3 nm). f(N) = T_TM(N) - T*.
       f > 0  (TM transmission higher than TE) -> grating too weak, INCREASE N
       f < 0  (TM transmission lower  than TE) -> we have passed the match
     Step N by STRIDE in the sign-indicated direction until f changes sign
     => an integer bracket [lo, hi] straddling the match.
  3. Integer bisection inside [lo, hi] until hi-lo == 1.
  4. Best integer N = the endpoint whose peak T is closest to T* (and we also
     compare against the global closest evaluated point, so a non-monotonic
     response can never make us report a worse N).
  5. Write a per-iteration CSV log, a summary .mat, and comparison PNGs.

PHYSICS (honest expectation, verified by the data — not assumed): the materials
are loss-free real indices, so peak T < 1 comes from RADIATION loss. TM couples
more weakly to the corrugation than TE, so at N=80 the TM resonance is broader
and its peak T sits ABOVE the (stronger, lossier) TE peak. Increasing N raises
the TM grating strength kappa*L -> higher finesse -> more round-trip radiation
loss -> the TM peak T drops toward the TE value. The search confirms the
direction empirically and falls back to the global-closest integer if the
response is not monotonic.

UNATTENDED ROBUSTNESS (9-hour run, no operator):
  - Every FDTD evaluation is cached on disk (result_*.mat) and in memory keyed by
    (pol, N); re-submitting the job RESUMES instead of recomputing.
  - The bracket/bisection logic lives in a pure function (_search_integer_match)
    that is unit-tested locally before dispatch.
  - Each evaluation is retried once on failure; a hard bound (N_MAX) and an
    evaluation cap (MAX_EVALS) prevent any runaway.
  - Pinned to a non-preemptible public partition so the long job is not killed.

DISPATCH (ample walltime comes from athena.conf ARRAY_TIME=23:30, QOS 24h_1g):
    bash athena/deploy_athena.sh --option2 --run=tm_match_periods_bisect --gpu=a100-public

Outputs land in results/tm_match_bisect/results/ on Athena and sync to
results_from_athena/tm_match_bisect/results/ via `--results-no-fsp`:
    bisect_log.csv                              every evaluation, in order
    result_tm_match_summary.mat                 target/best/Q verdict + history
    peakT_vs_periods.png                        TM peak T vs N, TE line + match
    Q_vs_periods.png                            TM Q vs N, TE Q line
    combined_transmission_TE80_vs_TMmatched.png TE@80 and matched-N TM spectra
"""

import copy
import gc
import glob
import os
import re
import sys

# Project root importable when run directly (local unit test).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

# Auto-discovery contract for athena_run.py: top-level `run` callable, no
# IS_HELPER, dedicated results folder via STUDY_DIR_NAME.
STUDY_DIR_NAME = "tm_match_bisect"

# ── Search / device constants ────────────────────────────────────────────────
TE_N        = 80          # fixed TE reference period count (kept as-is)
TE_PITCH_M  = 500e-9      # TE resonates ~1570.7 nm at pitch 500
TM_PITCH_M  = 518.3e-9    # TM-calibrated pitch -> TM resonates ~1571 nm
CENTER_M    = 1.571e-6    # scan window center (covers both TE and TM resonances)
WIDTH_NM    = 20.0        # wide enough that the resonance never leaves the window
N_POINTS    = 4001        # ~5 pm spacing -> resolves the narrowing FWHM (hence Q)
STRIDE      = 40          # bracket-expansion step in periods
N_MIN       = 20          # lower guard on N
N_MAX       = 400         # upper guard on N (runaway backstop)
TOL_T       = 1e-4        # |T_TM - T*| treated as an exact match (early stop)
MAX_EVALS   = 30          # total TM evaluations cap (backstop)


# ═══════════════════════════════════════════════════════════════════════════════
# Pure integer bracket-then-bisection search (unit-testable, no FDTD)
# ═══════════════════════════════════════════════════════════════════════════════

def _search_integer_match(eval_fn, t_star, n_start, stride, n_min, n_max,
                          tol, max_evals):
    """Find the integer N whose eval_fn(N) is closest to t_star.

    eval_fn(N) -> measured value (here: TM resonance peak transmission). Returns
    (best_N, evals) where evals is the {N: value} dict of everything evaluated.

    Strategy: evaluate the anchor, expand by `stride` in the direction that moves
    the value toward t_star until the residual changes sign (an integer bracket),
    then integer-bisect the bracket. If no sign change is found within the guard
    band, return the globally closest evaluated N. The global-closest is also used
    as a final safety net so a non-monotonic response can never make us return a
    strictly worse integer than one we already saw.
    """
    evals = {}

    def f(n):
        n = int(n)
        if n not in evals:
            evals[n] = eval_fn(n)
        return evals[n] - t_star

    n0 = int(n_start)
    f0 = f(n0)
    if abs(f0) <= tol:
        return n0, evals

    direction = 1 if f0 > 0 else -1  # f>0 => value too high => (for T) increase N
    n_prev, f_prev = n0, f0
    bracket = None
    n = n0
    while len(evals) < max_evals:
        n_next = n + direction * stride
        n_next = max(n_min, min(n_max, n_next))
        if n_next == n:           # already at the guard bound, cannot move
            break
        f_next = f(n_next)
        if abs(f_next) <= tol:
            return n_next, evals
        if f_prev * f_next < 0:    # straddled the target -> integer bracket
            lo, hi = sorted((n_prev, n_next))
            bracket = (lo, hi)
            break
        n_prev, f_prev = n_next, f_next
        n = n_next
        if n_next in (n_min, n_max):
            break

    if bracket is None:
        best_n = min(evals, key=lambda k: abs(evals[k] - t_star))
        return best_n, evals

    lo, hi = bracket
    f_lo, f_hi = f(lo), f(hi)
    while hi - lo > 1 and len(evals) < max_evals:
        mid = (lo + hi) // 2
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            return mid, evals
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    best_n = lo if abs(f_lo) <= abs(f_hi) else hi
    g_best = min(evals, key=lambda k: abs(evals[k] - t_star))
    if abs(evals[g_best] - t_star) < abs(evals[best_n] - t_star):
        best_n = g_best
    return best_n, evals


# ═══════════════════════════════════════════════════════════════════════════════
# FDTD evaluation + caching + logging (only used on the cluster)
# ═══════════════════════════════════════════════════════════════════════════════

def _q(lam_nm, fwhm_nm):
    # post_processing stores spectral_fwhm_nm = widths*dw, and dw < 0 because the
    # wavelength axis runs descending (ascending frequency) — so the stored value
    # is the linewidth with a flipped sign. Take abs() to recover the FWHM.
    if fwhm_nm is None or not np.isfinite(fwhm_nm) or fwhm_nm == 0:
        return float("nan")
    return float(lam_nm) / abs(float(fwhm_nm))


def _parse_pol_N(fname):
    """(pol, N) from a result filename, or (None, None)."""
    m = re.search(r"_?N(\d+)", fname)
    if not m:
        return None, None
    n = int(m.group(1))
    if fname.endswith("_te.mat"):
        return "TE", n
    if fname.endswith("_tm.mat"):
        return "TM", n
    return None, None


def _load_cache(results_dir):
    """Warm-start cache {(pol,N): rec} from any result_*.mat already in the dir
    (resume after a requeue). Same folder + same window guarantee consistency."""
    cache = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, "result_*.mat"))):
        fname = os.path.basename(fp)
        if "summary" in fname:
            continue
        pol, n = _parse_pol_N(fname)
        if pol is None:
            continue
        try:
            d = sio.loadmat(fp)
            T = float(np.asarray(d["resonance_transmission"]).squeeze())
            lam = float(np.asarray(d["resonance_wavelength_nm"]).squeeze())
            fwhm = float(np.asarray(d["spectral_fwhm_nm"]).squeeze())
        except Exception as e:  # noqa: BLE001 — ignore unreadable, recompute
            print(f"[bisect] cache skip {fname}: {e}")
            continue
        cache[(pol, n)] = {"pol": pol, "N": n, "T": T, "lambda_nm": lam,
                           "fwhm_nm": fwhm, "Q": _q(lam, fwhm), "mat": fp}
        print(f"[bisect] cache hit  pol={pol} N={n}  T={T:.5f}  Q={cache[(pol,n)]['Q']:.1f}")
    return cache


def _init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("order,pol,N,peakT,lambda_nm,fwhm_nm,Q,target_T,delta_T,decision\n")


def _append_log(log_path, order, rec, target_T):
    dT = rec["T"] - target_T if target_T is not None else float("nan")
    if target_T is None:
        decision = "TE reference"
    elif abs(dT) <= TOL_T:
        decision = "match (within tol)"
    elif dT > 0:
        decision = "TM higher than TE -> increase N"
    else:
        decision = "TM lower than TE -> passed match"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{order},{rec['pol']},{rec['N']},{rec['T']:.6f},"
                 f"{rec['lambda_nm']:.4f},{rec['fwhm_nm']:.6f},{rec['Q']:.4f},"
                 f"{'' if target_T is None else f'{target_T:.6f}'},"
                 f"{'' if target_T is None else f'{dT:+.6f}'},{decision}\n")


def _evaluate(base, pol, N, pitch_m, cache, log_path, target_T, order_box):
    """One FDTD scan for (pol, N): build cfg, run, record, log. Cached + retried."""
    key = (pol, int(N))
    if key in cache:
        rec = cache[key]
        print(f"[bisect] reuse pol={pol} N={N}  T={rec['T']:.5f}  Q={rec['Q']:.1f}")
        return rec

    from runners.tm import _tm_vs_te_common as tvt
    cfg = copy.deepcopy(base)
    cfg.grating.n_periods_each_side = int(N)
    cfg.grating.pitch_m = float(pitch_m)

    last_err = None
    res = None
    for attempt in (1, 2):
        try:
            res = tvt.run_one_scan(cfg, pol, center_m=CENTER_M,
                                   width_nm=WIDTH_NM, n_points=N_POINTS)
            break
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last_err = e
            print(f"[bisect] FDTD FAILED pol={pol} N={N} attempt {attempt}: {e!r}")
            plt.close("all"); gc.collect()
    if res is None:
        raise RuntimeError(f"evaluation pol={pol} N={N} failed twice: {last_err!r}")

    rec = {"pol": pol, "N": int(N), "T": float(res["resonance_T"]),
           "lambda_nm": float(res["lambda_res_nm"]),
           "fwhm_nm": float(res["spectral_fwhm_nm"]),
           "mat": res["scan_mat"]}
    rec["Q"] = _q(rec["lambda_nm"], rec["fwhm_nm"])
    cache[key] = rec
    order_box[0] += 1
    _append_log(log_path, order_box[0], rec, target_T)
    print(f"[bisect] DONE pol={pol} N={N}  T={rec['T']:.5f}  "
          f"lam={rec['lambda_nm']:.3f}nm  FWHM={rec['fwhm_nm']:.4f}nm  Q={rec['Q']:.1f}")
    plt.close("all"); gc.collect()
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting / summary
# ═══════════════════════════════════════════════════════════════════════════════

def _spectrum(mat_path):
    try:
        d = sio.loadmat(mat_path)
        wl = np.asarray(d["wl_nm"]).squeeze()
        T = np.asarray(d["T"]).squeeze()
        if wl.size and T.size:
            return wl, T
    except Exception as e:  # noqa: BLE001
        print(f"[bisect] could not read spectrum {mat_path}: {e}")
    return None, None


def _write_outputs(results_dir, te, tm_points, best, n_cross):
    tm_sorted = sorted(tm_points, key=lambda r: r["N"])
    Ns = [r["N"] for r in tm_sorted]
    Ts = [r["T"] for r in tm_sorted]
    Qs = [r["Q"] for r in tm_sorted]

    # peak T vs N
    fig, ax = plt.subplots()
    ax.plot(Ns, Ts, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    ax.axhline(te["T"], color="C1", ls="--", lw=1.6,
               label=f"TE@{te['N']}  T={te['T']:.4f}")
    ax.axvline(best["N"], color="0.4", ls=":", lw=1.4, label=f"match N={best['N']}")
    ax.set_xlabel("TM periods per side, N"); ax.set_ylabel("Peak transmission, T")
    ax.set_title("TM peak transmission vs period count")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "peakT_vs_periods.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # Q vs N
    fig, ax = plt.subplots()
    ax.plot(Ns, Qs, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    ax.axhline(te["Q"], color="C1", ls="--", lw=1.6,
               label=f"TE@{te['N']}  Q={te['Q']:.0f}")
    ax.axvline(best["N"], color="0.4", ls=":", lw=1.4, label=f"match N={best['N']}")
    ax.set_xlabel("TM periods per side, N")
    ax.set_ylabel(r"Spectral Q = $\lambda/\Delta\lambda$")
    ax.set_title("TM Q factor vs period count")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "Q_vs_periods.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # combined transmission overlay
    te_wl, te_T = _spectrum(te["mat"])
    tm_wl, tm_T = _spectrum(best["mat"])
    fig, ax = plt.subplots()
    if te_wl is not None:
        ax.plot(te_wl, te_T, "-", lw=1.6, color="C1",
                label=f"TE  N={te['N']}  (Q≈{te['Q']:.0f})")
    if tm_wl is not None:
        ax.plot(tm_wl, tm_T, "-", lw=1.6, color="C0",
                label=f"TM  N={best['N']}  (Q≈{best['Q']:.0f})")
    ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Transmission, T")
    ax.set_title("Combined transmission: TE@80 vs period-matched TM")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "combined_transmission_TE80_vs_TMmatched.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    higher_Q = "TE" if (np.isfinite(te["Q"]) and te["Q"] > best["Q"]) else "TM"
    summary = {
        "target_pol": "TE", "target_N": te["N"], "target_pitch_nm": TE_PITCH_M * 1e9,
        "target_T": te["T"], "target_lambda_nm": te["lambda_nm"],
        "target_fwhm_nm": te["fwhm_nm"], "target_Q": te["Q"],
        "tm_pitch_nm": TM_PITCH_M * 1e9,
        "best_N": best["N"], "best_T": best["T"], "delta_T": best["T"] - te["T"],
        "best_lambda_nm": best["lambda_nm"], "best_fwhm_nm": best["fwhm_nm"],
        "best_Q": best["Q"],
        "crossing_N_interp": float("nan") if n_cross is None else float(n_cross),
        "higher_Q_polarization": higher_Q,
        "n_tm_evals": len(tm_points),
        "history_N": np.array(Ns), "history_T": np.array(Ts), "history_Q": np.array(Qs),
    }
    sio.savemat(os.path.join(results_dir, "result_tm_match_summary.mat"), summary)
    return higher_Q


def _interp_crossing(tm_points, t_star):
    pts = sorted(((r["N"], r["T"]) for r in tm_points), key=lambda p: p[0])
    for (na, ta), (nb, tb) in zip(pts, pts[1:]):
        da, db = ta - t_star, tb - t_star
        if da == 0:
            return float(na)
        if da * db < 0:
            return na + (da / (da - db)) * (nb - na)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg=None):
    """Unattended bisection on the cluster. `cfg` from the dispatcher is ignored;
    the device is rebuilt from the pinned TM/TE baseline so the dispatcher's
    global tweaks (detuning, mesh) are not inherited."""
    import config
    from runners.tm._tm_vs_te_common import build_base_cfg
    from simulation_config import SimulationConfig

    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "bisect_log.csv")
    _init_log(log_path)

    base = build_base_cfg(SimulationConfig())
    base.run.cleanup_lumerical_data = True   # multi-run job: don't fill NFS with .h5

    cache = _load_cache(results_dir)
    order_box = [sum(1 for _ in open(log_path, encoding="utf-8")) - 1]  # rows already logged

    print("=" * 64)
    print(f"[bisect] results_dir = {results_dir}")
    print(f"[bisect] TE ref: N={TE_N} pitch={TE_PITCH_M*1e9:.1f}nm | "
          f"TM pitch={TM_PITCH_M*1e9:.1f}nm | window {CENTER_M*1e9:.1f}±{WIDTH_NM/2:.1f}nm")
    print("=" * 64)

    # 1) TE reference target.
    te = _evaluate(base, "TE", TE_N, TE_PITCH_M, cache, log_path, None, order_box)
    t_star = te["T"]
    print(f"[bisect] TARGET  T* = {t_star:.6f}  (TE Q = {te['Q']:.1f})")

    # 2-3) integer bracket-then-bisection over TM period count.
    def eval_tm(n):
        rec = _evaluate(base, "TM", n, TM_PITCH_M, cache, log_path, t_star, order_box)
        return rec["T"]

    best_N, _evals = _search_integer_match(
        eval_tm, t_star, n_start=TE_N, stride=STRIDE,
        n_min=N_MIN, n_max=N_MAX, tol=TOL_T, max_evals=MAX_EVALS)

    tm_points = [rec for (pol, _n), rec in cache.items() if pol == "TM"]
    best = cache[("TM", int(best_N))]
    n_cross = _interp_crossing(tm_points, t_star)

    # 4-5) outputs + verdict.
    higher_Q = _write_outputs(results_dir, te, tm_points, best, n_cross)

    print("\n" + "=" * 64)
    print(f"TE@{te['N']} (pitch {TE_PITCH_M*1e9:.1f} nm): peak T = {te['T']:.5f}  Q = {te['Q']:.1f}")
    if n_cross is not None:
        print(f"TM peak T crosses the TE value at N ≈ {n_cross:.1f} (interpolated).")
    else:
        print("TM peak T did not bracket the TE value — reporting the closest N found.")
    print(f"BEST INTEGER N = {best['N']} (pitch {TM_PITCH_M*1e9:.1f} nm): "
          f"peak T = {best['T']:.5f}  (ΔT = {best['T']-t_star:+.5f})  Q = {best['Q']:.1f}")
    print(f"HIGHER Q: {'TE@'+str(te['N']) if higher_Q=='TE' else 'TM@'+str(best['N'])}  "
          f"(TE {te['Q']:.1f} vs TM {best['Q']:.1f})")
    print(f"TM evaluations: {len(tm_points)}   log: {log_path}")
    print("=" * 64)
    return {"target": te, "best": best, "higher_Q": higher_Q}


# Auto-discovery contract: athena_run.py looks for a top-level `run` callable.


if __name__ == "__main__":
    # Local smoke test of the PURE search only (no lumapi/FDTD): a synthetic,
    # monotonically decreasing T(N) crossing the target between two integers.
    def synthetic_T(n):
        return 1.0 - 0.0015 * n          # decreasing in N

    t_star = synthetic_T(137)            # true match at N=137
    best, evals = _search_integer_match(
        synthetic_T, t_star, n_start=80, stride=40,
        n_min=20, n_max=400, tol=1e-4, max_evals=30)
    print(f"[selftest] evaluated N: {sorted(evals)}")
    print(f"[selftest] best N = {best} (expected 137), |evals|={len(evals)}")
    assert best == 137, f"expected 137, got {best}"
    # Non-crossing case: constant above target -> global closest, no runaway.
    best2, evals2 = _search_integer_match(
        lambda n: 0.9, 0.5, n_start=80, stride=40,
        n_min=20, n_max=400, tol=1e-4, max_evals=30)
    print(f"[selftest] non-crossing best N = {best2}, |evals|={len(evals2)} (capped)")
    print("[selftest] PASS")
