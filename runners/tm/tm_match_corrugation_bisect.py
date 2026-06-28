"""
Self-contained, unattended FLOAT-BISECTION search (one sequential Athena GPU job)
that finds the TM corrugation depth whose SPATIAL MODE WIDTH matches the fixed
TE@80/300 nm reference — i.e. equal coupling coefficient kappa, hence equal mode
confinement — keeping N=80 and the TM pitch (518.3 nm) fixed.

WHY MODE WIDTH (the physics the user asked about): in a pi-shift (quarter-wave-
shifted) Bragg grating the localized defect mode decays into the two mirrors as
|E| ~ exp(-kappa*|x|), so the ENERGY envelope ~ exp(-2*kappa*|x|) and its FWHM =
ln2/kappa. The mode width therefore depends ONLY on kappa, not on device length,
as long as the grating is long enough to contain the mode (kappa*L >> 1, true
here). So "same kappa" <=> "same mode width", and the operational target to
equalize is the spatial mode FWHM `fwhm_m` (energy envelope along x), which the
pipeline already computes every run from the always-on `field_profile` monitor.
The total length N*Lambda sets kappa*L, which controls the SPECTRAL linewidth / Q
/ peak-T — recorded here as cross-checks but NOT the matched quantity.

METHOD
  1. Run TE @ N=80, pitch 500 nm, corr 300 nm -> target spatial FWHM  w*  (meters).
  2. Evaluate TM `fwhm_m` vs corrugation depth c (pitch 518.3 nm). f(c)=fwhm(c)-w*.
       f > 0  (TM mode WIDER than TE)  -> grating too weak  -> INCREASE corrugation
       f < 0  (TM mode NARROWER)       -> we have passed the match
     Step c by STRIDE in the sign-indicated direction until f changes sign
     => a bracket [lo, hi] straddling the match; then continuous bisection.
  3. Best corrugation = the bracket endpoint whose fwhm is closest to w* (and we
     compare against the global-closest evaluated point, so a non-monotone
     response can never make us report a worse corrugation).
  4. Write a per-iteration CSV log, a summary .mat, and comparison PNGs.

PHYSICS (honest expectation): TM couples WEAKLY to the sidewall corrugation (the
field sits at the top/bottom interfaces), so at 300 nm kappa_TM < kappa_TE and the
TM mode is WIDER. The match is therefore expected at a corrugation ABOVE 300 nm.
The search confirms direction empirically and falls back to global-closest.

INDEX: n_core=1.97 / n_clad=1.444 for BOTH polarizations (user, 2026-06-28). NOTE
the 518.3 nm TM pitch was calibrated at 1.9963, so at 1.97 the resonance shifts
DOWN ~15 nm to ~1555 nm — handled by a WIDE, env-recenterable scan window.

UNATTENDED ROBUSTNESS:
  - Every FDTD evaluation is cached on disk (result_corrmatch_*_C<pm>.mat, stamped
    with the corrugation so TM evals don't overwrite each other) and in memory
    keyed by (pol, corr_pm); re-submitting the job RESUMES instead of recomputing.
  - The bracket/bisection logic is a pure function (_search_float_match),
    unit-tested locally before dispatch.
  - Each evaluation is retried once; bounds (CORR_MIN/MAX) + MAX_EVALS prevent
    runaway. A mode wider than the grating yields fwhm_m=0 -> treated as
    "increase corrugation" (degenerate guard).

DISPATCH (default partition; resume-on-requeue covers any preemption):
    bash athena/deploy_athena.sh --option2 --run=tm_match_corrugation_bisect

Outputs land in results/tm_match_corr/ on Athena and sync to
results_from_athena/tm_match_corr/ via `--results-no-fsp`:
    corr_bisect_log.csv                             every evaluation, in order
    result_tm_match_corr_summary.mat                target/best + history
    fwhm_vs_corrugation.png                         TM mode FWHM vs corr, TE line
    stopband_vs_corrugation.png                     spectral-kappa cross-check
    combined_field_envelope_TE_vs_TMmatched.png     overlaid mode envelopes
"""

import copy
import gc
import glob
import os
import re
import shutil
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
STUDY_DIR_NAME = "tm_match_corr"

# ── Search / device constants ────────────────────────────────────────────────
N            = 80          # fixed period count per side, BOTH polarizations
TE_PITCH_M   = 500e-9      # TE reference pitch
TM_PITCH_M   = 516.14e-9   # TM-calibrated pitch (n 1.97; was 518.3 @ 1.9963), kept fixed
TE_CORR_M    = 300e-9      # TE reference corrugation depth (the target geometry)
N_CORE       = 1.97        # user 2026-06-28 (both pols)
N_CLAD       = 1.444

# Scan window — WIDE + env-recenterable because n_core=1.97 shifts the resonance
# down ~15 nm from the 1.9963-calibrated ~1571 nm to ~1555 nm. Extra spectral
# points are nearly free in FDTD (DFT sampling, not extra time-stepping).
CENTER_M     = float(os.environ.get("TM_CORR_CENTER_NM", "1556")) * 1e-9
WIDTH_NM     = float(os.environ.get("TM_CORR_WIDTH_NM", "60"))
N_POINTS     = int(os.environ.get("TM_CORR_NPTS", "12001"))     # ~5 pm spacing

# Continuous corrugation search (meters).
CORR_MIN_M   = float(os.environ.get("TM_CORR_MIN_NM", "300")) * 1e-9  # = TE depth
CORR_MAX_M   = float(os.environ.get("TM_CORR_MAX_NM", "800")) * 1e-9  # narrow=400nm, fab-safe
STRIDE_M     = 100e-9     # bracket-expansion step (aligns with the seed grid)
TOL_FWHM_FRAC = 0.02      # |fwhm_TM - w*| / w* <= 2%  => match (early stop)
TOL_X_M      = 5e-9       # stop bisection when (hi-lo) <= 5 nm corrugation
MAX_EVALS    = 24         # total TM evaluations cap (backstop)
# Smart seed: we already measured TE@300=15.54 µm and TM@300=19.10 µm (matched λ,
# n 1.97). TM is wider => need MORE corrugation; FWHM=ln2/κ gives κ-up 1.23× =>
# linear estimate ~370 nm, sub-linear TM coupling pushes it to ~350-500 nm. This
# tight grid brackets that band; the adaptive bracket-search (100 nm stride from
# 300) still walks higher on its own if TM is weaker than expected.
SEED_GRID_NM = (350, 400, 450)


# ═══════════════════════════════════════════════════════════════════════════════
# Pure float bracket-then-bisection search (unit-testable, no FDTD)
# ═══════════════════════════════════════════════════════════════════════════════

def _search_float_match(eval_fn, target, c_start, stride, c_min, c_max,
                        tol_frac, tol_x, max_evals):
    """Find corrugation c (meters) whose eval_fn(c) (= spatial fwhm, meters) is
    closest to `target`. Returns (best_c, evals) where evals is {corr_pm: value}.

    DIRECTION: larger corrugation -> stronger kappa -> SMALLER fwhm. With residual
    f(c)=eval_fn(c)-target, f>0 (mode too wide) => increase c. Same sign rule as
    the integer template (increasing the control variable lowers the residual).
    Bracket by stepping until f flips sign, then bisect to tol_x on c (or until
    |f|/target<=tol_frac). Global-closest fallback so a non-monotone fwhm(c) never
    returns a worse point than one already seen.
    """
    evals = {}  # corr in pm (int) -> measured fwhm (m)

    def _key(c):
        return int(round(c * 1e12))

    def f(c):
        k = _key(c)
        if k not in evals:
            evals[k] = eval_fn(k * 1e-12)
        return evals[k] - target

    c0 = float(c_start)
    f0 = f(c0)
    if abs(f0) <= tol_frac * target:
        return c0, evals

    direction = 1 if f0 > 0 else -1   # f>0 (mode too wide) -> increase corrugation
    c_prev, f_prev = c0, f0
    bracket = None
    c = c0
    while len(evals) < max_evals:
        c_next = c + direction * stride
        c_next = max(c_min, min(c_max, c_next))
        if abs(c_next - c) < 1e-13:        # clamped at a guard bound, cannot move
            break
        f_next = f(c_next)
        if abs(f_next) <= tol_frac * target:
            return c_next, evals
        if f_prev * f_next < 0:             # straddled the target -> bracket
            bracket = tuple(sorted((c_prev, c_next)))
            break
        c_prev, f_prev = c_next, f_next
        c = c_next
        if c_next in (c_min, c_max):
            break

    if bracket is None:
        best_k = min(evals, key=lambda k: abs(evals[k] - target))
        return best_k * 1e-12, evals

    lo, hi = bracket
    f_lo, f_hi = f(lo), f(hi)
    while (hi - lo) > tol_x and len(evals) < max_evals:
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol_frac * target:
            return mid, evals
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    best_c = lo if abs(f_lo) <= abs(f_hi) else hi
    g_best_k = min(evals, key=lambda k: abs(evals[k] - target))
    if abs(evals[g_best_k] - target) < abs(evals[_key(best_c)] - target):
        best_c = g_best_k * 1e-12
    return best_c, evals


# ═══════════════════════════════════════════════════════════════════════════════
# FDTD evaluation + caching + logging (only used on the cluster)
# ═══════════════════════════════════════════════════════════════════════════════

def _q(lam_nm, fwhm_nm):
    # post_processing stores spectral_fwhm_nm signed (dw<0 on a descending lambda
    # axis); take abs() to recover the linewidth. Q = lambda / |FWHM|.
    if fwhm_nm is None or not np.isfinite(fwhm_nm) or fwhm_nm == 0:
        return float("nan")
    return float(lam_nm) / abs(float(fwhm_nm))


def _parse_pol_corr(fname):
    """(pol, corr_pm) from a stamped result filename, or (None, None)."""
    m = re.search(r"corrmatch_(te|tm)_C(\d+)", fname)
    if not m:
        return None, None
    return m.group(1).upper(), int(m.group(2))


def _load_cache(results_dir):
    """Warm-start cache {(pol, corr_pm): rec} from stamped result_*.mat already in
    the dir (resume after a requeue). Same folder + same window => consistent."""
    from runners.tm import _tm_vs_te_common as tvt
    cache = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, "result_corrmatch_*.mat"))):
        fname = os.path.basename(fp)
        pol, corr_pm = _parse_pol_corr(fname)
        if pol is None:
            continue
        try:
            d = sio.loadmat(fp)
            if "fwhm_m" not in d:
                print(f"[corr] cache skip {fname}: no fwhm_m -> will recompute")
                continue
            T = float(np.asarray(d["resonance_transmission"]).squeeze())
            lam = float(np.asarray(d["resonance_wavelength_nm"]).squeeze())
            sfwhm = float(np.asarray(d["spectral_fwhm_nm"]).squeeze())
            fwhm_m = float(np.asarray(d["fwhm_m"]).squeeze())
            res_like = {"wl_nm": d["wl_nm"], "T": d["T"],
                        "resonance_wavelength_nm": lam}
            stopband = tvt.stopband_nm(res_like)
        except Exception as e:  # noqa: BLE001 — ignore unreadable, recompute
            print(f"[corr] cache skip {fname}: {e}")
            continue
        rec = {"pol": pol, "corr_nm": corr_pm * 1e-3,
               "fwhm_m": fwhm_m, "fwhm_um": fwhm_m * 1e6,
               "spectral_fwhm_nm": sfwhm, "stopband_nm": stopband,
               "lambda_nm": lam, "T": T, "Q": _q(lam, sfwhm), "mat": fp}
        cache[(pol, corr_pm)] = rec
        print(f"[corr] cache hit  pol={pol} corr={rec['corr_nm']:.1f}nm  "
              f"FWHM={rec['fwhm_um']:.3f}um  Q={rec['Q']:.1f}")
    return cache


def _init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("order,pol,corr_nm,fwhm_um,spectral_fwhm_nm,stopband_nm,"
                     "lambda_nm,peakT,Q,target_fwhm_um,delta_fwhm_um,decision\n")


def _append_log(log_path, order, rec, target_fwhm_m):
    if target_fwhm_m is None:
        decision = "TE reference"
        tgt_um, dfw_um = None, None
    else:
        tgt_um = target_fwhm_m * 1e6
        dfw_um = rec["fwhm_um"] - tgt_um
        if rec["fwhm_m"] <= 0 or not np.isfinite(rec["fwhm_m"]):
            decision = "DEGENERATE fwhm=0 -> increase corrugation"
        elif abs(dfw_um) <= TOL_FWHM_FRAC * tgt_um:
            decision = "match (within tol)"
        elif dfw_um > 0:
            decision = "TM mode wider than TE -> increase corrugation"
        else:
            decision = "TM mode narrower than TE -> passed match"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{order},{rec['pol']},{rec['corr_nm']:.2f},{rec['fwhm_um']:.5f},"
                 f"{rec['spectral_fwhm_nm']:.6f},{rec['stopband_nm']:.4f},"
                 f"{rec['lambda_nm']:.4f},{rec['T']:.6f},{rec['Q']:.4f},"
                 f"{'' if tgt_um is None else f'{tgt_um:.5f}'},"
                 f"{'' if dfw_um is None else f'{dfw_um:+.5f}'},{decision}\n")


def _evaluate(base, pol, corr_m, pitch_m, cache, log_path, target_fwhm_m, order_box):
    """One FDTD scan for (pol, corr): build cfg, run, stamp+cache, log. Retried."""
    corr_pm = int(round(corr_m * 1e12))
    key = (pol, corr_pm)
    if key in cache:
        rec = cache[key]
        print(f"[corr] reuse pol={pol} corr={rec['corr_nm']:.1f}nm  "
              f"FWHM={rec['fwhm_um']:.3f}um  Q={rec['Q']:.1f}")
        return rec

    from runners.tm import _tm_vs_te_common as tvt
    cfg = copy.deepcopy(base)
    cfg.grating.n_periods_each_side = N
    cfg.grating.pitch_m = float(pitch_m)
    cfg.geometry.corrugation_depth_m = float(corr_m)   # widths derive automatically

    last_err = None
    res = None
    for attempt in (1, 2):
        try:
            res = tvt.run_one_scan(cfg, pol, center_m=CENTER_M,
                                   width_nm=WIDTH_NM, n_points=N_POINTS)
            break
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last_err = e
            print(f"[corr] FDTD FAILED pol={pol} corr={corr_m*1e9:.1f}nm "
                  f"attempt {attempt}: {e!r}")
            plt.close("all"); gc.collect()
    if res is None:
        raise RuntimeError(
            f"evaluation pol={pol} corr={corr_m*1e9:.1f}nm failed twice: {last_err!r}")

    # Stamp the .mat with pol + corrugation so multiple TM evals (same pitch) do
    # not overwrite each other -> enables disk resume on requeue.
    src = res["scan_mat"]
    stamped = os.path.join(os.path.dirname(src),
                           f"result_corrmatch_{pol.lower()}_C{corr_pm}.mat")
    try:
        if os.path.abspath(stamped) != os.path.abspath(src):
            shutil.copyfile(src, stamped)
    except Exception as e:  # noqa: BLE001
        print(f"[corr] stamp copy failed ({e}); using original {src}")
        stamped = src

    rec = {"pol": pol, "corr_nm": corr_m * 1e9,
           "fwhm_m": float(res["fwhm_m"]), "fwhm_um": float(res["fwhm_m"]) * 1e6,
           "spectral_fwhm_nm": float(res["spectral_fwhm_nm"]),
           "stopband_nm": float(res["stopband_nm"]),
           "lambda_nm": float(res["lambda_res_nm"]),
           "T": float(res["resonance_T"]), "mat": stamped}
    rec["Q"] = _q(rec["lambda_nm"], rec["spectral_fwhm_nm"])
    cache[key] = rec
    order_box[0] += 1
    _append_log(log_path, order_box[0], rec, target_fwhm_m)
    print(f"[corr] DONE pol={pol} corr={rec['corr_nm']:.1f}nm  "
          f"FWHM={rec['fwhm_um']:.3f}um  lam={rec['lambda_nm']:.3f}nm  "
          f"stopband={rec['stopband_nm']:.3f}nm  Q={rec['Q']:.1f}")
    plt.close("all"); gc.collect()
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting / summary
# ═══════════════════════════════════════════════════════════════════════════════

def _envelope(mat_path):
    try:
        d = sio.loadmat(mat_path)
        x = np.asarray(d["field_x"]).squeeze()
        env = np.asarray(d["field_envelope_1D"]).squeeze()
        if x.size and env.size and x.size == env.size:
            return x, env
    except Exception as e:  # noqa: BLE001
        print(f"[corr] could not read envelope {mat_path}: {e}")
    return None, None


def _interp_crossing(tm_points, target_um):
    pts = sorted(((r["corr_nm"], r["fwhm_um"]) for r in tm_points), key=lambda p: p[0])
    for (ca, wa), (cb, wb) in zip(pts, pts[1:]):
        da, db = wa - target_um, wb - target_um
        if da == 0:
            return float(ca)
        if da * db < 0:
            return ca + (da / (da - db)) * (cb - ca)
    return None


def _write_outputs(results_dir, te, tm_points, best, corr_cross):
    tm_sorted = sorted(tm_points, key=lambda r: r["corr_nm"])
    Cs = [r["corr_nm"] for r in tm_sorted]
    Ws = [r["fwhm_um"] for r in tm_sorted]
    Ss = [r["stopband_nm"] for r in tm_sorted]
    te_w_um = te["fwhm_um"]

    # spatial mode FWHM vs corrugation (the matched quantity)
    fig, ax = plt.subplots()
    ax.plot(Cs, Ws, "s-", lw=1.6, ms=7, mfc="w", label="TM")
    ax.axhline(te_w_um, color="C1", ls="--", lw=1.6,
               label=f"TE@{N}/300nm  FWHM={te_w_um:.3f} µm")
    ax.axvline(best["corr_nm"], color="0.4", ls=":", lw=1.4,
               label=f"match corr={best['corr_nm']:.1f} nm")
    ax.set_xlabel("TM corrugation depth [nm]")
    ax.set_ylabel("Spatial mode FWHM [µm]")
    ax.set_title("TM mode width vs corrugation (match TE @ same κ)")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fwhm_vs_corrugation.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # stopband (spectral-kappa proxy) vs corrugation — consistency cross-check
    fig, ax = plt.subplots()
    ax.plot(Cs, Ss, "o-", lw=1.6, ms=6, mfc="w", color="C2", label="TM stopband")
    ax.axhline(te["stopband_nm"], color="C1", ls="--", lw=1.6,
               label=f"TE stopband={te['stopband_nm']:.2f} nm")
    ax.axvline(best["corr_nm"], color="0.4", ls=":", lw=1.4)
    ax.set_xlabel("TM corrugation depth [nm]")
    ax.set_ylabel("Stopband width [nm]  (∝ κ)")
    ax.set_title("TM stopband vs corrugation (spectral κ cross-check)")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "stopband_vs_corrugation.png"), dpi=150,
                bbox_inches="tight"); plt.close(fig)

    # overlaid mode envelopes: TE vs matched-TM (visual proof of equal width)
    te_x, te_env = _envelope(te["mat"])
    tm_x, tm_env = _envelope(best["mat"])
    fig, ax = plt.subplots()
    if te_x is not None:
        te_env = te_env / np.max(te_env)
        ax.plot(te_x * 1e6, te_env, "-", lw=1.6, color="C1",
                label=f"TE@{N}/300nm  (FWHM {te_w_um:.2f} µm)")
    if tm_x is not None:
        tm_env = tm_env / np.max(tm_env)
        ax.plot(tm_x * 1e6, tm_env, "-", lw=1.6, color="C0",
                label=f"TM corr={best['corr_nm']:.0f}nm  (FWHM {best['fwhm_um']:.2f} µm)")
    ax.set_xlabel("x [µm]"); ax.set_ylabel("Normalized energy envelope")
    ax.set_title("Mode envelope: TE vs κ-matched TM")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "combined_field_envelope_TE_vs_TMmatched.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    summary = {
        "target_pol": "TE", "target_N": N, "target_pitch_nm": TE_PITCH_M * 1e9,
        "target_corr_nm": TE_CORR_M * 1e9, "target_fwhm_um": te_w_um,
        "target_spectral_fwhm_nm": te["spectral_fwhm_nm"],
        "target_stopband_nm": te["stopband_nm"],
        "target_lambda_nm": te["lambda_nm"], "target_Q": te["Q"],
        "n_core": N_CORE, "n_clad": N_CLAD,
        "tm_pitch_nm": TM_PITCH_M * 1e9,
        "best_corr_nm": best["corr_nm"], "best_fwhm_um": best["fwhm_um"],
        "delta_fwhm_um": best["fwhm_um"] - te_w_um,
        "best_spectral_fwhm_nm": best["spectral_fwhm_nm"],
        "best_stopband_nm": best["stopband_nm"],
        "best_lambda_nm": best["lambda_nm"], "best_Q": best["Q"],
        "best_T": best["T"],
        "corr_cross_interp_nm": float("nan") if corr_cross is None else float(corr_cross),
        "n_tm_evals": len(tm_points),
        "history_corr_nm": np.array(Cs), "history_fwhm_um": np.array(Ws),
        "history_stopband_nm": np.array(Ss),
    }
    sio.savemat(os.path.join(results_dir, "result_tm_match_corr_summary.mat"), summary)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg=None):
    """Unattended corrugation-match bisection on the cluster. `cfg` from the
    dispatcher is ignored; the device is rebuilt from the pinned TM/TE baseline so
    the dispatcher's global tweaks are not inherited. Index forced to 1.97/1.444."""
    import config
    from runners.tm._tm_vs_te_common import build_base_cfg
    from simulation_config import SimulationConfig

    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "corr_bisect_log.csv")
    _init_log(log_path)

    base = build_base_cfg(SimulationConfig())
    base.material.n_core_const = N_CORE     # override build_base_cfg's legacy 1.9963
    base.material.n_clad_const = N_CLAD
    base.run.cleanup_lumerical_data = True  # multi-run job: don't fill NFS with .h5

    cache = _load_cache(results_dir)
    order_box = [sum(1 for _ in open(log_path, encoding="utf-8")) - 1]

    print("=" * 70)
    print(f"[corr] results_dir = {results_dir}")
    print(f"[corr] TE ref: N={N} pitch={TE_PITCH_M*1e9:.1f}nm corr={TE_CORR_M*1e9:.0f}nm | "
          f"TM pitch={TM_PITCH_M*1e9:.1f}nm | n_core={N_CORE} n_clad={N_CLAD}")
    print(f"[corr] window {CENTER_M*1e9:.1f}±{WIDTH_NM/2:.1f}nm, {N_POINTS} pts | "
          f"corr search [{CORR_MIN_M*1e9:.0f},{CORR_MAX_M*1e9:.0f}]nm")
    print("=" * 70)

    # 1) TE reference target (spatial mode FWHM).
    te = _evaluate(base, "TE", TE_CORR_M, TE_PITCH_M, cache, log_path, None, order_box)
    w_star = te["fwhm_m"]
    if not (np.isfinite(w_star) and w_star > 0):
        raise RuntimeError(
            f"TE reference fwhm_m is degenerate ({w_star}). The mode is wider than "
            f"the grating or the resonance left the window — widen/recenter via "
            f"TM_CORR_CENTER_NM / TM_CORR_WIDTH_NM and requeue.")
    print(f"[corr] TARGET  w* = {w_star*1e6:.4f} µm  (TE Q = {te['Q']:.1f}, "
          f"lam = {te['lambda_nm']:.2f} nm)")

    # 2) optional coarse grid first: documents the fwhm(corr) curve AND brackets
    #    the root for free (the bisection then reuses these from cache).
    def eval_tm(c):
        rec = _evaluate(base, "TM", c, TM_PITCH_M, cache, log_path, w_star, order_box)
        fw = rec["fwhm_m"]
        return fw if (np.isfinite(fw) and fw > 0) else w_star * 10.0   # degenerate guard

    if os.environ.get("TM_CORR_SCAN_FIRST", "1") != "0":
        print("[corr] seeding coarse grid before bisection ...")
        for c_nm in SEED_GRID_NM:
            if CORR_MIN_M - 1e-12 <= c_nm * 1e-9 <= CORR_MAX_M + 1e-12:
                eval_tm(c_nm * 1e-9)

    # 3) float bracket-then-bisection over TM corrugation depth.
    best_c, _evals = _search_float_match(
        eval_tm, w_star, c_start=CORR_MIN_M, stride=STRIDE_M,
        c_min=CORR_MIN_M, c_max=CORR_MAX_M,
        tol_frac=TOL_FWHM_FRAC, tol_x=TOL_X_M, max_evals=MAX_EVALS)

    tm_points = [rec for (pol, _c), rec in cache.items() if pol == "TM"]
    best = cache[("TM", int(round(best_c * 1e12)))]
    corr_cross = _interp_crossing(tm_points, w_star * 1e6)

    # 4-5) outputs + verdict.
    _write_outputs(results_dir, te, tm_points, best, corr_cross)

    truncation_flag = best["fwhm_m"] > 0.5 * (2.0 * N * TM_PITCH_M)
    print("\n" + "=" * 70)
    print(f"TE@{N} (pitch {TE_PITCH_M*1e9:.1f}nm, corr {TE_CORR_M*1e9:.0f}nm): "
          f"mode FWHM = {w_star*1e6:.4f} µm  Q = {te['Q']:.1f}")
    if corr_cross is not None:
        print(f"TM mode FWHM crosses the TE value at corr ≈ {corr_cross:.1f} nm (interp).")
    else:
        print("TM mode FWHM did not bracket the TE value — reporting closest corr.")
    print(f"BEST CORRUGATION = {best['corr_nm']:.1f} nm (pitch {TM_PITCH_M*1e9:.1f}nm): "
          f"mode FWHM = {best['fwhm_um']:.4f} µm "
          f"(Δ = {(best['fwhm_um']-w_star*1e6):+.4f} µm)  Q = {best['Q']:.1f}")
    print(f"  stopband TE {te['stopband_nm']:.3f} nm vs TM {best['stopband_nm']:.3f} nm "
          f"(spectral-κ cross-check)")
    if truncation_flag:
        print("  WARNING: matched mode FWHM exceeds half the device — result may be "
              "truncation-limited, not κ-limited.")
    if abs(best["corr_nm"] - CORR_MAX_M * 1e9) < 1.0:
        print("  WARNING: match pinned at the fab-safe cap — TM κ may be too weak to "
              "reach TE at N=80 within corrugation ≤ %.0f nm." % (CORR_MAX_M * 1e9))
    print(f"TM evaluations: {len(tm_points)}   log: {log_path}")
    print("=" * 70)
    return {"target": te, "best": best, "corr_cross_nm": corr_cross}


# Auto-discovery contract: athena_run.py looks for a top-level `run` callable.


if __name__ == "__main__":
    # Local smoke test of the PURE search only (no lumapi/FDTD): a synthetic,
    # monotonically DECREASING fwhm(corr) crossing the target between grid points.
    def synthetic_fwhm(c):                       # larger corrugation -> smaller fwhm
        return 1e-6 * (1.0 + (600e-9 - c) / 100e-9)

    target = synthetic_fwhm(450e-9)              # true match at corr = 450 nm
    best, evals = _search_float_match(
        synthetic_fwhm, target, c_start=300e-9, stride=100e-9,
        c_min=300e-9, c_max=800e-9, tol_frac=0.0, tol_x=1e-9, max_evals=24)
    print(f"[selftest] evaluated corr(nm): {sorted(round(k/1000,1) for k in evals)}")
    print(f"[selftest] best corr = {best*1e9:.3f} nm (expected 450), |evals|={len(evals)}")
    assert abs(best - 450e-9) < 2e-9, f"expected 450 nm, got {best*1e9:.3f} nm"
    # Non-crossing case: constant above target -> global closest, no runaway.
    best2, evals2 = _search_float_match(
        lambda c: 5e-6, 2.5e-6, c_start=300e-9, stride=100e-9,
        c_min=300e-9, c_max=800e-9, tol_frac=0.0, tol_x=1e-9, max_evals=24)
    print(f"[selftest] non-crossing best corr = {best2*1e9:.1f} nm, "
          f"|evals|={len(evals2)} (capped at {24})")
    assert len(evals2) <= 24
    print("[selftest] PASS")
