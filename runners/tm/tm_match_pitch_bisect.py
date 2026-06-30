"""
Unattended SECANT search (one sequential Athena GPU job) for the grating PITCH
that puts the TM pi-shift defect resonance at a TARGET wavelength (default 1590
nm). Runs at a SHORT device (N=80 periods/side) so it is cheap — we only need the
resonance WAVELENGTH (peak location), which is robust to device length and to the
transverse box size, so the default 1.8λ box is fine here (peak T may read >1 on
this near-cutoff guide; harmless, we read only the peak location).

WHY A PITCH SEARCH, AND WHY IT IS "SMART"
  The Bragg / cavity condition is  λ_res = 2·n_eff(λ)·Λ , so λ_res is NEARLY LINEAR
  in the pitch Λ (only slightly sub-linear through the weak waveguide dispersion of
  n_eff). Rooting g(Λ)=λ_res(Λ)-λ_target by regula-falsi (secant) therefore lands
  on the target in ~1-2 evaluations past two seeds — no random/grid search. The two
  seeds are placed around a physics first-guess Λ0 = λ_target/(2·n_eff_guess), with
  n_eff_guess taken from the known calibration point (pitch 531.5 nm → 1549.33 nm on
  the 200 nm-height device ⇒ n_eff ≈ 1.4575).

WHY THE CORRUGATION IS FIXED DURING THE PITCH SEARCH
  λ_res depends on the AVERAGE index, which depends on the corrugation depth, so the
  pitch-search device must carry the SAME corrugation the final device will have. We
  fix it at 227.7 nm (the depth that gave an 80 µm spatial mode FWHM at 1550 nm on
  this same 200 nm guide); the 40 nm wavelength retarget barely moves the 80 µm match
  (κ is ~wavelength-flat), so this is consistent. The downstream wide-mode corrugation
  secant (tm_wide_mode_corr.py) then re-trims the depth to hit 80 µm exactly at the
  new pitch, starting from ≈227.7 nm.

INDEX / GEOMETRY: n_core=1.97, n_clad=1.444 (user, 2026-06-28). Height from
TM_HEIGHT_NM (default 350; pass 200 for this study).

UNATTENDED ROBUSTNESS (same pattern as the sibling searches):
  - Every FDTD eval is cached on disk (result_pitchmatch_tm_P<pm>.mat, stamped with
    the pitch) and in memory keyed by pitch_pm → re-submitting RESUMES.
  - Each eval retried once; PITCH_MIN/MAX + MAX_EVALS bound the search; an off-window
    resonance is surfaced as a warning (CLAUDE.md sanity policy).
  - The pure secant (_secant_pitch) is unit-tested locally before dispatch.

DISPATCH (default partition; resume-on-requeue covers preemption):
    TM_HEIGHT_NM=200 bash athena/deploy_athena.sh --option2 --run=tm_match_pitch_bisect

Env overrides (all have working in-file defaults; only TM_HEIGHT_NM is forwarded by
deploy, so the rest must keep usable defaults):
    TM_PITCH_TARGET_NM   target resonance wavelength in nm   (default 1590)
    TM_PITCH_N           periods per side                    (default 80)
    TM_PITCH_CORR_NM     fixed corrugation depth in nm       (default 227.7)
    TM_PITCH_NEFF_GUESS  n_eff for the Λ0 first-guess        (default 1.4575)
    TM_PITCH_SEEDS_NM    comma seeds, e.g. "542.5,548.5"     (default Λ0±3)
    TM_PITCH_MIN_NM / TM_PITCH_MAX_NM  pitch bounds          (default Λ0±15)
    TM_PITCH_TOL_NM      |λ_res-target| early-stop (nm)      (default 0.5)
    TM_PITCH_MAX_EVALS   total evaluations cap               (default 8)
    TM_PITCH_CENTER_NM / TM_PITCH_WIDTH_NM / TM_PITCH_NPTS    scan window
                                                  (default target / 90 / 9001)

Outputs land in results/tm_pitch_match[_H<h>]/ on Athena and sync via
`--results-no-fsp`:
    pitch_secant_log.csv               every evaluation, in order
    result_tm_pitch_match_summary.mat  target/found pitch + history + linear fit
    lambda_vs_pitch.png                λ_res(pitch) data + the linear fit used
"""

import copy
import gc
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the proven Q helper + material constants from the corrugation driver.
from runners.tm import tm_match_corrugation_bisect as tmc

# Auto-discovery contract: top-level `run` + dedicated (height-aware) study folder.
_HEIGHT_NM = float(os.environ.get("TM_HEIGHT_NM", "350"))
STUDY_DIR_NAME = ("tm_pitch_match" if abs(_HEIGHT_NM - 350.0) < 0.5
                  else f"tm_pitch_match_H{_HEIGHT_NM:.0f}")

# ── Search / device constants (env-overridable; deploy forwards only TM_HEIGHT_NM,
#    so every default below must be usable as-is) ───────────────────────────────
TARGET_LAM_NM = float(os.environ.get("TM_PITCH_TARGET_NM") or "1590")
N_SIDE        = int(os.environ.get("TM_PITCH_N") or "80")
CORR_M        = float(os.environ.get("TM_PITCH_CORR_NM") or "227.7") * 1e-9
NEFF_GUESS    = float(os.environ.get("TM_PITCH_NEFF_GUESS") or "1.4575")

# Physics first-guess pitch and the seeds/bounds placed around it.
_P0_NM        = TARGET_LAM_NM / (2.0 * NEFF_GUESS)             # λ = 2·n_eff·Λ
SEEDS_NM      = tuple(float(s) for s in
                      (os.environ.get("TM_PITCH_SEEDS_NM")
                       or f"{_P0_NM - 3.0:.3f},{_P0_NM + 3.0:.3f}").split(","))
PITCH_MIN_NM  = float(os.environ.get("TM_PITCH_MIN_NM") or f"{_P0_NM - 15.0:.3f}")
PITCH_MAX_NM  = float(os.environ.get("TM_PITCH_MAX_NM") or f"{_P0_NM + 15.0:.3f}")
TOL_NM        = float(os.environ.get("TM_PITCH_TOL_NM") or "0.5")
MAX_EVALS     = int(os.environ.get("TM_PITCH_MAX_EVALS") or "8")

# Scan window: centered on the target, wide enough that a first-guess miss of a few
# nm still lands inside (±45 nm). Spectral points are nearly free in FDTD.
CENTER_M = float(os.environ.get("TM_PITCH_CENTER_NM") or f"{TARGET_LAM_NM}") * 1e-9
WIDTH_NM = float(os.environ.get("TM_PITCH_WIDTH_NM") or "90")
N_POINTS = int(os.environ.get("TM_PITCH_NPTS") or "9001")


# ═══════════════════════════════════════════════════════════════════════════════
# Pure secant root-find on the NEAR-LINEAR coordinate λ_res vs pitch
# (unit-testable; no FDTD).
# ═══════════════════════════════════════════════════════════════════════════════

def _secant_pitch(eval_lam, target_nm, seeds_nm, p_min_nm, p_max_nm,
                  tol_nm, max_evals):
    """Find pitch p (nm) with eval_lam(p)=resonance wavelength (nm) closest to
    target_nm. Roots g(p)=λ(p)-target in (pitch, λ) space, which is ~linear because
    λ=2·n_eff·Λ. Returns (best_p_nm, evals{p_nm: lam_nm}).

    DIRECTION falls out of the secant automatically: larger pitch -> longer λ.
    Global-closest fallback if linearity is poor.
    """
    evals = {}

    def F(p_nm):
        p_nm = round(min(p_max_nm, max(p_min_nm, float(p_nm))), 3)
        if p_nm not in evals:
            evals[p_nm] = eval_lam(p_nm)
        return p_nm, evals[p_nm]

    pts = []  # (p_nm, lam_nm)
    for s in seeds_nm:
        p, lam = F(s)
        if abs(lam - target_nm) <= tol_nm:
            return p, evals
        pts.append((p, lam))

    while len(evals) < max_evals:
        (p1, l1), (p2, l2) = pts[-2], pts[-1]
        if l2 == l1:                       # flat -> cannot step; bail to fallback
            break
        p_next = p2 + (target_nm - l2) * (p2 - p1) / (l2 - l1)
        before = len(evals)
        p, lam = F(p_next)
        if abs(lam - target_nm) <= tol_nm:
            return p, evals
        if len(evals) == before:           # clamped onto an existing point -> stall
            break
        pts.append((p, lam))

    best = min(evals, key=lambda p: abs(evals[p] - target_nm))
    return best, evals


# ═══════════════════════════════════════════════════════════════════════════════
# FDTD evaluation + caching + logging (only used on the cluster)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_pitch(fname):
    """pitch_pm from a stamped result filename, or None."""
    m = re.search(r"pitchmatch_tm_P(\d+)", fname)
    return int(m.group(1)) if m else None


def _load_cache(results_dir):
    """Warm-start cache {pitch_pm: rec} from stamped result_*.mat already present
    (resume after a requeue). Same folder + same window => consistent."""
    cache = {}
    for fp in sorted(glob.glob(os.path.join(results_dir, "result_pitchmatch_tm_P*.mat"))):
        fname = os.path.basename(fp)
        pitch_pm = _parse_pitch(fname)
        if pitch_pm is None:
            continue
        try:
            d = sio.loadmat(fp)
            lam = float(np.asarray(d["resonance_wavelength_nm"]).squeeze())
            T = float(np.asarray(d["resonance_transmission"]).squeeze())
            sfwhm = float(np.asarray(d["spectral_fwhm_nm"]).squeeze())
        except Exception as e:  # noqa: BLE001 — ignore unreadable, recompute
            print(f"[pitch] cache skip {fname}: {e}")
            continue
        rec = {"pitch_nm": pitch_pm * 1e-3, "lambda_nm": lam, "T": T,
               "spectral_fwhm_nm": sfwhm, "Q": tmc._q(lam, sfwhm), "mat": fp}
        cache[pitch_pm] = rec
        print(f"[pitch] cache hit  pitch={rec['pitch_nm']:.3f}nm  "
              f"λ={lam:.3f}nm  T={T:.4f}  Q={rec['Q']:.1f}")
    return cache


def _init_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("order,pitch_nm,lambda_nm,delta_nm,peakT,spectral_fwhm_nm,"
                     "Q,target_nm,decision\n")


def _append_log(log_path, order, rec, target_nm):
    dlam = rec["lambda_nm"] - target_nm
    if abs(dlam) <= TOL_NM:
        decision = "match (within tol)"
    elif dlam < 0:
        decision = "resonance below target -> increase pitch"
    else:
        decision = "resonance above target -> decrease pitch"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{order},{rec['pitch_nm']:.3f},{rec['lambda_nm']:.4f},"
                 f"{dlam:+.4f},{rec['T']:.6f},{rec['spectral_fwhm_nm']:.6f},"
                 f"{rec['Q']:.4f},{target_nm:.3f},{decision}\n")


def _evaluate(base, pitch_m, cache, log_path, order_box):
    """One FDTD scan at a pitch (N, corr, height fixed): build cfg, run, stamp+cache,
    log. Retried once."""
    pitch_pm = int(round(pitch_m * 1e12))
    if pitch_pm in cache:
        rec = cache[pitch_pm]
        print(f"[pitch] reuse pitch={rec['pitch_nm']:.3f}nm  λ={rec['lambda_nm']:.3f}nm")
        return rec

    from runners.tm import _tm_vs_te_common as tvt
    cfg = copy.deepcopy(base)
    cfg.grating.n_periods_each_side = N_SIDE
    cfg.grating.pitch_m = float(pitch_m)
    cfg.geometry.corrugation_depth_m = float(CORR_M)   # widths derive automatically

    last_err = None
    res = None
    for attempt in (1, 2):
        try:
            res = tvt.run_one_scan(cfg, "TM", center_m=CENTER_M,
                                   width_nm=WIDTH_NM, n_points=N_POINTS)
            break
        except Exception as e:  # noqa: BLE001 — retry once, then surface
            last_err = e
            print(f"[pitch] FDTD FAILED pitch={pitch_m*1e9:.3f}nm attempt {attempt}: {e!r}")
            plt.close("all"); gc.collect()
    if res is None:
        raise RuntimeError(
            f"evaluation pitch={pitch_m*1e9:.3f}nm failed twice: {last_err!r}")

    src = res["scan_mat"]
    stamped = os.path.join(os.path.dirname(src),
                           f"result_pitchmatch_tm_P{pitch_pm}.mat")
    try:
        if os.path.abspath(stamped) != os.path.abspath(src):
            shutil.copyfile(src, stamped)
    except Exception as e:  # noqa: BLE001
        print(f"[pitch] stamp copy failed ({e}); using original {src}")
        stamped = src

    lam = float(res["lambda_res_nm"])
    rec = {"pitch_nm": pitch_m * 1e9, "lambda_nm": lam,
           "T": float(res["resonance_T"]),
           "spectral_fwhm_nm": float(res["spectral_fwhm_nm"]), "mat": stamped}
    rec["Q"] = tmc._q(lam, rec["spectral_fwhm_nm"])
    cache[pitch_pm] = rec
    order_box[0] += 1
    _append_log(log_path, order_box[0], rec, TARGET_LAM_NM)

    lo, hi = (CENTER_M * 1e9 - WIDTH_NM / 2.0), (CENTER_M * 1e9 + WIDTH_NM / 2.0)
    flag = "" if (lo < lam < hi) else "  *** OFF-WINDOW — widen TM_PITCH_WIDTH_NM ***"
    print(f"[pitch] DONE pitch={rec['pitch_nm']:.3f}nm  λ={lam:.3f}nm  "
          f"(Δ={lam-TARGET_LAM_NM:+.3f}nm)  T={rec['T']:.4f}  Q={rec['Q']:.1f}{flag}")
    plt.close("all"); gc.collect()
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting / summary
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(results_dir, points, best, pitch_found_nm):
    pts = sorted(points, key=lambda r: r["pitch_nm"])
    Ps = np.array([r["pitch_nm"] for r in pts])
    Ls = np.array([r["lambda_nm"] for r in pts])

    fig, ax = plt.subplots()
    ax.plot(Ps, Ls, "s", ms=8, mfc="w", color="C0", label="FDTD evaluations")
    if Ps.size >= 2:
        a, b = np.polyfit(Ps, Ls, 1)              # λ ≈ a·pitch + b (near-linear)
        pp = np.linspace(Ps.min(), Ps.max(), 200)
        ax.plot(pp, a * pp + b, "-", lw=1.3, color="C0", alpha=0.6,
                label="linear fit")
    ax.axhline(TARGET_LAM_NM, color="C3", ls="--", lw=1.6,
               label=f"target {TARGET_LAM_NM:.1f} nm")
    ax.axvline(pitch_found_nm, color="0.4", ls=":", lw=1.4,
               label=f"pitch = {pitch_found_nm:.3f} nm")
    ax.set_xlabel("Grating pitch [nm]")
    ax.set_ylabel("Resonance wavelength [nm]")
    ax.set_title(f"TM pitch search (N={N_SIDE}/side, corr {CORR_M*1e9:.1f} nm)")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "lambda_vs_pitch.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    slope = float(np.polyfit(Ps, Ls, 1)[0]) if Ps.size >= 2 else float("nan")
    summary = {
        "target_lambda_nm": TARGET_LAM_NM, "n_periods_each_side": N_SIDE,
        "corr_nm": CORR_M * 1e9, "n_core": tmc.N_CORE, "n_clad": tmc.N_CLAD,
        "height_nm": _HEIGHT_NM, "neff_guess": NEFF_GUESS,
        "p0_guess_nm": _P0_NM,
        "best_pitch_nm": pitch_found_nm,
        "best_lambda_nm": best["lambda_nm"],
        "delta_lambda_nm": best["lambda_nm"] - TARGET_LAM_NM,
        "best_T": best["T"], "best_Q": best["Q"],
        "best_spectral_fwhm_nm": best["spectral_fwhm_nm"],
        "dlambda_dpitch": slope,           # nm resonance per nm pitch
        "n_evals": len(points),
        "history_pitch_nm": Ps, "history_lambda_nm": Ls,
    }
    sio.savemat(os.path.join(results_dir, "result_tm_pitch_match_summary.mat"), summary)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg=None):
    """Unattended secant pitch search on the cluster. `cfg` from the dispatcher is
    ignored; the device is rebuilt from the pinned TM baseline (index forced to
    1.97/1.444) so dispatcher tweaks aren't inherited."""
    import config
    from runners.tm._tm_vs_te_common import build_base_cfg
    from simulation_config import SimulationConfig

    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "pitch_secant_log.csv")
    _init_log(log_path)

    base = build_base_cfg(SimulationConfig())
    base.material.n_core_const = tmc.N_CORE      # override build_base_cfg legacy 1.9963
    base.material.n_clad_const = tmc.N_CLAD
    base.run.cleanup_lumerical_data = True       # multi-run job: don't fill NFS with .h5

    cache = _load_cache(results_dir)
    order_box = [sum(1 for _ in open(log_path, encoding="utf-8")) - 1]

    print("=" * 70)
    print(f"[pitch] results_dir = {results_dir}")
    print(f"[pitch] target λ = {TARGET_LAM_NM:.1f} nm | N={N_SIDE}/side | "
          f"corr={CORR_M*1e9:.1f}nm | height={_HEIGHT_NM:.0f}nm | "
          f"n_core={tmc.N_CORE} n_clad={tmc.N_CLAD}")
    print(f"[pitch] first-guess Λ0={_P0_NM:.3f}nm (n_eff_guess={NEFF_GUESS}) | "
          f"seeds {SEEDS_NM} | bounds [{PITCH_MIN_NM:.2f},{PITCH_MAX_NM:.2f}]nm | "
          f"tol {TOL_NM}nm")
    print(f"[pitch] window {CENTER_M*1e9:.0f}±{WIDTH_NM/2:.0f}nm, {N_POINTS} pts")
    print("=" * 70)

    def eval_lam(p_nm):
        rec = _evaluate(base, p_nm * 1e-9, cache, log_path, order_box)
        return rec["lambda_nm"]

    best_p_nm, _evals = _secant_pitch(
        eval_lam, TARGET_LAM_NM, SEEDS_NM, PITCH_MIN_NM, PITCH_MAX_NM,
        TOL_NM, MAX_EVALS)

    points = list(cache.values())
    best = cache[int(round(best_p_nm * 1e-9 * 1e12))]

    _write_outputs(results_dir, points, best, best_p_nm)

    print("\n" + "=" * 70)
    print(f"TARGET resonance = {TARGET_LAM_NM:.2f} nm  (N={N_SIDE}/side, "
          f"corr {CORR_M*1e9:.1f} nm, height {_HEIGHT_NM:.0f} nm)")
    print(f"BEST PITCH = {best['pitch_nm']:.3f} nm: λ_res = {best['lambda_nm']:.3f} nm "
          f"(Δ = {best['lambda_nm']-TARGET_LAM_NM:+.3f} nm)")
    print(f"  peak T = {best['T']:.4f} | Q = {best['Q']:.1f} | "
          f"spectral FWHM = {abs(best['spectral_fwhm_nm']):.4f} nm")
    if abs(best["lambda_nm"] - TARGET_LAM_NM) > TOL_NM:
        print(f"  NOTE: not within {TOL_NM} nm — raise TM_PITCH_MAX_EVALS or check "
              f"linearity (see lambda_vs_pitch.png).")
    print(f"  -> feed this pitch into the wide-mode corrugation search:")
    print(f"     TM_HEIGHT_NM={_HEIGHT_NM:.0f} TM_WIDE_PITCH_NM={best['pitch_nm']:.3f} "
          f"SPAN_MULT=4 SBATCH_MEM=256G bash athena/deploy_athena.sh "
          f"--option2 --run=tm_wide_mode_corr")
    print(f"Evaluations: {len(points)}   log: {log_path}")
    print("=" * 70)
    return {"target_lambda_nm": TARGET_LAM_NM, "best": best,
            "best_pitch_nm": best["pitch_nm"]}


if __name__ == "__main__":
    # Local smoke test of the PURE secant only (no lumapi/FDTD): a synthetic
    # NEAR-LINEAR λ(pitch) = 2·n_eff·Λ with a weak dispersion term, crossing the
    # target between the two seeds. The secant must converge in ~1 step past them.
    def synth(p_nm):
        # n_eff decreasing slightly with λ (normal dispersion): tiny curvature.
        neff = 1.4575 - 1.0e-4 * (p_nm - 545.0)
        return 2.0 * neff * p_nm

    target = 1590.0
    best, evals = _secant_pitch(synth, target, (542.0, 548.0),
                                530.0, 560.0, 0.01, 8)
    print(f"[selftest] evals(nm)={sorted(round(p, 3) for p in evals)}")
    print(f"[selftest] best pitch={best:.3f} nm -> lam={synth(best):.4f} nm "
          f"(target {target})")
    assert abs(synth(best) - target) <= 0.01, synth(best)
    assert len(evals) <= 5, f"too many evals: {len(evals)}"
    # Degenerate flat response -> global-closest, no runaway.
    best2, evals2 = _secant_pitch(lambda p: 1600.0, 1590.0, (542.0, 548.0),
                                  530.0, 560.0, 0.01, 8)
    assert len(evals2) <= 8
    print(f"[selftest] flat-response best={best2:.1f} nm, |evals|={len(evals2)}")
    print("[selftest] PASS")
