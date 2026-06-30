"""
Unattended SECANT search (one sequential Athena GPU job) for the TM corrugation
depth that yields a target SPATIAL MODE FWHM of 80 µm — a deliberately WIDE,
weakly-coupled pi-shift cavity. Runs at N=300 periods/side so the long mode is
not truncated by the device ends.

WHY A WIDE MODE NEEDS BOTH A WEAK CORRUGATION *AND* A LONG DEVICE
  The localized defect mode of a pi-shift grating decays as |E| ~ exp(-kappa|x|),
  so the energy envelope ~ exp(-2 kappa |x|) and its FWHM = ln2/kappa. The width
  is set ONLY by kappa, and kappa ~ (proportional to) the sidewall corrugation
  depth for a shallow grating. An 80 µm FWHM therefore needs kappa = ln2/80µm ≈
  87 cm^-1 — ~5x weaker than the TE@300nm reference (~446 cm^-1), i.e. corrugation
  near ~70 nm. BUT an 80 µm FWHM is already a ±40 µm object before its exponential
  tails: to contain ~93% of the energy the device must be ~3.9 FWHM long, hence
  N=300/side (310 µm total). At N=80 the mode would be truncation-limited, not
  kappa-limited. (See the sibling tm_match_corrugation_bisect.py for the same
  physics applied to matching the TE width.)

SMART METHOD — secant on the LINEAR coordinate
  Because FWHM = ln2/kappa and kappa ∝ corrugation, the quantity 1/FWHM is ~LINEAR
  in corrugation. We therefore root-find g(c) = 1/fwhm(c) - 1/target by regula-
  falsi in (corrugation, 1/fwhm) space: two physics-seeded points (60, 100 nm)
  bracket the ~70 nm estimate, and each secant step lands on the target to within
  TOL (2%) in ~1-2 further evaluations. Far fewer FDTD runs than bisecting fwhm(c)
  directly (which is convex, ~1/c). Global-closest fallback guards non-linearity.

INDEX / GEOMETRY (user, anchored 2026-06-28): n_core=1.97, n_clad=1.444, TM pitch
516.83 nm (co-resonant + width-matched anchor). Override via env if needed.

UNATTENDED ROBUSTNESS (inherited from the bisection driver's helpers):
  - Every FDTD eval is cached on disk (result_corrmatch_tm_C<pm>.mat) and in memory
    keyed by corrugation -> re-submitting RESUMES instead of recomputing.
  - Each eval retried once; CORR_MIN/MAX + MAX_EVALS bound the search; a mode wider
    than the grating (fwhm<=0 / truncated) is treated as "too shallow -> increase".

DISPATCH (default partition; resume-on-requeue covers preemption):
    bash athena/deploy_athena.sh --option2 --run=tm_wide_mode_corr

Env overrides:
    TM_WIDE_TARGET_UM   target spatial FWHM in µm        (default 80)
    TM_WIDE_N           periods per side                 (default 300)
    TM_WIDE_PITCH_NM    pitch in nm                      (default 516.83)
    TM_WIDE_SEEDS_NM    comma seeds, e.g. "60,100"       (default 60,100)
    TM_WIDE_CORR_MIN_NM / TM_WIDE_CORR_MAX_NM            (default 40 / 200)
    TM_WIDE_CENTER_NM / TM_WIDE_WIDTH_NM / TM_WIDE_NPTS  scan window (1560/100/16001)
    TM_WIDE_TOL_FRAC    spatial-FWHM early-stop fraction  (default 0.02)
    TM_WIDE_MAX_EVALS   total TM evaluations cap          (default 7)

Outputs land in results/tm_wide_mode/ on Athena and sync to
results_from_athena/tm_wide_mode/ via `--results-no-fsp`:
    wide_mode_secant_log.csv               every evaluation, in order
    result_tm_wide_mode_summary.mat        target/best + history
    fwhm_vs_corrugation_wide.png           FWHM(corr) with 1/fwhm-linear fit
    field_envelope_wide_best.png           the matched ~80 µm mode envelope
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the proven FDTD-eval + caching + logging + envelope helpers.
from runners.tm import tm_match_corrugation_bisect as tmc

# Auto-discovery contract: top-level `run` + dedicated study folder. Make the
# folder HEIGHT-AWARE: _evaluate() stamps cache files as result_corrmatch_tm_C<pm>
# (corrugation only, NOT height), so a thin-guide (e.g. 200 nm) run MUST land in a
# separate folder or it would wrongly reuse the 350 nm device's cached evals.
_HEIGHT_NM = float(os.environ.get("TM_HEIGHT_NM", "350"))
if abs(_HEIGHT_NM - 350.0) < 0.5:
    STUDY_DIR_NAME = "tm_wide_mode"
else:
    # Thin-guide RETARGETS reuse the SAME corr-keyed cache filenames
    # (result_corrmatch_tm_C<pm>.mat — corrugation only, NOT pitch) at DIFFERENT
    # pitches (e.g. 200 nm guide at 1550 vs 1590 nm), so the folder must ALSO be
    # pitch-aware or a new-pitch run would wrongly reuse the old-pitch cache.
    # TM_WIDE_PITCH_NM is forwarded by deploy; default matches PITCH_M below.
    _PITCH_NM_DIR = float(os.environ.get("TM_WIDE_PITCH_NM") or "516.83")
    STUDY_DIR_NAME = f"tm_wide_mode_H{_HEIGHT_NM:.0f}_P{int(round(_PITCH_NM_DIR))}"

# ── Search / device constants (env-overridable) ──────────────────────────────
TARGET_FWHM_M = float(os.environ.get("TM_WIDE_TARGET_UM", "80")) * 1e-6
N_SIDE        = int(os.environ.get("TM_WIDE_N", "300"))
PITCH_M       = float(os.environ.get("TM_WIDE_PITCH_NM") or "516.83") * 1e-9  # `or` not default: deploy forwards this empty when unset
# Thin guides couple to the sidewall corrugation FAR weaker (TM near cutoff), so
# 80 um needs much DEEPER corrugation than the well-confined 350 nm guide
# (350 nm: ~68 nm; 200 nm: measured ~220 nm). Search range is therefore set per
# height. All three remain env-overridable (`or` guards the deploy empty-string).
if abs(_HEIGHT_NM - 350.0) < 0.5:
    _SEEDS_DEF, _CMIN_DEF, _CMAX_DEF = "60,100", "40", "200"
else:                                    # thin guide -> deeper corrugation band
    _SEEDS_DEF, _CMIN_DEF, _CMAX_DEF = "180,260", "100", "340"
SEEDS_NM      = tuple(float(s) for s in
                      (os.environ.get("TM_WIDE_SEEDS_NM") or _SEEDS_DEF).split(","))
CORR_MIN_NM   = float(os.environ.get("TM_WIDE_CORR_MIN_NM") or _CMIN_DEF)
CORR_MAX_NM   = float(os.environ.get("TM_WIDE_CORR_MAX_NM") or _CMAX_DEF)
TOL_FRAC      = float(os.environ.get("TM_WIDE_TOL_FRAC") or "0.01")   # 1% = ±0.8 µm of 80; `or` guards empty-string
MAX_EVALS     = int(os.environ.get("TM_WIDE_MAX_EVALS") or "9")       # headroom for the tighter target

# Wide, recenterable window: a shallow corrugation raises the average index and
# shifts the TM resonance UP a few nm from the anchored ~1556 nm; this band
# contains it with margin. Spectral points are nearly free in FDTD.
# `or` (not .get default): deploy forwards these present-but-EMPTY when unset, and
# float("")/int("") would crash — the `or` falls through empty strings to the default.
CENTER_M = float(os.environ.get("TM_WIDE_CENTER_NM") or "1560") * 1e-9
WIDTH_NM = float(os.environ.get("TM_WIDE_WIDTH_NM") or "100")
N_POINTS = int(os.environ.get("TM_WIDE_NPTS") or "16001")


# ═══════════════════════════════════════════════════════════════════════════════
# Pure secant root-find on the LINEAR coordinate 1/fwhm vs corrugation
# (unit-testable; no FDTD).
# ═══════════════════════════════════════════════════════════════════════════════

def _secant_invfwhm(eval_fwhm, target_m, seeds_nm, c_min_nm, c_max_nm,
                    tol_frac, max_evals):
    """Find corrugation c (nm) with eval_fwhm(c)=spatial FWHM (m) closest to
    target_m. Roots g(c)=1/fwhm(c)-1/target in (c, 1/fwhm) space, which is ~linear
    because fwhm=ln2/kappa and kappa∝c. Returns (best_c_nm, evals{c_nm: fwhm_m}).

    DIRECTION falls out of the secant automatically: larger c -> stronger kappa ->
    smaller fwhm -> larger 1/fwhm. Global-closest fallback if linearity is poor.
    """
    evals = {}
    g_t = 1.0 / target_m

    def F(c_nm):
        c_nm = round(min(c_max_nm, max(c_min_nm, float(c_nm))), 3)
        if c_nm not in evals:
            evals[c_nm] = eval_fwhm(c_nm)
        return c_nm, evals[c_nm]

    pts = []  # (c_nm, g=1/fwhm)
    for s in seeds_nm:
        c, fw = F(s)
        if abs(fw - target_m) <= tol_frac * target_m:
            return c, evals
        pts.append((c, 1.0 / fw))

    # Robustness: the secant step below indexes pts[-2], so it needs TWO points. A
    # single usable seed leaves only one — e.g. a comma-delimited TM_WIDE_SEEDS_NM
    # gets truncated to one value by sbatch --export (itself comma-separated), or two
    # seeds round to the same corrugation. Synthesize the second point from the
    # through-origin model 1/fwhm ∝ corr (fwhm=ln2/kappa, kappa∝corr) =>
    # corr2 = corr1·fwhm1/target. This both prevents an IndexError crash (which wastes
    # a completed GPU eval) and is a physics-good first step toward the target.
    if len(pts) == 1:
        c1, g1 = pts[0]
        c2_guess = c1 * (1.0 / g1) / target_m       # = corr1 * fwhm1 / target
        c, fw = F(c2_guess)
        if abs(fw - target_m) <= tol_frac * target_m:
            return c, evals
        pts.append((c, 1.0 / fw))

    while len(evals) < max_evals:
        (c1, g1), (c2, g2) = pts[-2], pts[-1]
        if g2 == g1:                       # flat -> cannot step; bail to fallback
            break
        c_next = c2 + (g_t - g2) * (c2 - c1) / (g2 - g1)
        before = len(evals)
        c, fw = F(c_next)
        if abs(fw - target_m) <= tol_frac * target_m:
            return c, evals
        if len(evals) == before:           # clamped onto an existing point -> stall
            break
        pts.append((c, 1.0 / fw))

    best = min(evals, key=lambda c: abs(evals[c] - target_m))
    return best, evals


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting / summary
# ═══════════════════════════════════════════════════════════════════════════════

def _write_outputs(results_dir, target_m, tm_points, best):
    tm_sorted = sorted(tm_points, key=lambda r: r["corr_nm"])
    Cs = np.array([r["corr_nm"] for r in tm_sorted])
    Ws = np.array([r["fwhm_um"] for r in tm_sorted])
    tgt_um = target_m * 1e6

    # FWHM(corr): data + the 1/fwhm-linear model used by the secant.
    fig, ax = plt.subplots()
    ax.plot(Cs, Ws, "s", ms=8, mfc="w", color="C0", label="TM evaluations")
    if Cs.size >= 2:
        a, b = np.polyfit(Cs, 1.0 / Ws, 1)        # 1/fwhm ≈ a·corr + b (linear)
        cc = np.linspace(Cs.min(), Cs.max(), 200)
        ax.plot(cc, 1.0 / (a * cc + b), "-", lw=1.3, color="C0", alpha=0.6,
                label="1/FWHM linear fit")
    ax.axhline(tgt_um, color="C3", ls="--", lw=1.6, label=f"target {tgt_um:.0f} µm")
    ax.axvline(best["corr_nm"], color="0.4", ls=":", lw=1.4,
               label=f"corr = {best['corr_nm']:.1f} nm")
    ax.set_xlabel("TM corrugation depth [nm]")
    ax.set_ylabel("Spatial mode FWHM [µm]")
    ax.set_title(f"TM wide-mode search (N={N_SIDE}/side)")
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "fwhm_vs_corrugation_wide.png"),
                dpi=150, bbox_inches="tight"); plt.close(fig)

    # The matched ~target-µm mode envelope.
    x, env = tmc._envelope(best["mat"])
    if x is not None:
        fig, ax = plt.subplots()
        env = env / np.max(env)
        ax.plot(x * 1e6, env, "-", lw=1.6, color="C0",
                label=f"corr={best['corr_nm']:.0f}nm  FWHM={best['fwhm_um']:.2f} µm")
        ax.axhline(0.5, color="0.6", ls=":", lw=1.0, label="half-max")
        ax.set_xlabel("x [µm]"); ax.set_ylabel("Normalized energy envelope")
        ax.set_title(f"TM wide mode (target {tgt_um:.0f} µm, N={N_SIDE}/side)")
        ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "field_envelope_wide_best.png"),
                    dpi=150, bbox_inches="tight"); plt.close(fig)

    summary = {
        "target_fwhm_um": tgt_um, "n_periods_each_side": N_SIDE,
        "pitch_nm": PITCH_M * 1e9, "n_core": tmc.N_CORE, "n_clad": tmc.N_CLAD,
        "device_length_um": 2.0 * N_SIDE * PITCH_M * 1e6,
        "best_corr_nm": best["corr_nm"], "best_fwhm_um": best["fwhm_um"],
        "delta_fwhm_um": best["fwhm_um"] - tgt_um,
        "best_spectral_fwhm_nm": best["spectral_fwhm_nm"],
        "best_stopband_nm": best["stopband_nm"],
        "best_lambda_nm": best["lambda_nm"], "best_Q": best["Q"], "best_T": best["T"],
        "n_tm_evals": len(tm_points),
        "history_corr_nm": Cs, "history_fwhm_um": Ws,
    }
    sio.savemat(os.path.join(results_dir, "result_tm_wide_mode_summary.mat"), summary)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg=None):
    """Unattended secant search on the cluster. `cfg` from the dispatcher is
    ignored; the device is rebuilt from the pinned TM baseline (index forced to
    1.97/1.444) so dispatcher tweaks aren't inherited."""
    import config
    from runners.tm._tm_vs_te_common import build_base_cfg
    from simulation_config import SimulationConfig

    # Drive the reused tmc._evaluate / run_one_scan with OUR device + window by
    # overriding the module-level constants it reads (N and the scan window).
    tmc.N = N_SIDE
    tmc.CENTER_M = CENTER_M
    tmc.WIDTH_NM = WIDTH_NM
    tmc.N_POINTS = N_POINTS
    tmc.TOL_FWHM_FRAC = TOL_FRAC          # used only for the CSV "decision" column

    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "wide_mode_secant_log.csv")
    tmc._init_log(log_path)

    base = build_base_cfg(SimulationConfig())
    base.material.n_core_const = tmc.N_CORE      # override build_base_cfg legacy 1.9963
    base.material.n_clad_const = tmc.N_CLAD
    base.run.cleanup_lumerical_data = True       # multi-run job: don't fill NFS with .h5

    cache = tmc._load_cache(results_dir)
    order_box = [sum(1 for _ in open(log_path, encoding="utf-8")) - 1]

    print("=" * 70)
    print(f"[wide] results_dir = {results_dir}")
    print(f"[wide] target FWHM = {TARGET_FWHM_M*1e6:.1f} µm | N={N_SIDE}/side "
          f"(L={2*N_SIDE*PITCH_M*1e6:.0f} µm) | pitch={PITCH_M*1e9:.2f}nm | "
          f"n_core={tmc.N_CORE} n_clad={tmc.N_CLAD}")
    print(f"[wide] window {CENTER_M*1e9:.0f}±{WIDTH_NM/2:.0f}nm, {N_POINTS} pts | "
          f"corr [{CORR_MIN_NM:.0f},{CORR_MAX_NM:.0f}]nm | seeds {SEEDS_NM}")
    # Containment sanity: half-device must be several mode-FWHMs.
    half_um = N_SIDE * PITCH_M * 1e6
    print(f"[wide] half-device {half_um:.0f} µm = {half_um/(TARGET_FWHM_M*1e6):.1f}× "
          f"target FWHM (>~2 keeps the mode kappa-limited, not truncated)")
    print("=" * 70)

    # Single-corrugation DIAGNOSTIC mode (no search): evaluate one corrugation and
    # return — used to isolate-test e.g. a sim-time/boundary change (TM_SIM_TIME_PS,
    # SPAN_MULT) against a fixed geometry. Set TM_WIDE_SINGLE_CORR_NM=<nm>.
    single = os.environ.get("TM_WIDE_SINGLE_CORR_NM")
    if single:
        c_nm = float(single)
        print(f"[wide] SINGLE-CORR diagnostic: corr={c_nm} nm, sim_time="
              f"{os.environ.get('TM_SIM_TIME_PS', '2000')} ps, SPAN_MULT="
              f"{os.environ.get('SPAN_MULT', 'default')} (no search)")
        rec = tmc._evaluate(base, "TM", c_nm * 1e-9, PITCH_M, cache,
                            log_path, TARGET_FWHM_M, order_box)
        print(f"[wide] RESULT corr={c_nm}nm -> FWHM={rec['fwhm_um']:.3f}um  "
              f"T={rec['T']:.4f}  Q={rec['Q']:.1f}  lam={rec['lambda_nm']:.3f}nm")
        return {"single_corr_nm": c_nm, "rec": rec}

    def eval_fwhm(c_nm):
        rec = tmc._evaluate(base, "TM", c_nm * 1e-9, PITCH_M, cache,
                            log_path, TARGET_FWHM_M, order_box)
        fw = rec["fwhm_m"]
        # Degenerate / truncated guard: mode wider than ~half device -> treat as
        # "too shallow" by returning a large fwhm so the secant increases corr.
        if not (np.isfinite(fw) and fw > 0) or fw > 0.9 * half_um * 1e-6:
            return 10.0 * TARGET_FWHM_M
        return fw

    best_c_nm, _evals = _secant_invfwhm(
        eval_fwhm, TARGET_FWHM_M, SEEDS_NM, CORR_MIN_NM, CORR_MAX_NM,
        TOL_FRAC, MAX_EVALS)

    tm_points = [rec for (pol, _c), rec in cache.items() if pol == "TM"]
    best = cache[("TM", int(round(best_c_nm * 1e-9 * 1e12)))]

    _write_outputs(results_dir, TARGET_FWHM_M, tm_points, best)

    truncated = best["fwhm_m"] > 0.5 * (2.0 * N_SIDE * PITCH_M)
    print("\n" + "=" * 70)
    print(f"TARGET spatial FWHM = {TARGET_FWHM_M*1e6:.1f} µm  (N={N_SIDE}/side, "
          f"pitch {PITCH_M*1e9:.2f} nm)")
    print(f"BEST CORRUGATION = {best['corr_nm']:.2f} nm: FWHM = {best['fwhm_um']:.3f} µm "
          f"(Δ = {best['fwhm_um']-TARGET_FWHM_M*1e6:+.3f} µm)")
    print(f"  λ_res = {best['lambda_nm']:.3f} nm | peak T = {best['T']:.4f} | "
          f"Q = {best['Q']:.1f} | stopband = {best['stopband_nm']:.3f} nm")
    if truncated:
        print("  WARNING: FWHM exceeds half the device — increase TM_WIDE_N.")
    print(f"TM evaluations: {len(tm_points)}   log: {log_path}")
    print("=" * 70)
    return {"target_fwhm_um": TARGET_FWHM_M * 1e6, "best": best}


if __name__ == "__main__":
    # Local smoke test of the PURE secant only (no lumapi/FDTD): synthetic
    # fwhm(corr)=ln2/kappa with kappa∝corr, so 1/fwhm is exactly linear and the
    # secant must converge in ~1 step past the two seeds.
    K_PER_NM = 0.693 / (300.0 * 18.5e-6)         # calibrate to TM@300nm -> 18.5 µm
    def synth(c_nm):
        return 0.693 / (K_PER_NM * c_nm)         # = ln2/kappa, kappa = K_PER_NM·c

    target = 80e-6
    best, evals = _secant_invfwhm(synth, target, (60.0, 100.0),
                                  40.0, 200.0, 0.0, 7)
    print(f"[selftest] evals(nm)={sorted(round(c,2) for c in evals)}")
    print(f"[selftest] best corr={best:.3f} nm -> fwhm={synth(best)*1e6:.4f} µm "
          f"(expect {0.693/(K_PER_NM*80e-6):.3f} nm)")
    assert abs(synth(best) - target) <= 1e-9, synth(best)
    # Linear-exact => should need only ONE secant step beyond the 2 seeds.
    assert len(evals) <= 3, f"too many evals: {len(evals)}"
    # Single-seed (the comma-truncation case): must synthesize a 2nd point from the
    # through-origin model and still converge — no IndexError crash.
    best1, evals1 = _secant_invfwhm(synth, target, (60.0,), 40.0, 200.0, 0.0, 7)
    assert abs(synth(best1) - target) <= 1e-9, synth(best1)
    assert len(evals1) <= 3, f"single-seed too many evals: {len(evals1)}"
    print(f"[selftest] single-seed best={best1:.3f} nm, |evals|={len(evals1)}")
    print("[selftest] PASS")
