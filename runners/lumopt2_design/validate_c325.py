"""Validation gates B0-B4 for the corr-325 lumopt2 campaign (run BEFORE any campaign).

Study dir: runners/lumopt2_design/   |   Created 2026-08-13   |   Job(s): TBD
Purpose: prove the cost function reader, the geometry map, and the adjoint
gradients actually work before spending GPU-days (user rule 2026-08-13:
"make sure our cost function is actually working, verify what you do").

LOCAL (zero GPU, run with `python -m runners.lumopt2_design.validate_c325`):
  B0  soft-max reader on STORED .mat spectra — must reproduce the measured
      ordering comb270 > hedge536 > ctrl > comb90 and be linewidth-blind.
  B1  build-only smoke: base .fsp builds, object-name map holds, the
      Parametrization func reproduces the as-built scene at the seed, per-tooth
      widths match a builder-built scene, shift algebra is contiguous.

CLUSTER (dispatch, 4 array tasks ≈ 1 + 1 + ~6 + ~9 GPU-h):
  task 0  B2a  bare canary vs STORED N=100 anchor (IGUM 51736:
               T 0.9104 / λ 1559.006 / Q 1760 / mode 19.24 µm — not re-run)
  task 1  B2b  seed-comb canary → calibrates σ0 for the width tripwire
  task 2  B3   validate_gradient on 6 params (corr_1, corr_25, shift_1,
               r_29, x_29, d_comb) + PASS gates: sign 6/6, scale α∈[0.8,1.25]
  task 3  B4   known-answer mini-opt: ONE free param = global comb δx seeded
               at 300 nm (detuned); must move toward the measured 401 (270°)
Dispatch:
    SBATCH_MEM=160G ARRAY_TIME=24:00:00 bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.validate_c325
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng

SPEC = eng.CampaignSpec(label="lumopt2_val_c325")
N_TASKS = 21         # 0=B2a bare | 1=B2b comb | 2=B3 gradients | 3=B4 mini-opt
                     # gradient-fix experiment matrix (all B3-style, point 1
                     # unless noted, gates = per-class α vs FD):
                     # 4=α-stability point 2 | 5=co-location (option 3)
                     # 6=boundary patch (option 2) | 7=patch+co-location
                     # 8=★quadrature probe pt1 (adjoint-only, 2 sims)
                     # 9=★quadrature probe pt2 (C-universality check, 2 sims)
                     # ── V2 width-gradient gates (2026-08-21, V2_FWHM_PLAN §5) ──
                     # 10=W2 v2 canary (width_grad fwd, ±5nm/501 window; prints
                     #    the measured softW/fwhm pair = the campaign anchors)
                     # 11=W1r toy campaign (2-3 evals THROUGH run_campaign incl.
                     #    the completion path — item 23's lesson)
                     # 12=W3a width-gradient FD gate (wg_pure: J=−softW; naive
                     #    C_field=(1,0); FD first — ~12 sims)
                     # 13=W3b adjoint-only quadrature C_field=(0,1) (2 sims;
                     #    fit C_field offline from 12+13, task-8 recipe)
                     # ── mesh-correction revalidation (2026-08-22, tasks 14-17) ──
                     # 14=MX identity: bare+conformal+pitch-locked vs STORED family
                     # 15=MX v2.1 anchor (comb origin, PVA, dx 51.683)
                     # 16=MX retrim d+52.5   17=MX rho-neutral a=1.5
                     # 18=MX-GRAD: B3 tooth-gradient FD at pitch-locked+CONFORMAL
                     #    (the old staircase verdict was measured at the
                     #    MISALIGNED 50nm grid; sane tooth alpha here => v2.1
                     #    campaigns can run in the production convention)

# ── V2 anchors — MEASURED by W2 (job 135971 task 10, 2026-08-21): the
# seed-comb device at the v2 window, through the full v2 stack.
# softw = the single-λ twin monitor's own sample (softw_adj_um — the FOM
# carrier; broadband softW read 18.1709, Δ 0.010 µm = twin consistency).
# fwhm_env = 17.7136 (stored-fsp recovery read 17.7005 at ±3nm/301 — the
# +0.013 µm is the window/numerics delta, now anchored). W2 gate: T 0.8905
# vs 0.8912 anchor, λ 1564.274.
PROV_SOFTW_UM = 18.160689
PROV_FWHM0_UM = 17.713551


def _w_spec(suffix, **kw):
    """V2 gate spec: width_grad on, wider recording window (±5 nm @ 20 pm =
    501 pts — the adopted v2 window, a NAMED §2 change anchored by task 10)."""
    import dataclasses
    return dataclasses.replace(
        SPEC, label=SPEC.label + suffix, width_grad=True,
        scan_width_nm=10.0, n_wl_points=501,
        fwhm0_um=PROV_FWHM0_UM,
        wg_anchor={"softw": PROV_SOFTW_UM, "fwhm": PROV_FWHM0_UM}, **kw)

# Stored anchors (MEASURED, never re-run — CLAUDE.md §6)
ANCHOR_BARE_N100 = {"t_pk": 0.9104, "lam_nm": 1559.006, "q": 1760.0, "tol_t": 0.005,
                    "tol_lam": 0.05, "tol_q_rel": 0.05}
# ★MEASURED 2026-08-14 (job 132631): the lumopt2 stack is a NAMED §2 numerics
# change vs the stored family — the optimization-region mesh override (uniform
# dx=dy=dz=50 nm, required by lumopt2's locked dEps grid) shifts the bare
# N=100 point to T 0.9126 / λ 1558.634 / Q 1661 (λ −372 pm, Q −5.6% vs stored).
# The campaign therefore uses ITS OWN in-study anchors below; stored-anchor
# deltas are printed for the record, not gated. The physics cross-check at
# identical numerics PASSED: comb−bare ΔT +0.0107 (vs +0.0105 stored family),
# comb Q_i +14.7% (vs +14%), width ratio 0.999 (width-neutral ✓).
# ★PVA anchors (job 132654, 2026-08-15) — mesh refinement is now "precise
# volume average" in the lumopt2 path (B3 job 132637 showed conformal-0
# staircases tooth dEps). PVA moved λ +5.2 nm; internal physics reproduces:
# comb−bare ΔT +0.0112 (family +0.0105), Q_i +11 %, width ratio 0.9993.
# (obsolete conformal anchors, job 132631: bare 0.9126/1558.634/1661/18.378;
#  comb 0.9233/1558.634/1670/18.360)
ANCHOR_LUMOPT2_BARE = {"t_pk": 0.8800, "lam_nm": 1564.213, "q": 2024.0,
                       "sigma_um": 17.505}
ANCHOR_LUMOPT2_COMB = {"t_pk": 0.8912, "lam_nm": 1564.213, "q": 2036.0,
                       "sigma_um": 17.493}   # σ0 for the campaign tripwire
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ATH = os.path.join(_ROOT, "results_from_athena", "comb_q3db", "results")
_IGM = os.path.join(_ROOT, "results_from_igum", "tm_nladder_c325", "results")
B0_FILES = {   # measured T at N=165 (job 130458) in comments
    "ctrl":    (os.path.join(_ATH, "result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat"), 0.4906),
    "comb270": (os.path.join(_ATH, "result_N165_TM_avg_C325_Ybox8p0_Zbox8p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat"), 0.5361),
    "comb90":  (os.path.join(_ATH, "result_N165_TM_avg_C325_Ybox8p0_Zbox8p8_scR80_arr57_X-14732to15004_Y1900to1900_C325_pair.mat"), 0.4371),
    "hedge536": (os.path.join(_ATH, "result_N165_TM_avg_C325_Ybox8p0_Zbox8p8_scR80_arr57_X-14604to15412_Y1900to1900_C325_pair.mat"), 0.5283),
    "bareN100": (os.path.join(_IGM, "result_N100_TM_avg_C325_Ybox8p0_Zbox8p8.mat"), 0.9104),
}


def _softmax_from_spectrum(wl_nm, T, win_mult=eng.WIN_FWHM_MULT):
    """The exact reader math on a plain spectrum (window multiplier exposed
    so B0 can test window-doubling insensitivity)."""
    lam_pk, t_pk, fwhm = eng.measure_peak(wl_nm, T)
    if fwhm is None:
        raise RuntimeError("no FWHM crossings in window")
    idx = np.abs(np.asarray(wl_nm) - lam_pk) <= win_mult * fwhm
    fom = float(np.mean(np.asarray(T)[idx] ** eng.P_SOFTMAX) ** (1.0 / eng.P_SOFTMAX))
    return fom, lam_pk, t_pk, fwhm


def gate_b0():
    """Reader + penalty on stored data. Prints a verdict per check."""
    from scipy.io import loadmat
    print("== B0: soft-max reader on stored spectra ==")
    # Mean-form reader scaling: doubling the window halves mean(T^p), so the
    # FOM shifts by exactly 2^(-1/p)-1 (~-5.6% at p=12). A CONSTANT factor —
    # cancels in every comparison. The real linewidth-blindness test is
    # FOM/T_peak being the same constant (~0.78, DERIVED) for every device.
    drift_expected = 2.0 ** (-1.0 / eng.P_SOFTMAX) - 1.0
    foms, ratios, ok = {}, {}, True
    for name, (path, t_meas) in B0_FILES.items():
        m = loadmat(path)
        wl, T = np.squeeze(m["wl_nm"]), np.squeeze(m["T"])
        fom, lam_pk, t_pk, fwhm = _softmax_from_spectrum(wl, T)
        fom2x = _softmax_from_spectrum(wl, T, win_mult=2 * eng.WIN_FWHM_MULT)[0]
        drift = fom2x / fom - 1.0
        foms[name], ratios[name] = fom, fom / t_pk
        line_ok = (abs(t_pk - t_meas) < 0.01
                   and abs(drift - drift_expected) < 0.01)
        ok &= line_ok
        print(f"  {name:9s} T_pk {t_pk:.4f} (meas {t_meas})  FOM {fom:.4f} "
              f"(FOM/T {ratios[name]:.3f})  win-2x drift {drift:+.4f} "
              f"(expect {drift_expected:+.4f})  λ {lam_pk:.3f} "
              f"fwhm {fwhm * 1e3:.0f} pm   {'ok' if line_ok else 'FAIL'}")
    spread = max(ratios.values()) - min(ratios.values())
    blind_ok = spread < 0.01
    print(f"  linewidth-blindness: FOM/T_peak spread {spread:.4f} across the "
          f"~8x linewidth range {'ok' if blind_ok else 'FAIL'}")
    ok &= blind_ok
    order_ok = foms["comb270"] > foms["hedge536"] > foms["ctrl"] > foms["comb90"]
    print(f"  ordering comb270 > hedge536 > ctrl > comb90: {'ok' if order_ok else 'FAIL'}")
    p0 = eng.seed_params(SPEC)
    pen0 = float(eng.kappa_penalty(p0))
    p_hi, p_lo = p0.copy(), p0.copy()
    p_hi[eng.SL_CORR] = 325.0 * 1.05          # rho = 1.05 (above deadband)
    p_lo[eng.SL_CORR] = 325.0 * 0.90          # rho = 0.90 (below deadband)
    g_hi = eng._kappa_penalty_grad(p_hi)[eng.SL_CORR]
    g_lo = eng._kappa_penalty_grad(p_lo)[eng.SL_CORR]
    pen_ok = (pen0 == 0.0 and eng.kappa_penalty(p_hi) > 0 and eng.kappa_penalty(p_lo) > 0
              and np.all(g_hi > 0) and np.all(g_lo < 0))
    print(f"  penalty: seed {pen0} | rho1.05 {float(eng.kappa_penalty(p_hi)):.4f} "
          f"(grad>0 {np.all(g_hi > 0)}) | rho0.90 {float(eng.kappa_penalty(p_lo)):.4f} "
          f"(grad<0 {np.all(g_lo < 0)})   {'ok' if pen_ok else 'FAIL'}")
    verdict = ok and order_ok and pen_ok
    print(f"== B0 {'PASS' if verdict else 'FAIL'} ==")
    return verdict


def _read_props(fdtd, names):
    """Current (x, x span, y span) of the named rects, in nm."""
    out = {}
    for n in names:
        out[n] = tuple(float(np.squeeze(fdtd.getnamed(n, k))) / eng.NM
                       for k in ("x", "x span", "y span"))
    return out


def gate_b1(workdir=None):
    """Local build smoke + geometry equivalence (no simulation, silent)."""
    print("== B1: build smoke + geometry equivalence ==")
    import tempfile
    workdir = workdir or os.path.join(tempfile.gettempdir(), "lumopt2_b1")
    os.makedirs(workdir, exist_ok=True)
    fsp = os.path.join(workdir, "b1_base.fsp")
    wl = eng.build_base_fsp(SPEC, fsp)             # asserts the name map itself
    assert len(wl) == SPEC.n_wl_points, f"λ grid {len(wl)} != {SPEC.n_wl_points}"

    sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
    import lumapi
    names, cavity = eng.tooth_names(SPEC.n_periods_side)
    tooth_objs = [n for quad in names.values() for n in quad] + [cavity]
    func = eng.make_func(SPEC)
    p0 = eng.seed_params(SPEC)

    # 1) func(seed) must reproduce the as-built scene (walk algebra vs builder)
    with lumapi.FDTD(filename=fsp, hide=True) as fdtd:
        built = _read_props(fdtd, tooth_objs)
        props0 = {k: eng._plain(v) for k, v in func(p0).items()}
        worst = 0.0
        for obj, (x, xs, ys) in built.items():
            for prop, val in (("x", x), ("x span", xs), ("y span", ys)):
                key = f"{obj}::{prop}"
                if key in props0:
                    worst = max(worst, abs(float(props0[key]) / eng.NM - val))
        print(f"  func(seed) vs as-built: worst discrepancy {worst:.4f} nm")
        seed_ok = worst < 0.1

        # 2) perturbed widths: apply func, compare vs a builder-built scene
        rng = np.random.default_rng(0)
        p1 = p0.copy()
        p1[eng.SL_CORR] += rng.uniform(-50, 120, eng.N_FREE)
        p1[eng.SL_AVG] += rng.uniform(-20, 20, eng.N_FREE)
        for k, v in func(p1).items():
            obj, prop = k.split("::")
            fdtd.setnamed(obj, prop, float(eng._plain(v)))
        applied = _read_props(fdtd, tooth_objs)

    cfg = eng.build_base_cfg(SPEC)
    w_n = (p1[eng.SL_AVG] - p1[eng.SL_CORR] / 2.0) * eng.NM
    w_w = (p1[eng.SL_AVG] + p1[eng.SL_CORR] / 2.0) * eng.NM
    cfg.grating.width_narrow_per_tooth_m = list(w_n)
    cfg.grating.width_wide_per_tooth_m = list(w_w)
    from bragg_device import PiShiftBraggFDTD
    sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())
    try:
        sim.build()
        ref = _read_props(sim.fdtd, tooth_objs)
    finally:
        sim.close()
    worst2 = max(abs(a - b) for obj in tooth_objs
                 for a, b in zip(applied[obj], ref[obj]))
    print(f"  func(widths) vs builder per-tooth build: worst discrepancy {worst2:.4f} nm")
    width_ok = worst2 < 0.1

    # 3) shift algebra self-check (pure python, no lumapi)
    p2 = p0.copy()
    p2[eng.SL_SHIFT] = rng.uniform(0, 150, eng.N_FREE)
    pr = {k: float(eng._plain(v)) / eng.NM for k, v in func(p2).items()}
    hp, l0 = eng.PITCH_NM / 2.0, eng.PITCH_NM / 2.0
    x_out = l0 / 2.0 + eng.N_FREE * eng.PITCH_NM
    def edges(o):
        return pr[f"{o}::x"] - pr[f"{o}::x span"] / 2.0, pr[f"{o}::x"] + pr[f"{o}::x span"] / 2.0
    gaps = []
    prev_end = -x_out
    for d in range(eng.N_FREE, 0, -1):
        for o in (names[d][0], names[d][1]):
            lo_e, hi_e = edges(o)
            gaps.append(abs(lo_e - prev_end))
            prev_end = hi_e
    gaps.append(abs(edges(cavity)[0] - prev_end))
    prev_end = edges(cavity)[1]
    for d in range(1, eng.N_FREE + 1):
        for o in (names[d][2], names[d][3]):
            lo_e, hi_e = edges(o)
            gaps.append(abs(lo_e - prev_end))
            prev_end = hi_e
    edge_err = abs(prev_end - x_out)
    cav_err = abs(pr[f"{cavity}::x span"] - (l0 + 2.0 * float(np.sum(p2[eng.SL_SHIFT]))))
    print(f"  shifts: max contiguity gap {max(gaps):.2e} nm, frozen-edge error "
          f"{edge_err:.2e} nm, cavity absorption error {cav_err:.2e} nm")
    shift_ok = max(gaps) < 1e-6 and edge_err < 1e-6 and cav_err < 1e-6

    verdict = seed_ok and width_ok and shift_ok
    print(f"== B1 {'PASS' if verdict else 'FAIL'} ==")
    return verdict


# ═══ cluster tasks ═══════════════════════════════════════════════════════════

def main(task_idx):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    if task_idx == 0:                              # B2a — bare anchor canary
        import dataclasses
        # (The PVA λ was hunted with a 20 nm window in job 132654 after the
        # narrow window saw only stopband floor — job 132652. SPEC is now
        # centered on the measured PVA λ, so the standard narrow window works.)
        spec = dataclasses.replace(SPEC, label=SPEC.label + "_bare", bare=True,
                                   free_comb=False)
        row = eng.run_canary(spec, out_dir)
        a, s = ANCHOR_BARE_N100, ANCHOR_LUMOPT2_BARE
        # Hard gate = reproduce the lumopt2 in-study anchor (same numerics);
        # stored-family deltas are the DOCUMENTED named-numerics offset.
        ok = (abs(row["t_pk"] - s["t_pk"]) < 0.003
              and abs(row["lam_pk_nm"] - s["lam_nm"]) < 0.05
              and abs(row["q_loaded"] / s["q"] - 1.0) < 0.03)
        print(f"B2a vs lumopt2 in-study anchor: {'PASS' if ok else 'FAIL'} "
              f"(anchor T {s['t_pk']} λ {s['lam_nm']} Q {s['q']:.0f}); "
              f"offset vs stored family: dT {row['t_pk'] - a['t_pk']:+.4f} "
              f"dλ {row['lam_pk_nm'] - a['lam_nm']:+.3f} nm "
              f"dQ {100 * (row['q_loaded'] / a['q'] - 1):+.1f}%")
    elif task_idx == 1:                            # B2b — seed-comb canary (σ0)
        row = eng.run_canary(SPEC, out_dir)
        print(f"B2b σ0 for the campaign tripwire: sigma_um = {row.get('sigma_um')}")
    elif task_idx == 2:                            # B3 — gradient validation
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB,
                   eng.I_CAV]
        eng.run_validate_gradient(SPEC, out_dir, indices)
        print("B3 gates (apply by hand from the printout): sign agreement 6/6; "
              "adjoint/FD scale α ∈ [0.8, 1.25]; vec-error ≤ 0.15")
    elif task_idx == 3:                            # B4 — known-answer δx recovery
        _b4_mini_opt(out_dir)
    elif task_idx in (4, 5, 6, 7):                 # gradient-fix experiments
        import dataclasses
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB,
                   eng.I_CAV]
        variant = {
            4: dict(detune=2, spec=SPEC),          # α stability (option 1 basis)
            5: dict(detune=1, spec=dataclasses.replace(
                SPEC, label=SPEC.label + "_coloc", colocate_fields=True)),
            6: dict(detune=1, spec=dataclasses.replace(
                SPEC, label=SPEC.label + "_bc", bc_patch=True)),
            7: dict(detune=1, spec=dataclasses.replace(
                SPEC, label=SPEC.label + "_bccoloc", bc_patch=True,
                colocate_fields=True)),
        }[task_idx]
        eng.run_validate_gradient(variant["spec"], out_dir, indices,
                                  detune=variant["detune"])
        print("gates: point-1 naive baseline (132657) α = corr 0.20/0.13, "
              "shift 0.06, comb 0.77/0.77/d-flip. Task 4 PASS = α stable ±30% "
              "at point 2 → calibration viable. Tasks 5-7 PASS = tooth α → "
              "[0.8, 1.25] (5 alone is expected PARTIAL: → ~0.3-0.5).")
    elif task_idx == 8:                            # ★adjoint PHASE FIX (2 sims)
        # Root-cause fix found 2026-08-16 (offline reconstruction): remove the
        # spurious 90° in the adjoint scaling, g = -1 hypothesis. Adjoint-only:
        # compare the printed vector against the STORED detune-1 FD
        # (132657/task5/task6, reproducible ≤2%):
        #   fd ≈ [-3.45e-05, -6.39e-06, +3.99e-05, +4.02e-06, +2.77e-06,
        #         +8.09e-06, +2.75e-05]
        # PASS = sign 7/7 AND per-param |adj/fd| ∈ [0.5, 2] (the offline
        # replica predicts most params within ~25%, corr weakest).
        import dataclasses
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB,
                   eng.I_CAV]
        spec = dataclasses.replace(SPEC, label=SPEC.label + "_phasefix",
                                   adj_phase_fix=True, adj_fix_re=0.0,
                                   adj_fix_im=1.0)
        eng.run_adjoint_only(spec, out_dir, indices, detune=1)
    elif task_idx == 9:                            # ★quadrature probe, point 2
        # Same +1j quadrature as task 8 but at detune=2: combined with task
        # 4's Re{Z2} this solves the (φ, s) fit independently at the second
        # operating point. C is trusted for campaigns only if it matches the
        # detune-1 fit (0.8685+0.1022i) within ~3°/10%.
        import dataclasses
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB,
                   eng.I_CAV]
        spec = dataclasses.replace(SPEC, label=SPEC.label + "_quad2",
                                   adj_phase_fix=True, adj_fix_re=0.0,
                                   adj_fix_im=1.0)
        eng.run_adjoint_only(spec, out_dir, indices, detune=2)
    elif task_idx == 10:                           # W2 — v2 canary + anchors
        row = eng.run_canary(_w_spec("_w2"), out_dir)
        a = ANCHOR_LUMOPT2_COMB
        # The v2 window is a NAMED §2 change — T may shift slightly; gate
        # LOOSELY vs the ±3nm-window anchor and RECORD the deltas. The
        # measured (softw_um, fwhm_env_um) pair below IS the campaign anchor.
        ok = (row.get("t_pk") and abs(row["t_pk"] - a["t_pk"]) < 0.010
              and abs(row["lam_pk_nm"] - a["lam_nm"]) < 0.20
              and row.get("fwhm_env_um") and row.get("softw_um"))
        print(f"W2 {'PASS' if ok else 'FAIL'}: T {row.get('t_pk')} (anchor "
              f"{a['t_pk']}, window-change delta is EXPECTED and now anchored) "
              f"λ {row.get('lam_pk_nm')}  ★CAMPAIGN ANCHORS: softw_adj_um "
              f"{row.get('softw_adj_um')} (the FOM carrier's own sample — use "
              f"THIS for wg_anchor['softw']; broadband softw_um "
              f"{row.get('softw_um')} for reference)  fwhm_env_um "
              f"{row.get('fwhm_env_um')} — write into wg_anchor/fwhm0_um.")
    elif task_idx == 11:                           # W1r — toy campaign, 2-3 evals
        # Exercises: MixedFom fwd+2-adjoint loop, jsonl logging, wg re-anchor
        # path, and the COMPLETION path (item 23: the code that runs once at
        # the end is the least-tested code you own). Tiny trust regions keep
        # the 2 steps physically negligible.
        spec = _w_spec("_w1toy", max_iter=2, max_feval=3,
                       trust_nm={"corr": 2.0, "avg": 2.0, "shift": 2.0,
                                 "r": 1.0, "x": 1.0, "d": 1.0, "wcav": 1.0})
        best = eng.run_campaign(spec, out_dir)
        print(f"W1r COMPLETED THROUGH THE FINISH LINE: best_fom {best['fom']:.5f} "
              f"(value is bookkeeping; the PASS is reaching this line + a "
              f"written _best.json + softw/fwhm rows in the jsonl)")
    elif task_idx == 12:                           # W3a — width-grad FD gate
        # J = −softW (wg_pure) ⇒ FD measures d(softW)/dp; adjoint runs naive
        # C_field=(1,0). Indices span the width-relevant classes (comb is
        # measured width-flat — excluded). perturbation 4 nm: expected ΔsoftW
        # ~0.02-0.08 µm ≫ the 0.03% width noise floor.
        indices = [0, eng.N_FREE - 1, eng.SL_AVG.start, eng.SL_SHIFT.start,
                   eng.I_CAV]
        eng.run_validate_gradient(_w_spec("_w3fd", wg_pure=True), out_dir,
                                  indices, perturbation=4.0)
        print("W3a: FD FIRST (validate_gradient returns (fd, adjoint, err%)). "
              "Fit C_field from tasks 12+13 offline (task-8 recipe); PASS = "
              "post-fit sign 5/5 and per-param residual ≤ ~10%.")
    elif task_idx == 13:                           # W3b — quadrature, C_field
        indices = [0, eng.N_FREE - 1, eng.SL_AVG.start, eng.SL_SHIFT.start,
                   eng.I_CAV]
        eng.run_adjoint_only(_w_spec("_w3quad", wg_pure=True,
                                     adj_fix_field_re=0.0,
                                     adj_fix_field_im=1.0),
                             out_dir, indices)
    elif task_idx in (14, 15, 16, 17):
        # ★MESH-CORRECTION REVALIDATION (user order 2026-08-22, after catching
        # that the region's 50.0 nm broke the device's pitch-locked alignment
        # dx = pitch/10 = 51.683): quantify how much changes and how much
        # does not, and prove lumopt2 == regular physics.
        import dataclasses
        DXP = 51.683
        if task_idx == 14:
            # IDENTITY TEST: bare device, CONFORMAL + pitch-locked, through
            # the full lumopt2 stack — must land on the STORED regular anchor
            # (T 0.9104 / λ 1559.006 / FWHM 19.245, never re-run). PASS =
            # λ within ~0.15 nm, FWHM within ~0.5%, T within ~0.005.
            spec = _w_spec("_mx_ident", bare=True, free_comb=False,
                           region_dx_nm=DXP,
                           mesh_refinement="conformal variant 0",
                           scan_center_nm=1559.0)
            spec.width_grad = False     # plain forward physics, no twin needed
            row = eng.run_canary(spec, out_dir)
            print(f"MX-IDENT vs stored family (0.9104/1559.006/19.2448): "
                  f"T {row.get('t_pk')} λ {row.get('lam_pk_nm')} "
                  f"FWHM {row.get('fwhm_env_um')}")
        else:
            base = dataclasses.replace(SPEC, scan_width_nm=10.0, n_wl_points=501,
                                       region_dx_nm=DXP)
            if task_idx == 15:      # corrected-mesh v2.1 anchor (comb origin)
                spec = dataclasses.replace(base, label=SPEC.label + "_mx_origin")
            elif task_idx == 16:    # retrim-verdict-class design at d+52.5
                from runners.lumopt2_design.best_designs import BEST_T9635
                p = np.asarray(BEST_T9635, dtype=float).copy()
                p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + 52.5, SPEC.corr_max_nm)
                spec = dataclasses.replace(base, label=SPEC.label + "_mx_retrim")
                spec.seed_override = tuple(eng.replay_params(spec, p))
            else:                   # 17: rho-neutral a=1.5 (in-band winner)
                from runners.lumopt2_design.rho_neutral_shape import shape_profile
                spec = dataclasses.replace(base, label=SPEC.label + "_mx_rho15",
                                           corr_seed_nm=tuple(shape_profile(1.5)))
            row = eng.run_canary(spec, out_dir)
            print(f"MX task {task_idx} [{spec.label}] at pitch-locked "
                  f"{DXP} nm: T {row.get('t_pk')} λ {row.get('lam_pk_nm')} "
                  f"FWHM {row.get('fwhm_env_um')} — compare vs the 50.0-nm "
                  f"rows (origin 0.8905/17.7136; retrim ~0.960/17.8; "
                  f"rho15 0.9243/18.025) to size the mesh-correction shift")
    elif task_idx == 18:
        # MX-GRAD: does pitch-locked alignment cure conformal's tooth-dEps
        # staircasing (job 132637's alpha 0.07-0.26)? Port-path gradients,
        # naive C=(1,0) — judge SIGNS + per-class alpha spread vs the PVA
        # baseline; the C-recipe re-fit comes after if shapes look sane.
        import dataclasses
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB,
                   eng.I_CAV]
        spec = dataclasses.replace(SPEC, label=SPEC.label + "_mxgrad",
                                   region_dx_nm=51.683,
                                   mesh_refinement="conformal variant 0",
                                   scan_center_nm=1559.0)
        eng.run_validate_gradient(spec, out_dir, indices)
        print("MX-GRAD gates: FD FIRST; compare per-class alpha vs the PVA "
              "point-1 table; PASS-shape = tooth classes no longer 10-30x off")
    elif task_idx == 19:
        # ★GPU-IMPORT-SOURCE TEST (user challenge 2026-08-22: only the
        # FieldRegion object is proven GPU-rejected; a standard import source
        # is NOT on the unsupported list). W3b-style adjoint-only, GPU lane.
        # PASS = the adjoint SOLVES on GPU (any finite vector) — then the
        # true FWHM gradient costs ~minutes, not 8.7 h, and the in-loop
        # width-gradient architecture reopens. FAIL signature = the same
        # 'invalid configuration argument'.
        i_r = eng.SL_R.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_AVG.start, eng.SL_SHIFT.start,
                   eng.I_CAV]
        spec = _w_spec("_w3gpu", wg_pure=True, wg_source="import",
                       wg_adj_resource="GPU")
        eng.run_adjoint_only(spec, out_dir, indices)
    elif task_idx == 20:
        # W3-GPU: the CORRECTNESS gate for the GPU import-source width
        # adjoint (dispatch only after task 19 proves the mechanism runs).
        # J = -softW (wg_pure) so FD measures d(softW)/dp directly; FD legs
        # are forward-only (~50 min each on A100) => ~9 h for 5 params + the
        # adjoint. Feed the printed (fd, adjoint) into fit_c_field.py with
        # the task-13-style quadrature run for the phase.
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]   # 3 classes; quota-safe
        eng.run_validate_gradient(_w_spec("_w3gpufd", wg_pure=True,
                                          wg_source="import",
                                          wg_adj_resource="GPU"),
                                  out_dir, indices, perturbation=4.0)
        print("W3-GPU: FD FIRST. PASS = sign 5/5 and per-param residual <=10% "
              "after the C_field fit; then the in-loop width gradient is LIVE "
              "at GPU speed and campaign_v2_seesaw can replace v2proj.")
    else:
        raise ValueError(f"no validation task {task_idx}")


def _b4_mini_opt(out_dir):
    """One free param: global comb offset δx, seeded detuned at 300 nm.
    The optimizer must move toward the measured optimum 401 nm (270°).

    PREEMPTION-RESUME (added after job 132739 lost 8.9 h to a REQUEUE —
    CLAUDE.md §6 critical rule): every evaluation is appended to
    b4_dx_evals.jsonl; a cold restart warm-starts from the best logged δx,
    so a preemption costs ≤ 1 evaluation."""
    import json
    lmpt = eng.import_lumopt2()
    fsp = os.path.join(out_dir, "b4_base.fsp")
    os.makedirs(out_dir, exist_ok=True)
    wl_nm = eng.build_base_fsp(SPEC, fsp)

    log_path = os.path.join(out_dir, "b4_dx_evals.jsonl")
    start_dx, best_logged = 300.0, None
    if os.path.exists(log_path):
        with open(log_path) as f:
            rows = [json.loads(x) for x in f]
        if rows:
            best_logged = max(rows, key=lambda r: r["fom"])
            start_dx = float(best_logged["dx"])
            print(f"B4 RESUME: {len(rows)} logged evals, restarting from "
                  f"best δx {start_dx:.1f} (FOM {best_logged['fom']:.5f})")

    from lumopt2.utils.callbacks import BaseCallback

    class B4Log(BaseCallback):
        def on_function_eval(self, project, eval_num, params, fom_value,
                             gradient=None, **kw):
            with open(log_path, "a") as f:
                f.write(json.dumps({"dx": float(np.squeeze(params)),
                                    "fom": float(fom_value)}) + "\n")

    def func(p):                                  # p = [dx_nm]
        props = {}
        for j, (top, bot) in enumerate(eng.scatterer_names()):
            x = (j - eng.COMB_N_HALF) * eng.COMB_LAM_NM + p[0]
            props[f"{top}::x"] = props[f"{bot}::x"] = x * eng.NM
        return props

    region = lmpt.Box(x_span=33e-6, y_span=5e-6, z_span=0.8e-6,
                      dx=50e-9, dy=50e-9, dz=50e-9)   # explicit mesh REQUIRED
                                                      # (None crashes addmesh)
    par = lmpt.Parametrization(func=func, bounds=[(140.0, 550.0)],
                               optimization_region=region,
                               initial_params=np.array([start_dx]), dp=2.0)
    # dp as a SCALAR: a 1-element list hits a lumopt2 n_params==1 bug
    # (dp.squeeze().shape[0] IndexError — measured, job 132730)
    fom = lmpt.Fom(lmpt.PortResults("Port_2", "transmission",
                                    [w * eng.NM for w in wl_nm]),
                   fct=eng.make_fct(wl_nm))
    project = lmpt.Project(setup=fsp, parametrization=par, fom=fom,
                           runner=lmpt.LocalRunner(resource="GPU"),
                           project_name=os.path.join(out_dir, "b4_dx"))
    project.fom.config_map.project_folder = os.path.join(out_dir, "b4_dx_files")
    # (same project_folder bug as in make_project — see the comment there)
    optimizer = lmpt.ScipyOptimizer(method="L-BFGS-B", max_iter=8,
                                    bounds=[(140.0, 550.0)], max_line_search=4)
    result = lmpt.Optimization(project, optimizer,
                               callbacks=[B4Log(), lmpt.FileLogger()]).run()
    dx = float(np.squeeze(result.optimal_params))
    moved = (dx - 300.0) / (401.0 - 300.0)
    ok = moved > 0.5 and abs(dx - 401.0) <= 50.0
    print(f"B4: δx 300 → {dx:.1f} nm (target 401, recovered {100 * moved:.0f}%) "
          f"FOM {result.initial_fom:.4f} → {result.final_fom:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    # Windows consoles default to cp1252, which cannot print the λ/σ/α glyphs
    # in the gate output (UnicodeEncodeError). Cluster runs are UTF-8 already.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b0 = gate_b0()
    b1 = gate_b1()
    sys.exit(0 if (b0 and b1) else 1)
