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
N_TASKS = 4          # cluster array: B2a bare / B2b comb / B3 gradients / B4 mini-opt

# Stored anchors (MEASURED, never re-run — CLAUDE.md §6)
ANCHOR_BARE_N100 = {"t_pk": 0.9104, "lam_nm": 1559.006, "q": 1760.0, "tol_t": 0.005,
                    "tol_lam": 0.05, "tol_q_rel": 0.05}
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
        spec = dataclasses.replace(SPEC, label=SPEC.label + "_bare", bare=True,
                                   free_comb=False)
        row = eng.run_canary(spec, out_dir)
        a = ANCHOR_BARE_N100
        ok = (abs(row["t_pk"] - a["t_pk"]) < a["tol_t"]
              and abs(row["lam_pk_nm"] - a["lam_nm"]) < a["tol_lam"]
              and abs(row["q_loaded"] / a["q"] - 1.0) < a["tol_q_rel"])
        print(f"B2a vs stored anchor: {'PASS' if ok else 'FAIL'} "
              f"(anchor T {a['t_pk']} λ {a['lam_nm']} Q {a['q']:.0f})")
    elif task_idx == 1:                            # B2b — seed-comb canary (σ0)
        row = eng.run_canary(SPEC, out_dir)
        print(f"B2b σ0 for the campaign tripwire: sigma_um = {row.get('sigma_um')}")
    elif task_idx == 2:                            # B3 — gradient validation
        i_r, i_x = eng.SL_R.start + eng.COMB_N_HALF, eng.SL_X.start + eng.COMB_N_HALF
        indices = [0, eng.N_FREE - 1, eng.SL_SHIFT.start, i_r, i_x, eng.I_DCOMB]
        eng.run_validate_gradient(SPEC, out_dir, indices)
        print("B3 gates (apply by hand from the printout): sign agreement 6/6; "
              "adjoint/FD scale α ∈ [0.8, 1.25]; vec-error ≤ 0.15")
    elif task_idx == 3:                            # B4 — known-answer δx recovery
        _b4_mini_opt(out_dir)
    else:
        raise ValueError(f"no validation task {task_idx}")


def _b4_mini_opt(out_dir):
    """One free param: global comb offset δx, seeded detuned at 300 nm.
    The optimizer must move toward the measured optimum 401 nm (270°)."""
    lmpt = eng.import_lumopt2()
    fsp = os.path.join(out_dir, "b4_base.fsp")
    os.makedirs(out_dir, exist_ok=True)
    wl_nm = eng.build_base_fsp(SPEC, fsp)

    def func(p):                                  # p = [dx_nm]
        props = {}
        for j, (top, bot) in enumerate(eng.scatterer_names()):
            x = (j - eng.COMB_N_HALF) * eng.COMB_LAM_NM + p[0]
            props[f"{top}::x"] = props[f"{bot}::x"] = x * eng.NM
        return props

    region = lmpt.Box(x_span=33e-6, y_span=5e-6, z_span=0.8e-6)
    par = lmpt.Parametrization(func=func, bounds=[(140.0, 550.0)],
                               optimization_region=region,
                               initial_params=np.array([300.0]), dp=[2.0])
    fom = lmpt.Fom(lmpt.PortResults("Port_2", "transmission",
                                    [w * eng.NM for w in wl_nm]),
                   fct=eng.make_fct(wl_nm))
    project = lmpt.Project(setup=fsp, parametrization=par, fom=fom,
                           runner=lmpt.LocalRunner(resource="GPU"),
                           project_name=os.path.join(out_dir, "b4_dx"))
    optimizer = lmpt.ScipyOptimizer(method="L-BFGS-B", max_iter=8,
                                    bounds=[(140.0, 550.0)], max_line_search=4)
    result = lmpt.Optimization(project, optimizer).run()
    dx = float(np.squeeze(result.optimal_params))
    moved = (dx - 300.0) / (401.0 - 300.0)
    ok = moved > 0.5 and abs(dx - 401.0) <= 50.0
    print(f"B4: δx 300 → {dx:.1f} nm (target 401, recovered {100 * moved:.0f}%) "
          f"FOM {result.initial_fom:.4f} → {result.final_fom:.4f}  "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    b0 = gate_b0()
    b1 = gate_b1()
    sys.exit(0 if (b0 and b1) else 1)
