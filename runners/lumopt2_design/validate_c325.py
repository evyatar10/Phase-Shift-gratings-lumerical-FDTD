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
N_TASKS = 46         # 0=B2a bare | 1=B2b comb | 2=B3 gradients | 3=B4 mini-opt
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
                     # ── route-1 GPU FieldRegion size ladder (2026-08-24,
                     #    PRIORITY ZERO, HANDOFF "RETRY THE GPU WIDTH-ADJOINT"):
                     #    27-32 = _GFR_RUNGS below (27 control reproduces the
                     #    CUDA error; 28-31 shrink the region; 32 thin-3D)

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
    501 pts — the adopted v2 window, a NAMED §2 change anchored by task 10).

    ★n_wl_points is overridable because lumopt2 holds the FULL
    (nx,ny,nz,3,n_wl) forward AND adjoint optimization-region arrays for
    EVERY fom entry at once (base_fom.py:473-511 — phase 1 collects all
    e_fwd, phase 2 adds each e_adj), and a MixedFom adds a third cached
    region read (the width adjoint's own file, which records the same 501
    λ even though only one is used). That extra ~25 GB is what OOM-killed
    the 160 G FD gate (job 136122, exit 137) after the same contraction had
    already blown the walltime (136108). For a wg_pure gate (J = −softW)
    the T spectrum enters the FOM NOWHERE — softW is read from the
    single-λ twin at the scan centre — so a coarser grid is the same
    physics at a fraction of the memory. Keep the count ODD so the centre
    λ stays on-grid, and keep enough points to resolve the 0.73 nm peak for
    the logged diagnostics (151 → 66 pm, ~11 points across the FWHM).
    """
    import dataclasses
    kw.setdefault("n_wl_points", 501)
    return dataclasses.replace(
        SPEC, label=SPEC.label + suffix, width_grad=True,
        scan_width_nm=10.0,
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


# ── route-1 GPU FieldRegion size ladder (tasks 27-32, 2026-08-24) ────────────
# Diagnoses the CUDA "invalid configuration argument" that kills the FieldRegion
# width-adjoint on GPU (job 136026, 3/3 tasks). That is cudaErrorInvalidConfig-
# uration = kernel block/grid dims out of range — characteristically a SIZE
# problem. The FieldRegion twin copies field_profile's spans (nearly the whole
# grating). If a smaller region launches, the production fix is tiling, not
# abandoning the object. Rung 32 keeps full spans but makes the region 3D
# (a few z cells) to test the singleton-dimension hypothesis instead.
# ★Rungs 31/32 may die in PYTHON (softW envelope on a tiny window; 3D array
# shapes in _line_from_res) BEFORE the adjoint launches — a Python traceback is
# an INCONCLUSIVE rung, only a CUDA error / a running solve is a verdict.
# ★A running solve is NOT success either: h5-gate the adjoint output for
# non-zero fields first (the 52-min all-zero fake is the precedent), then the
# FD gate vs the stored 136189 vector [-0.00365, +0.01825, +0.02026].
# Dispatch (short lane, ~1 seat/task):
#   SBATCH_MEM=160G LUMOPT2_TIME=04:00:00 bash athena/deploy_athena.sh \
#       --lumopt2-design=runners.lumopt2_design.validate_c325 --array-tasks=27-32

_GFR_RUNGS = {
    27: dict(tag="full"),                            # control — must reproduce
    28: dict(tag="yhalf", scale_y=0.5),
    29: dict(tag="xhalf", scale_x=0.5),
    30: dict(tag="quart", scale_x=0.25, scale_y=0.25),
    31: dict(tag="patch", x_span_um=6.0, y_span_um=0.8),
    32: dict(tag="thin3d", z_span_um=0.16),          # ~3 dz cells, 3D region
    33: dict(tag="small3d", scale_x=0.25, scale_y=0.5, z_span_um=0.16),
    # ── MEASURED 2026-08-24: every PASS has x=528, every FAIL has x=2112,
    #    regardless of y, z or total cells (33 passed at 22,176 cells while
    #    28 failed at 29,568). 34 is the decisive test: x stays 528 but the
    #    cell count goes to 45,936 — ABOVE every failure so far.
    #    PASS ⇒ the bound is on X ALONE ⇒ tile in x only.
    #    FAIL ⇒ it is a total-cell budget ⇒ tile in both axes.
    34: dict(tag="xnarrow_big", scale_x=0.25, scale_y=1.0, z_span_um=0.16),
    # ── MEASURED 2026-08-24 23:19: PERFECT separation on x alone —
    #    x=528 PASS (3/3, cells 3,696-45,936) | x>=1056 FAIL (4/4, cells
    #    29,568-183,744). The cell ranges OVERLAP, so cells is not the
    #    variable; x is. ★MECHANISM HYPOTHESIS: CUDA's hard limit is 1024
    #    threads per block, and the plugin carries the string "Total threads
    #    per block %u exceeds device limit of %d" — 528 <= 1024 passes,
    #    1056 > 1024 fails. 35/36 bracket it WITHOUT sitting on the exact
    #    boundary (mesh snapping could tip a 1024-cell rung either way):
    35: dict(tag="x1000", x_span_um=50.0),    # 1000 cells — safely under
    36: dict(tag="x1080", x_span_um=54.0),    # 1080 cells — safely over
}
# 35 PASS + 36 FAIL ⇒ threshold in (1000, 1080), consistent with 1024 ⇒ a
# 704-cell tile (3 tiles across the 2112-cell region) is safe with margin.
# ★32 and 33 are the HIGH-VALUE PAIR after the 2026-08-24 binary/doc finding
# (HANDOFF "ZERO-DIMENSION"): 32 = 3D at FULL size, 33 = 3D at quarter x.
# 32 launches ⇒ z span 0 was the whole bug and no tiling is needed.
# 32 dies "exceeds device limit" but 33 launches ⇒ dimension AND size both
# bite ⇒ tiling. Both die ⇒ 3D is not the cure, fall back to routes 2/3.


def _shrink_twin(tag, scale_x=1.0, scale_y=1.0, x_span_um=None, y_span_um=None,
                 z_span_um=None):
    """Wrap eng.build_base_fsp so the saved scene's field_profile_adj (monitor
    AND adjoint source — addfieldregion is both) is resized before any run.
    Scene-local: the engine and the running campaigns are untouched."""
    orig = eng.build_base_fsp

    def wrapped(spec, out_path):
        wl = orig(spec, out_path)
        sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
        import lumapi
        with lumapi.FDTD(filename=out_path, hide=True) as f:
            xs = float(np.squeeze(f.getnamed("field_profile_adj", "x span")))
            ys = float(np.squeeze(f.getnamed("field_profile_adj", "y span")))
            xs = x_span_um * 1e-6 if x_span_um else xs * scale_x
            ys = y_span_um * 1e-6 if y_span_um else ys * scale_y
            f.setnamed("field_profile_adj", "x span", xs)
            f.setnamed("field_profile_adj", "y span", ys)
            zs = 0.0
            if z_span_um:
                f.setnamed("field_profile_adj", "monitor type", "3D")
                zs = z_span_um * 1e-6
                f.setnamed("field_profile_adj", "z span", zs)
            dx = spec.region_dx_nm * 1e-9
            print(f"[gfr] rung={tag} x_span={xs * 1e6:.3f}um "
                  f"y_span={ys * 1e6:.3f}um z_span={zs * 1e6:.3f}um "
                  f"~cells {xs / dx:.0f} x {ys / dx:.0f}"
                  f"{f' x {max(zs / dx, 1):.0f}' if z_span_um else ''}",
                  flush=True)
            f.save()
        return wl

    eng.build_base_fsp = wrapped


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
        # ★DISPATCH LANE (both prior attempts died on sizing, not physics —
        # 136108 TIMEOUT at 1:55, 136122 OOM at 1:58 in a 2 h lane): central
        # differences = 2 forwards PER INDEX, so 3 indices cost
        # fwd + adj + 6 legs ~= 5-6 h at 151 points. Dispatch as
        #   SBATCH_MEM=250G LUMOPT2_QOS=12h_4g LUMOPT2_TIME=09:00:00
        # never in the 2 h lane.
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]   # 3 classes; quota-safe
        eng.run_validate_gradient(_w_spec("_w3gpufd", wg_pure=True,
                                          wg_source="import",
                                          wg_adj_resource="GPU",
                                          n_wl_points=151),   # see _w_spec
                                  out_dir, indices, perturbation=4.0)
        print("W3-GPU: FD FIRST. PASS = sign 3/3 and per-param residual <=10% "
              "after the C_field fit; then the in-loop width gradient is LIVE "
              "at GPU speed (EXACT_WIDTH_GRAD switch in campaign_v2_uniform).")
    elif task_idx == 21:
        # W3-GPU-QUAD: the Im-quadrature partner of task 20 (same detune=1
        # point, same import/GPU/151-pt config) — adj_fix_field=(0,1) makes
        # the printed vector Im{Z}. fit_c_field.py needs FD (t20) + Re (t20's
        # adjoint) + THIS. Chain behind task 20 with --after=<its job id>.
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        eng.run_adjoint_only(_w_spec("_w3gpuquad", wg_pure=True,
                                     wg_source="import",
                                     wg_adj_resource="GPU",
                                     n_wl_points=151,
                                     adj_fix_field_re=0.0,
                                     adj_fix_field_im=1.0),
                             out_dir, indices)
    elif task_idx == 22:
        # W2-DET: ONE forward at the exact detune-1 gate point, v2 window
        # (user 2026-08-23: "be sure the field-profile point is really on
        # the resonance"). The gate's softW twin sits at the SCAN CENTER
        # (engine build, wavelength center = cfg.spectral center), while the
        # detuned device's resonance is somewhere else in the window. This
        # row measures that offset: lam_pk vs center 1564.21, plus softw
        # (twin@center) vs fwhm_env (@lam_pk). Reading: offset ≲1 linewidth
        # (0.73 nm) = the gate probed near-resonant physics, C fit transfers;
        # offset ≫1 linewidth = the C fit was made on off-peak fields — flag
        # it in §22 and re-gate at a corrected center before ANY exact-mode
        # campaign. Chain: --after=<136190's job> (afterok).
        spec = _w_spec("_w2det")
        p = eng.seed_params(spec)
        p[eng.SL_SHIFT] = 20.0
        p[eng.SL_R.start + eng.COMB_N_HALF] = 100.0
        p[eng.SL_X.start + eng.COMB_N_HALF] += 50.0
        p[eng.I_DCOMB] = 1750.0
        spec.seed_override = tuple(p)
        row = eng.run_canary(spec, out_dir)
        lam = row.get("lam_pk_nm")
        # (fix 2026-08-23 review: the old f-string formatted None with +.3f
        # → TypeError on a failed resonance read, masking the actual answer)
        off = ("n/a (no resonance read)" if lam is None else
               f"{lam - 1564.21:+.3f} nm ({abs(lam - 1564.21) / 0.73:.2f} linewidths)")
        print(f"[w2det] lam_pk {lam} nm | twin/center 1564.21 | offset {off} | "
              f"T {row.get('t_pk')} | fwhm_env {row.get('fwhm_env_um')} | "
              f"softw_adj (twin@center) {row.get('softw_adj_um')} | "
              f"softw (broadband@lam_pk) {row.get('softw_um')}")
    elif task_idx == 23:
        # MX-PRERETRIM (user question 2026-08-23: "how do we manage to reduce
        # the mode width? check there is no hidden issue"). BEST_T9635 WITHOUT
        # the +52.5 retrim, at the pitch-locked mesh — the one row missing
        # from the width lineage. Everything else identical to task 16, so
        # t23 vs t16 isolates the retrim's TRUE narrowing at a mesh whose
        # width readings are phase-unbiased (§24); the historical "20.34 ->
        # 17.70 = -14.9%" was measured at dx=50, where the SAME origin/retrim
        # pair reads +0.23% but reads -2.69% corrected — i.e. the old mesh
        # could not be trusted for cross-design width claims.
        import dataclasses   # other branches import it locally => it is a
                             # FUNCTION-local name for all of main() (job
                             # 136283 died UnboundLocalError in 10 s)
        from runners.lumopt2_design.best_designs import BEST_T9635
        spec = dataclasses.replace(
            SPEC, label=SPEC.label + "_mx_preretrim", scan_width_nm=10.0,
            n_wl_points=501, region_dx_nm=eng.DX_PITCHLOCK_NM,
            scan_center_nm=1566.9)   # ~+0.5 nm redder than retrim (shallower)
        spec.seed_override = tuple(eng.replay_params(
            spec, np.asarray(BEST_T9635, dtype=float)))
        row = eng.run_canary(spec, out_dir)
        fw = row.get("fwhm_env_um")
        print(f"[mx_preretrim] T {row.get('t_pk')} λ {row.get('lam_pk_nm')} "
              f"FWHM {fw} | vs retrim 17.8530 => retrim narrowed by "
              f"{None if not fw else (1 - 17.8530 / fw) * 100:.2f}% | "
              f"vs corrected origin 18.3460 => ratio "
              f"{None if not fw else fw / 18.3460:.4f}")
    elif task_idx in (24, 25):
        # ── What is actually doing the work? (user 2026-08-23) ──────────────
        # Every measurement so far changed the inner corrugation AND created
        # the 46 nm step at the tooth-25/26 boundary at once. That step sits
        # where the mode still carries 36% of peak intensity, so it cannot be
        # dismissed. Both rows start from the SAME retrimmed best as t16
        # (mx_retrim: T 0.95941, λ 1566.377, FWHM 17.8530) and change ONE
        # thing, at the pitch-locked mesh:
        #   24 = DE-STEPPED: teeth 21-25 ramp linearly down to the frozen
        #        outer 325 nm, so the discontinuity is removed using only
        #        free-block parameters (no engine change). Mean corr drops
        #        ~1.3% ⇒ expect ~+1% width; normalise with 0.00224 T/µm
        #        before reading T. T holds ⇒ the step is NOT the mechanism
        #        (bulk inner-κ is); T drops ⇒ the step is load-bearing.
        #   25 = COMB OFF at the retrimmed best (bare=True). The comb's
        #        +0.0107 T / +14% Q_i was measured on the UNIFORM device;
        #        whether it still pays on this much deeper, shifted design
        #        has never been tested.
        import dataclasses           # function-local elsewhere in main()
        from runners.lumopt2_design.best_designs import BEST_T9635
        p = np.asarray(BEST_T9635, dtype=float).copy()
        p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + 52.5, SPEC.corr_max_nm)
        tag = "_destep" if task_idx == 24 else "_comboff"
        if task_idx == 24:
            c = p[eng.SL_CORR].copy()
            ramp = np.linspace(c[19], eng.CORR_NM, 7)[1:]   # teeth 20→25 → 325
            c[19:] = ramp[:len(c) - 19]
            p[eng.SL_CORR] = c
        spec = dataclasses.replace(
            SPEC, label=SPEC.label + tag, scan_width_nm=10.0, n_wl_points=501,
            region_dx_nm=eng.DX_PITCHLOCK_NM, scan_center_nm=1566.377,
            bare=(task_idx == 25), free_comb=False)
        spec.seed_override = tuple(eng.replay_params(spec, p))
        row = eng.run_canary(spec, out_dir)
        t, fw = row.get("t_pk"), row.get("fwhm_env_um")
        norm = (t + 0.00224 * (17.8530 - fw)) if (t and fw) else None
        print(f"[{tag[1:]}] T {t} λ {row.get('lam_pk_nm')} FWHM {fw} | "
              f"T at the retrim width 17.8530: {norm} "
              f"(retrim reference 0.95941; mean corr "
              f"{float(np.mean(p[eng.SL_CORR])):.2f})")
    elif task_idx == 26:
        # ★COMB VALUE vs MODE WIDTH (user hypothesis 2026-08-23). The comb
        # measured +0.0107 T on the ORIGIN (18.346 µm) but only +0.0030 on the
        # re-trimmed best (17.853 µm) — but those differ in TWO ways: the
        # design AND the width. The comb is a FIXED row of scatterers spanning
        # ±15 µm, so a narrower mode overlaps it less (a tail effect, more
        # width-sensitive than the 2.7% suggests). This row is comb-OFF at the
        # +42 nm re-trim = the SAME design at the BAND-CENTRE width 18.346, so
        # comb-on minus comb-off is measured at the benchmark width. Its
        # comb-ON partner comes free: campaign 136465's iteration 0 is exactly
        # this geometry with the comb in.
        #   comb value at 18.35 >> 0.0030  ⇒ the comb's worth is WIDTH-driven,
        #     and it earns its 114 posts whenever the mode sits on spec;
        #   comb value ≈ 0.0030            ⇒ the DESIGN absorbed it, and the
        #     comb is genuinely marginal now (fabrication decision).
        import dataclasses           # function-local elsewhere in main()
        from runners.lumopt2_design.best_designs import BEST_T9635
        p = np.asarray(BEST_T9635, dtype=float).copy()
        p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + 42.0, SPEC.corr_max_nm)
        spec = dataclasses.replace(
            SPEC, label=SPEC.label + "_comboff42", scan_width_nm=10.0,
            n_wl_points=501, region_dx_nm=eng.DX_PITCHLOCK_NM,
            scan_center_nm=1566.398, bare=True, free_comb=False)
        spec.seed_override = tuple(eng.replay_params(spec, p))
        row = eng.run_canary(spec, out_dir)
        print(f"[comboff42] T {row.get('t_pk')} λ {row.get('lam_pk_nm')} "
              f"FWHM {row.get('fwhm_env_um')} — subtract from campaign 136465 "
              f"iteration 0 (same design, comb ON) for the comb's value at the "
              f"benchmark width; comb value at 17.853 was +0.0030")
    elif task_idx in _GFR_RUNGS:
        # route-1 ladder (header comment above _GFR_RUNGS): FieldRegion path
        # (wg_source default), GPU lane, adjoint-only. 151 λ points = the
        # memory-safe grid proven by the 136189 lane; region mesh stays the
        # task-19/20 default (50 nm) so rung 27 reproduces the failing config.
        rung = dict(_GFR_RUNGS[task_idx])
        tag = rung.pop("tag")
        _shrink_twin(tag, **rung)
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        print(f"[gfr] rung={tag}: verdicts — 'invalid configuration argument' "
              f"= CUDA launch REJECTED at this size; a real solve time = "
              f"LAUNCHED (then h5-gate + compare signs vs FD "
              f"[-0.00365, +0.01825, +0.02026]); Python traceback before the "
              f"adjoint = rung INCONCLUSIVE", flush=True)
        eng.run_adjoint_only(_w_spec(f"_gfr_{tag}", wg_pure=True,
                                     wg_source="fieldregion",
                                     wg_adj_resource="GPU",
                                     n_wl_points=151),
                             out_dir, indices)
        print(f"[gfr] rung={tag} adjoint RAN TO COMPLETION — now the h5 gate "
              f"(non-zero fields) before believing it", flush=True)
    elif task_idx == 37:
        # ★★★THE PRODUCTION WIDTH GRADIENT — full-size region, adjoint source
        # split into 4 x-tiles (2112/4 = 528 cells exactly = the highest
        # MEASURED-PASS width), all injected in ONE adjoint run on GPU.
        # Sources superpose linearly ⇒ this is the exact full-region gradient,
        # not an approximation, at the cost of one adjoint.
        # READ IN THIS ORDER: (1) the `[wg-tiles]` line — per-tile max|src|,
        # the dead-tile detector; (2) the printed vector's SIGNS vs the
        # keep-forever FD [-0.00365 corr_1, +0.01825 shift_1, +0.02026 wcav];
        # (3) only then fit C_field (fit_c_field.py) — and fit it HERE, at the
        # production tiling, never on a cropped rung (rung 30 vs 34 gave
        # ratios 0.40 vs 1.5 from the crop alone).
        # PREREQUISITE: gpu_probe task 1 (tiling identity) must have PASSED.
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        eng.run_adjoint_only(_w_spec("_wgtiled", wg_pure=True,
                                     wg_source="fieldregion",
                                     wg_adj_resource="GPU",
                                     wg_src_tiles=4,
                                     n_wl_points=151),
                             out_dir, indices)
    elif task_idx == 45:
        # ★★★FD AT THE PRODUCTION NUMERICS — the missing reference (2026-08-25).
        # WHY: C_field is MESH- AND λ-SPECIFIC (engine note :184 "C is
        # mesh/device-specific — recalibrate for a new device"). The
        # keep-forever FD [-0.00365,+0.01825,+0.02026] was measured at
        # region_dx 50.0 and the DEFAULT scan centre (1564.21); the campaign
        # runs pitch-locked 51.683 at centre 1564.614. Those centres are
        # 0.404 nm = 0.55 LINEWIDTHS apart, so the twin samples softW at a
        # different λ ⇒ a DIFFERENT FUNCTIONAL. Fitting the production adjoint
        # against the old FD gave phi -16°→-65°, vector residual 0.001→0.013
        # and corr_1 +9.6% — the fit absorbing a physics mismatch, not
        # calibrating a constant.
        # ★NOT the cause (checked, and my first hypothesis was WRONG):
        # `wg_track_resonance` does nothing here — `_wg_lam_track` is advanced
        # ONLY by the campaign log callback (:1501), which adjoint-only and
        # validate runs never invoke, so the twin falls back to
        # `scan_center_nm` (:562-568) in BOTH fits.
        # THIS RUN gives FD_prod AND Re_prod together (validate_gradient prints
        # both); Im_prod is already measured (task 43). Then refit.
        # COST: central differences, 3 indices = 6 legs + fwd + adj ~ 8 solves
        # ~ 7-8 h ⇒ needs LUMOPT2_QOS=12h_4g LUMOPT2_TIME=12:00:00.
        import dataclasses
        from runners.lumopt2_design.campaign_v2_proj import SPEC as PSPEC
        spec = dataclasses.replace(
            PSPEC, n_wl_points=151, wg_pure=True, wg_project=False,
            free_comb=True, label=SPEC.label + "_cfit_fd")
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        eng.run_validate_gradient(spec, out_dir, indices, perturbation=4.0)
        print("[cfit FD] FD FIRST in the printout. Paste FD + this run's "
              "adjoint (Re) + task 43's Im into fit_c_field.py — ALL THREE now "
              "at the SAME numerics, which the previous fit was not.")
    elif task_idx == 44:
        # ★★BEST DESIGN, LEVER 1: THE CAVITY-WIDTH HEADROOM (handoff item ②-1).
        # `wcav` is the cavity's TRANSVERSE (y) width; fwhm_env measures energy
        # along x, so confinement improves almost FREE in the metric. MEASURED
        # (rtdec task0 vs task1): +0.0409 T for +0.0305 um of x-width ~ 1.3
        # T/um — 50-60x the see-saw's 0.021 — BUT those rows were dx=50 and
        # their dW sat under the +-3.9% sampling error, so the slope is a
        # CANDIDATE, not a result (§2).
        # BEST_T9636 sits at wcav 961.1 with the bound at 1150 ⇒ 189 nm never
        # explored. This is ONE forward: the wcav-961 control is the STORED
        # 136465 eval-12 row (T 0.96361 / W 18.35309), already at the SAME
        # pitch-locked numerics, so §6 forbids re-running it.
        # Needs NO width gradient — independent of the whole projection stack.
        import dataclasses
        from runners.lumopt2_design.best_designs import BEST_T9636
        spec = dataclasses.replace(
            SPEC, label=SPEC.label + "_wcav1100",
            scan_width_nm=10.0, n_wl_points=501,
            region_dx_nm=eng.DX_PITCHLOCK_NM, scan_center_nm=1566.444,
            free_comb=False, rho_band=False)
        p = np.asarray(BEST_T9636, dtype=float).copy()
        p[eng.I_CAV] = 1100.0
        _b = np.asarray(eng.param_bounds(spec), dtype=float)
        p = np.clip(p, _b[:, 0], _b[:, 1])
        spec.seed_override = tuple(eng.replay_params(spec, p))
        row = eng.run_canary(spec, out_dir)
        t, w = row.get("t_pk"), row.get("fwhm_env_um")
        print(f"[wcav1100] T {t} | lam {row.get('lam_pk_nm')} | FWHM {w} | "
              f"Q_i {row.get('q_i')}")
        if t and w:
            print(f"  vs STORED BEST_T9636 (wcav 961.1): T 0.96361 W 18.35309 "
                  f"-> dT {t-0.96361:+.5f} over dW {w-18.35309:+.4f} um")
            print(f"  T per um = {(t-0.96361)/(w-18.35309):+.4f}" if abs(w-18.35309) > 1e-6
                  else "  width unchanged -> the lever is FREE in the metric")
    elif task_idx in (42, 43):
        # ★★★C_field RE-FIT AT THE PRODUCTION NUMERICS (audit B3, 2026-08-25).
        # The passing fit (C = 0.4554 + 0.1336i) came from tasks 37/40, which
        # ran `_w_spec` — i.e. region_dx_nm at the 50.0 DEFAULT and the default
        # scan centre. The CAMPAIGN runs the pitch-locked mesh
        # (DX_PITCHLOCK_NM = 51.683) at centre 1564.614 with resonance
        # tracking ON. Mesh ALIGNMENT is not a detail here: job 132637 measured
        # tooth-gradient scales moving 10-30x between mesh conventions, and
        # V2_FWHM_PLAN §24 explicitly rules a 50-nm-mesh C "NOT for production".
        # 42 = Re (C=(1,0)), 43 = Im (C=(0,1)); refit with fit_c_field.py.
        # ★The FD reference stays the keep-forever [-0.00365, +0.01825,
        # +0.02026] — it is config-independent (a finite difference of the
        # SAME functional), which is what makes this re-fit meaningful.
        import dataclasses
        from runners.lumopt2_design.campaign_v2_proj import SPEC as PSPEC
        quad = (task_idx == 43)
        # ★free_comb=True is REQUIRED here (measured: 137017 both tasks died in
        # 60 s, "Parameter 103 value 100.0 is outside bounds [79.999, 80.001]").
        # run_adjoint_only's detune=1 point DETUNES THE COMB (centre post
        # r 80->100, x +50, d 1750), and that is the operating point the
        # keep-forever FD was measured at — so the fit MUST sit there. With the
        # campaign's free_comb=False the comb bounds collapse and the seed is
        # rejected. Freeing the comb changes only the BOUNDS, not the geometry
        # at this explicitly-set point, and none of the three fitted indices
        # (corr_1, shift_1, wcav) is a comb parameter.
        spec = dataclasses.replace(
            PSPEC, n_wl_points=151, wg_pure=True, wg_project=False,
            free_comb=True,
            label=SPEC.label + ("_cfit_im" if quad else "_cfit_re"),
            adj_fix_field_re=(0.0 if quad else 1.0),
            adj_fix_field_im=(1.0 if quad else 0.0))
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        eng.run_adjoint_only(spec, out_dir, indices)
        print(f"[cfit {'Im' if quad else 'Re'}] production numerics: dx "
              f"{spec.region_dx_nm} nm, centre {spec.scan_center_nm}, tiles "
              f"{spec.wg_src_tiles}, track_res {spec.wg_track_resonance} — "
              f"paste into fit_c_field.py with the SAME FD and refit")
    elif task_idx == 41:
        # ★★★DOES THE FIXED GRADIENT ACTUALLY HELP? (user question 2026-08-25)
        # A SHORT projected campaign from the SAME uniform seed as 136753, so
        # the comparison is like-for-like. Two jobs in one:
        #   (a) the END-TO-END SMOKE of run_projected through the real wrapper
        #       stack — resume, guards, logging, completion path (item 23: the
        #       code that runs once is the least-tested code you own);
        #   (b) THE MEASUREMENT. Baseline 136753 (MEASURED, 5 evals / 5.85 h):
        #       W 18.409 -> 18.456 -> 18.508 -> 18.827, i.e. width EXPANDS
        #       every iteration, 1 of 5 evals rejected out-of-band, and
        #       +0.0016 T per hour.
        #   PASS = width stays within +-0.05 um (margin/2) of the 18.613
        #   target on EVERY accepted iterate, and NO eval is rejected for
        #   width. Read <label>_proj.jsonl: phase / lam / dw_pred vs the
        #   measured W in the evals jsonl — dw_pred vs actual dW is the
        #   direct test of whether the gradient's width prediction is true.
        import dataclasses
        from runners.lumopt2_design.campaign_v2_proj import SPEC as PSPEC
        # ★LANE: 4 iterates x ~2.7 h (3 solves each) = ~11 h, so this must be
        # dispatched with LUMOPT2_QOS=12h_4g LUMOPT2_TIME=12:00:00
        # SBATCH_MEM=300G — NOT the 4 h lane the other gates use (DERIVED
        # 2026-08-25; the 4 h lane would kill it after ~1 iterate).
        # ★max_iter 4 → 3 (2026-08-25): this run's job is to VALIDATE the
        # wg_lam_chain fix, and that verdict lands in 1-2 stepped iterates
        # (it0 = seed, so 3 iterates gives TWO ΔW values to compare against the
        # uncorrected toy's +0.0110 / +0.0122 µm). The λ-chain adds two CPU
        # assembly passes per iterate (~+15 min), so 4 would crowd the 12 h lane.
        # ★LABEL CHANGED 2026-08-26: `lumopt2_v2_proj_toy` is the label the
        # UNCORRECTED control (137075_41) already wrote under, and
        # `run_campaign` cold-start-resumes via `_best_from_log`, which reads
        # `<out_dir>/<label>_evals.jsonl`. Reusing the label would silently
        # START THE CORRECTED RUN AT THE CONTROL'S BEST POINT (fom 0.669780,
        # W 18.3684) instead of the uniform seed — destroying the very
        # comparison this run exists to make, and burning ~8 GPU-h to do it.
        # (Jobs 137267/137296 carried this flaw; neither reached an iterate.)
        # A fresh label has no log ⇒ genuine cold start from the seed.
        spec = dataclasses.replace(PSPEC, label="lumopt2_v2_projchain_toy",
                                   max_iter=3, max_feval=6)
        spec.adj_fix_field_re, spec.adj_fix_field_im = (0.4554, +0.1336)
        best = eng.run_campaign(spec, out_dir)
        print(f"[proj-toy] completed: best_fom {best['fom']:.5f} — now compare "
              f"the width trajectory against 136753's 18.409->18.827")
    elif task_idx == 27:
        # ★CONTROL TWIN of task 41 (2026-08-27, user: "we might need to rerun
        # anyways"): identical spec, wg_lam_chain OFF, fresh label. Re-measures
        # the uncorrected ΔW trajectory under the CURRENT engine + committed
        # per-tooth mesh fix (3120d38), so the toy-vs-control comparison cannot
        # be confounded by code/mesh drift since 137075_41. Judge exactly like
        # 137075: ΔW per accepted iterate (old control +0.0110 / +0.0122 µm)
        # and λ_pk drift (~+0.04 nm/iterate). Same 12 h lane as task 41.
        import dataclasses
        from runners.lumopt2_design.campaign_v2_proj import SPEC as PSPEC
        spec = dataclasses.replace(PSPEC, label="lumopt2_v2_projctrl_toy",
                                   max_iter=3, max_feval=6, wg_lam_chain=False)
        spec.adj_fix_field_re, spec.adj_fix_field_im = (0.4554, +0.1336)
        best = eng.run_campaign(spec, out_dir)
        print(f"[proj-ctrl] completed: best_fom {best['fom']:.5f} — uncorrected "
              f"control under current engine; compare ΔW vs 137075_41")
    elif task_idx == 40:
        # ★THE QUADRATURE PARTNER of task 37 — same point, same TILED config,
        # C_field = (0, 1) so the printed vector is Im{Z}. fit_c_field.py needs
        # three vectors: FD (the keep-forever [-0.00365, +0.01825, +0.02026]),
        # Re (task 37's), and THIS. Only then is C_field fitted AT the
        # production tiling, which is the only place it is meaningful.
        # Dispatch AFTER task 37 passes — if 37 is rejected this is wasted.
        indices = [0, eng.SL_SHIFT.start, eng.I_CAV]
        eng.run_adjoint_only(_w_spec("_wgtiledquad", wg_pure=True,
                                     wg_source="fieldregion",
                                     wg_adj_resource="GPU",
                                     wg_src_tiles=4,
                                     n_wl_points=151,
                                     adj_fix_field_re=0.0,
                                     adj_fix_field_im=1.0),
                             out_dir, indices)
    elif task_idx in (38, 39):
        # ★★P2 / P3 — the KNOWN-ANSWER sign gates for the projected method.
        # Each is ONE gradient evaluation whose PROJECTED DIRECTION must show a
        # sign pattern the programme has already MEASURED physically. Together
        # they falsify (or confirm) the whole formulation for ~2 evals, before
        # any campaign is dispatched. Magnitudes are NOT gated here — C_field
        # only rescales, it cannot flip a sign — so these run BEFORE the fit.
        #   38 = P2 free-shift: at the UNIFORM seed, shifts below the e=65 knee
        #        are MEASURED width-free, so projection should barely touch
        #        them ⇒ the shift block must come out strongly POSITIVE.
        #        PASS: >0 on >=20 of 25 teeth.
        #   39 = P3 see-saw: at CEILING CONTACT the corrugation block must show
        #        the inner/outer SIGN SPLIT (inner one way, outer the other),
        #        matching the measured ~11x inner/outer width-price ratio.
        #        PASS: split matches on >=20 of 25 teeth after grouping.
        import dataclasses
        from runners.lumopt2_design.campaign_v2_proj import (
            SPEC as PSPEC, W_TARGET_UM)
        # ★n_wl_points 501 -> 151 (MEASURED: job 137012 OOM-killed, exit 137,
        # at "Computing gradient fields"). wg_project runs the field assembly
        # TWICE, so two full (nx,ny,nz,3,n_wl) arrays are live at once on top
        # of lumopt2's per-entry fwd+adj arrays; at 501 lambda that is ~21 GB
        # EACH and blows 160 G. At 151 the same stack is ~32 GB total.
        # Safe for THIS gate because it checks SIGNS of the projected
        # direction, and spectral sampling cannot flip a gradient's sign.
        # (The CAMPAIGN keeps 501 and must be dispatched at 300 G.)
        # ★WINDOW + SAMPLING (fixed 2026-08-25 after the user asked whether the
        # spectrum is resolved). 151 pts over 10 nm = 66 pm = only ~11 points
        # across the 0.73 nm resonance ⇒ a ragged peak. NARROW the band instead
        # of adding points: 5 nm at 251 pts = 20 pm = ~36 points across the
        # FWHM — the SAME resolution as the campaign's 501/10 nm at half the
        # memory (the double field assembly is what OOM-killed 137012).
        # WIN_FWHM_MULT = 2.5 ⇒ the softmax window is ±1.825 nm, still inside
        # ±2.5 nm, so the FOM window is NOT clipped.
        # ★AND CENTRE ON THE RIGHT RESONANCE: the softW twin records AT THE
        # SCAN CENTRE. P3 seeds from BEST_T9636 (λ 1566.444), so inheriting the
        # uniform campaign's 1564.614 would sample 2.51 LINEWIDTHS off
        # resonance — a §22 violation that would silently measure the wrong
        # width. Lowering inner-8 by 30 nm moves λ by ~-0.035 nm (DERIVED at
        # 0.0036 nm per nm of mean corr) ⇒ 1566.409.
        centre = 1564.614 if task_idx == 38 else 1566.409
        spec = dataclasses.replace(PSPEC, n_wl_points=251, scan_width_nm=5.0,
                                   scan_center_nm=centre,
                                   label=SPEC.label +
                                   ("_p2shift" if task_idx == 38 else "_p3seesaw"))
        p = eng.seed_params(spec)
        if task_idx == 39:
            # put the design AT the ceiling: spend width via the measured
            # inner-lower lever so the constraint is active (P3 only bites in
            # PHASE B — at the uniform seed the method is still climbing).
            from runners.lumopt2_design.best_designs import BEST_T9636
            p = np.asarray(BEST_T9636, dtype=float).copy()
            # inner-8 down = widen toward the ceiling. SL_CORR is a SLICE, not
            # a sequence — index it by its own start (measured: 137011 task 39
            # died in 1 s on `SL_CORR[:8]`, 'slice' object is not subscriptable)
            i0 = eng.SL_CORR.start
            p[i0:i0 + 8] -= 30.0
        # ★CLAMP to the spec's own bounds before seeding. BEST_T9636 stores the
        # comb at r = 80.1386, but this spec freezes the comb (free_comb=False
        # ⇒ bounds 80.000 ± 0.001), and lumopt2 rejects an out-of-bounds seed
        # outright (measured, 137013: "Parameter 75 value 80.138566 is outside
        # bounds [79.999, 80.001]"). The clamp moves the comb by 0.14 nm —
        # sub-nm, i.e. inside the "comb unchanged" drift already measured.
        _b = np.asarray(eng.param_bounds(spec), dtype=float)
        p = np.clip(p, _b[:, 0], _b[:, 1])
        spec.seed_override = tuple(eng.replay_params(spec, p))
        lmpt = eng.import_lumopt2()
        project, _ = eng.make_project(spec, out_dir, lmpt)
        gT = np.asarray(project.compute_gradient(p), dtype=float)
        gW = np.asarray(project.parametrization.compute_gradient_from_fields(
            project.fom.gfields_W, project.fdtd_session, p), dtype=float)
        b = np.asarray(eng.param_bounds(spec), dtype=float)
        D = ((b[:, 1] - b[:, 0]) / 2.0) ** 2
        # ★NO jsonl dependency: this task calls compute_gradient directly, so no
        # eval row exists and `_row_of_params` would silently fall back to
        # W = target — i.e. it would quietly test the RIDE phase whatever the
        # true width. Evaluate BOTH phases explicitly instead; it is pure math
        # on the same two gradients, so it costs nothing.
        tag = f"P{2 if task_idx == 38 else 3}"
        print(f"[{tag}] |gT|={np.linalg.norm(gT):.4g} |gW|={np.linalg.norm(gW):.4g}"
              f"  (target {W_TARGET_UM:.4f} um)")
        for wname, W in (("climb", W_TARGET_UM - 0.50), ("ride", W_TARGET_UM)):
            step, phase, lam = eng._proj_step(gT, gW, D, W, W_TARGET_UM,
                                              spec.wgp_margin_um, spec.wgp_step,
                                              spec.wgp_step_max_nm)
            nrm = np.linalg.norm(gW) * np.linalg.norm(step)
            orth = abs(float(gW @ step)) / nrm if nrm else float("nan")
            sh, corr = step[eng.SL_SHIFT], step[eng.SL_CORR]
            print(f"  [{wname}] W={W:.4f} phase={phase} lam={lam:.4g} "
                  f"orth={orth:.3e}")
            print(f"     shift block >0 on {int((sh > 0).sum())}/25   "
                  f"(P2 PASS >= 20)")
            print(f"     corr inner-8 mean {corr[:8].mean():+.4g} | outer-17 "
                  f"mean {corr[8:].mean():+.4g} | opposite = "
                  f"{bool(corr[:8].mean() * corr[8:].mean() < 0)}   (P3 PASS)")
            print(f"     corr signs: {np.sign(corr).astype(int).tolist()}")
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
