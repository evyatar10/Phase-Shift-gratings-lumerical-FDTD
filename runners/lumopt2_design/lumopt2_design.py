"""lumopt2 inverse-design engine — corr-325 pi-shift grating + SiN comb.

Study dir: runners/lumopt2_design/   |   Created 2026-08-13
Plan: ~/.claude/plans/we-have-previously-talked-crispy-wand.md (approved 2026-08-13)

WHAT THIS IS
  Adjoint (gradient) optimization of the innermost 25 periods/side of the
  corr-325 device plus the 57-post SiN comb, driven by Ansys lumopt2 (ships
  inside Lumerical 2026 R1.3). One forward + one adjoint FDTD per gradient,
  regardless of parameter count.

PARAMETER VECTOR p (length 190, ALL VALUES IN NANOMETERS — one unit everywhere)
  p[  0: 25]  corr_d   tooth corrugation (w_wide - w_narrow), d=1 innermost
  p[ 25: 50]  avg_d    tooth average width ((w_wide + w_narrow)/2)
  p[ 50: 75]  shift_d  tooth shift (shortens the narrow segment of tooth d)
  p[ 75:132]  r_j      comb post radius, site j=1..57 (x-ordered, site 29 = center)
  p[132:189]  x_j      comb post x position
  p[189]      d_comb   shared transverse distance |y| of the comb row

CONVENTIONS (read before editing)
  * The same 25 (corr, avg, shift) values drive BOTH arms by tooth index —
    the builder's convention, used by every measured apodization study. The
    pi-shift layout is intrinsically half-pitch staggered between arms
    (the cavity IS the extra narrow half-period), so index-symmetric is
    geometric mirror symmetry up to that built-in stagger.
  * Shifts: the narrow segment of tooth d shrinks by shift_d on both arms;
    the cavity absorbs 2*sum(shift) so the frozen outer 75 periods NEVER
    move. (This differs from bragg_device's inner_shift_nm bookkeeping,
    which displaces the frozen right arm; production confirms of a winner
    therefore go through THIS engine's geometry, not inner_shift_nm.)
  * The comb is NOT index-mirrored in x: the 270-degree winner is a
    traveling lattice; its x-mirror is the measured-losing 90-degree phase.
  * d_comb is ONE shared knob (settled 2026-08-13): per-site distance is
    amplitude-degenerate with radius (e^-1.95*dd rule, measured), and
    per-site amplitude freedom already exists via r_j.

COST FUNCTION (settled 2026-08-10..13, memory project_inverse_design_cost_function)
  FOM  = softmax_p(T) over a +-2.5*FWHM window re-selected every evaluation
         (p=12; reads T_peak, blind to linewidth)  [adjoint through lumopt2]
       - penalty on rho = mean(corr_d)/325, asymmetric deadband +2%/-5%
         (analytic kappa-integral proxy for the mode width; exact autograd
         gradient, zero extra simulations)
  Measured sigma (2nd moment of the field_profile x-envelope) is a per-eval
  TRIPWIRE, never in the adjoint. Q appears nowhere (TCMT identity).
  Diagnostics logged every eval to <out_dir>/<label>_evals.jsonl.

VALIDATION GATES (run before any campaign; validate_c325.py drives these)
  B0 reader on stored .mat spectra   B1 local build smoke + geometry diff
  B2 canary vs stored N=100 anchor   B3 validate_gradient (6 params)
  B4 known-answer mini-opt (comb dx)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import autograd
import autograd.numpy as anp
from dataclasses import dataclass

import config
from sim_helpers import apply_monitor_overrides, find_bragg_resonance
from runners.scatterers import _common

NM = 1e-9
C0 = 299792458.0

# ── Device family (corr-325, TM h350 — CLAUDE.md §4) ─────────────────────────
PITCH_NM        = 516.83
CORR_NM         = 325.0          # frozen outer-tooth corrugation
AVG_W_NM        = 800.0          # nominal average width (W800)
N_FREE          = 25             # free innermost periods per side (user 2026-08-13)
KAPPA_PER_UM    = 0.0353         # MEASURED corr-325 (IGUM 51736); kappa ∝ corr
TWO_KL_FLOOR    = 3.5            # surrogate rule: 2*kappa*L ≥ 3.5 (hard floor 3.2)

# ── Comb seed = measured q3db winner (job 130458): Λ531/δx401(270°)/r80/d1.9 ─
COMB_LAM_NM     = 531.0
COMB_DX_NM      = 401.0
COMB_R_NM       = 80.0
COMB_D_NM       = 1900.0
COMB_N_HALF     = 28             # 57 posts
N_COMB          = 2 * COMB_N_HALF + 1

# ── Cost function (settled + signed off 2026-08-13) ──────────────────────────
P_SOFTMAX       = 12.0
WIN_FWHM_MULT   = 2.5            # FOM window = ±2.5 × measured FWHM
BETA_UP         = 18.0           # widening side of the rho deadband (the T-cheat)
BETA_DN         = 5.0            # narrowing side
RHO_UP          = 1.02           # +2 % deadband edge
RHO_DN          = 0.95           # −5 % deadband edge
DEAD_T_FLOOR    = 0.02           # dead-device guard (dead ≈ 0.0008, healthy ≥ 0.5)
RECENTER_NM     = 2.0            # |λ_peak − scan center| trip → rebuild + restart
MAX_RESTARTS    = 5              # recenter/width-trip restarts per campaign

# ── Parameter vector layout ──────────────────────────────────────────────────
SL_CORR  = slice(0, N_FREE)
SL_AVG   = slice(N_FREE, 2 * N_FREE)
SL_SHIFT = slice(2 * N_FREE, 3 * N_FREE)
SL_R     = slice(3 * N_FREE, 3 * N_FREE + N_COMB)
SL_X     = slice(3 * N_FREE + N_COMB, 3 * N_FREE + 2 * N_COMB)
I_DCOMB  = 3 * N_FREE + 2 * N_COMB
N_PARAMS = I_DCOMB + 1           # 190


@dataclass
class CampaignSpec:
    """Everything a runner may vary. Defaults = the settled campaign values."""
    label: str = "lumopt2_c325"
    n_periods_side: int = 100        # surrogate N (2*kappa*L = 3.65)
    box_y_um: float = 6.8            # gate-A0 decides 6.8 vs 8.0
    box_z_mult: float = 4.14         # z = 0.35 + mult*1.5595 → 6.8 µm (3.50 would
                                     # be z=5.8, the REJECTED span-study rung)
    scan_center_nm: float = 1559.0   # ladder-measured λ (N-independent)
    scan_width_nm: float = 6.0
    n_wl_points: int = 301           # 20 pm grid
    corr_seed_nm: tuple = (CORR_NM,) * N_FREE   # seed B overrides with a dip profile
    corr_max_nm: float = 500.0       # tightened 5 % on a width trip
    max_iter: int = 60
    max_feval: int = 100
    dp_tooth_nm: float = 1.0         # mesher-FD step (auto ≈0.1 nm is too small)
    dp_comb_nm: float = 1.0          # raise to 2-5 if gate B3 shows comb noise
    free_comb: bool = True           # False → comb frozen at the seed geometry
    bare: bool = False               # True → NO comb at all (B2 anchor canary)


def seed_params(spec):
    """Seed vector: uniform grating (or spec's corr profile) + winner comb."""
    p = np.empty(N_PARAMS)
    p[SL_CORR]  = np.asarray(spec.corr_seed_nm, dtype=float)
    p[SL_AVG]   = AVG_W_NM
    p[SL_SHIFT] = 0.0
    p[SL_R]     = COMB_R_NM
    p[SL_X]     = [k * COMB_LAM_NM + COMB_DX_NM
                   for k in range(-COMB_N_HALF, COMB_N_HALF + 1)]
    p[I_DCOMB]  = COMB_D_NM
    return p


def param_bounds(spec):
    """Box bounds (nm). Comb d upper bound derives from the y-PML clearance;
    frozen comb params get sliver bounds (inert — func never touches them)."""
    b = ([(150.0, spec.corr_max_nm)] * N_FREE +   # corr: dip and overshoot allowed
         [(775.0, 825.0)] * N_FREE +              # avg width: ±25 nm n_eff drift cap
         [(0.0, 200.0)] * N_FREE)                 # shift (repo convention, don't tighten)
    p0 = seed_params(spec)
    if spec.free_comb and not spec.bare:
        d_max = spec.box_y_um * 1000.0 / 2.0 - 240.0 - 1200.0  # y_half − r_max − clear
        b += [(70.0, 240.0)] * N_COMB             # radius: 70 ≈ dx=50 mesh floor
        b += [(x - 100.0, x + 100.0) for x in p0[SL_X]]
        b += [(1500.0, d_max)]
    else:
        b += [(v - 1e-3, v + 1e-3) for v in p0[3 * N_FREE:]]
    assert len(b) == N_PARAMS
    return b


# ═══════════════════════════════════════════════════════════════════════════
# Geometry: parameter vector → Lumerical object properties
# ═══════════════════════════════════════════════════════════════════════════

def tooth_names(n_side):
    """Builder object names for the free teeth (asserted against the .fsp).

    bragg_device numbers rects with one global seg counter: wg_left_inf_1,
    then the left arm walks d=n..1 emitting (L_narrow_d, L_wide_d), then the
    cavity, then the right arm d=1..n emitting (R_narrow_d, R_wide_d).
    """
    names = {}
    for d in range(1, N_FREE + 1):
        names[d] = (f"L_narrow_{d}_{2 + 2 * (n_side - d)}",
                    f"L_wide_{d}_{3 + 2 * (n_side - d)}",
                    f"R_narrow_{d}_{2 * n_side + 3 + 2 * (d - 1)}",
                    f"R_wide_{d}_{2 * n_side + 4 + 2 * (d - 1)}")
    return names, f"cavity_{2 * n_side + 2}"


def scatterer_names():
    """(+y, −y) object pairs, site j=1..57 (builder draws both y-mirror copies)."""
    return [(f"scatterer_{j}_1", f"scatterer_{j}_2") for j in range(1, N_COMB + 1)]


def make_func(spec):
    """Parametrization func: p (nm) → {"object::property": SI value}.

    All arithmetic is autograd.numpy so lumopt2 can differentiate the map.
    Both free regions are walked from their FIXED outer edges toward the
    cavity; the cavity absorbs 2*sum(shift) (see module docstring).
    """
    hp = PITCH_NM / 2.0
    cav_l0 = PITCH_NM / 2.0                       # builder default cavity length
    x_out = cav_l0 / 2.0 + N_FREE * PITCH_NM      # |x| of the free-region outer edge
    names, cavity = tooth_names(spec.n_periods_side)
    comb_free = spec.free_comb and not spec.bare

    def func(p):
        corr, avg, shift = p[SL_CORR], p[SL_AVG], p[SL_SHIFT]
        w_n, w_w = (avg - corr / 2.0) * NM, (avg + corr / 2.0) * NM
        props = {}
        xl = -x_out                                # left walk, d = 25 .. 1
        for i in range(N_FREE - 1, -1, -1):
            s = shift[i]
            ln, lw = names[i + 1][0], names[i + 1][1]
            props[f"{ln}::x"], props[f"{ln}::x span"] = (xl + (hp - s) / 2.0) * NM, (hp - s) * NM
            props[f"{lw}::x"], props[f"{lw}::x span"] = (xl + (hp - s) + hp / 2.0) * NM, hp * NM
            props[f"{ln}::y span"], props[f"{lw}::y span"] = w_n[i], w_w[i]
            xl = xl + 2.0 * hp - s
        xr = cav_l0 / 2.0 + anp.sum(shift)         # right walk, d = 1 .. 25
        for i in range(N_FREE):
            s = shift[i]
            rn, rw = names[i + 1][2], names[i + 1][3]
            props[f"{rn}::x"], props[f"{rn}::x span"] = (xr + (hp - s) / 2.0) * NM, (hp - s) * NM
            props[f"{rw}::x"], props[f"{rw}::x span"] = (xr + (hp - s) + hp / 2.0) * NM, hp * NM
            props[f"{rn}::y span"], props[f"{rw}::y span"] = w_n[i], w_w[i]
            xr = xr + 2.0 * hp - s
        props[f"{cavity}::x"] = 0.0
        props[f"{cavity}::x span"] = (cav_l0 + 2.0 * anp.sum(shift)) * NM
        props[f"{cavity}::y span"] = AVG_W_NM * NM  # TM base: cavity_width_option="avg"
                                                    # → GLOBAL avg width, fixed at 800
        if comb_free:
            d_c = p[I_DCOMB] * NM
            for j, (top, bot) in enumerate(scatterer_names()):
                props[f"{top}::radius"] = props[f"{bot}::radius"] = p[SL_R][j] * NM
                props[f"{top}::x"] = props[f"{bot}::x"] = p[SL_X][j] * NM
                props[f"{top}::y"], props[f"{bot}::y"] = d_c, -d_c
        return props

    return func


# ═══════════════════════════════════════════════════════════════════════════
# Cost function: soft-max reader + kappa-ratio penalty
# ═══════════════════════════════════════════════════════════════════════════

def _plain(x):
    """Peel autograd boxes → plain numpy (stop-gradient for index decisions)."""
    while hasattr(x, "_value"):
        x = x._value
    return np.asarray(x)


def measure_peak(wl_nm, T):
    """(λ_peak_nm, T_peak, fwhm_nm) from a plain spectrum.

    Resonance via the scored peak finder (NEVER argmax — CLAUDE.md §2);
    FWHM from half-max crossings by linear interpolation. fwhm_nm is None
    when a crossing leaves the recorded window (recenter condition).
    """
    wl_nm, T = np.asarray(wl_nm, dtype=float), np.asarray(T, dtype=float)
    if wl_nm[0] > wl_nm[-1]:
        wl_nm, T = wl_nm[::-1], T[::-1]
    i_pk = int(find_bragg_resonance(wl_nm, T))
    lam_pk, t_pk = wl_nm[i_pk], T[i_pk]
    half = t_pk / 2.0
    lo = hi = None
    for i in range(i_pk, 0, -1):
        if T[i - 1] <= half:
            f = (T[i] - half) / (T[i] - T[i - 1])
            lo = wl_nm[i] - f * (wl_nm[i] - wl_nm[i - 1])
            break
    for i in range(i_pk, len(T) - 1):
        if T[i + 1] <= half:
            f = (T[i] - half) / (T[i] - T[i + 1])
            hi = wl_nm[i] + f * (wl_nm[i + 1] - wl_nm[i])
            break
    fwhm = (hi - lo) if (lo is not None and hi is not None) else None
    return float(lam_pk), float(t_pk), fwhm


class RecenterNeeded(Exception):
    """Resonance drifted off the recorded grid — rebuild base + restart."""


class WidthTrip(Exception):
    """Measured sigma left the deadband while rho stayed inside — proxy failure."""


def make_fct(wl_nm):
    """FOM fct for lumopt2: T(λ) → windowed soft-max (autograd-differentiable).

    Window indices are picked on the DETACHED spectrum (stop-gradient), so
    the gradient flows only through the T values inside the window.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)

    def fct(T):
        Tp = np.abs(_plain(T))
        lam_pk, t_pk, fwhm = measure_peak(wl_nm, Tp)
        if t_pk < DEAD_T_FLOOR:
            raise RuntimeError(f"dead device: peak T {t_pk:.4g} < {DEAD_T_FLOOR}")
        if fwhm is None:
            raise RecenterNeeded(f"FWHM crossing outside recorded window (λpk {lam_pk:.3f})")
        idx = np.where(np.abs(wl_nm - lam_pk) <= WIN_FWHM_MULT * fwhm)[0]
        return anp.mean(anp.abs(T)[idx] ** P_SOFTMAX) ** (1.0 / P_SOFTMAX)

    return fct


def kappa_penalty(p):
    """Width anchor: asymmetric deadband on rho = mean(corr)/325 (autograd)."""
    rho = anp.mean(p[SL_CORR]) / CORR_NM
    return (BETA_UP * anp.maximum(0.0, rho - RHO_UP) ** 2
            + BETA_DN * anp.maximum(0.0, RHO_DN - rho) ** 2)


_kappa_penalty_grad = autograd.grad(kappa_penalty)


def two_kappa_L(p, n_side):
    """2*kappa*L over one side (the surrogate-rule quantity), kappa ∝ corr."""
    per_um = PITCH_NM * 1e-3
    frozen = 2.0 * KAPPA_PER_UM * (n_side - N_FREE) * per_um
    free = 2.0 * KAPPA_PER_UM * float(np.sum(_plain(p)[SL_CORR]) / CORR_NM) * per_um
    return frozen + free


def attach_penalty(project):
    """Subtract the analytic width penalty (and its exact gradient) from the
    project's FOM — Optimization reads both through these two methods."""
    fom0, grad0 = project.compute_fom, project.compute_gradient

    def compute_fom(params=None):
        pp = params if params is not None else project.parametrization.get_initial_params()
        return fom0(params) - float(kappa_penalty(np.asarray(pp, dtype=float)))

    def compute_gradient(params=None):
        pp = params if params is not None else project.parametrization.get_initial_params()
        return grad0(params) - _kappa_penalty_grad(np.asarray(pp, dtype=float))

    project.compute_fom, project.compute_gradient = compute_fom, compute_gradient


# ═══════════════════════════════════════════════════════════════════════════
# Base simulation (.fsp) — reuses the production builder unchanged
# ═══════════════════════════════════════════════════════════════════════════

def build_base_cfg(spec):
    """SimulationConfig for the campaign base scene (comb_q3db numerics family)."""
    cfg = _common.build_ports_base()
    cfg.y_span_override_m = spec.box_y_um * 1e-6
    cfg.span_multiplier_override = spec.box_z_mult
    cfg.geometry.corrugation_depth_m = CORR_NM * NM   # NOT cfg.grating — silent no-op
    cfg.grating.n_periods_each_side = spec.n_periods_side
    cfg.spectral.center_wavelength_m = spec.scan_center_nm * NM
    cfg.spectral.scan_width_nm = spec.scan_width_nm
    cfg.spectral.n_wl_points = spec.n_wl_points
    if spec.bare:
        cfg.scatterer.enabled = False
        cfg.scatterer.radius_m = 0.0
    else:
        p0 = seed_params(spec)
        cfg.scatterer.enabled = True
        cfg.scatterer.radius_m = COMB_R_NM * NM
        cfg.scatterer.x_list_m = [x * NM for x in p0[SL_X]]
        cfg.scatterer.y_list_m = [p0[I_DCOMB] * NM] * N_COMB
        cfg.scatterer.height_m = 350.0 * NM
    assert cfg.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"
    return cfg


def build_base_fsp(spec, out_path):
    """Build the seed scene with the production builder and save it (no run).

    Returns the port wavelength grid in nm (uniform in FREQUENCY, matching
    the monitor sampling, so PortResults keys land exactly on the samples).
    """
    from bragg_device import PiShiftBraggFDTD
    cfg = build_base_cfg(spec)
    sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())
    try:
        sim.build()
        sim.update_scan(center_lambda_m=cfg.spectral.center_wavelength_m,
                        width_nm=cfg.spectral.scan_width_nm,
                        n_points=cfg.spectral.n_wl_points)
        apply_monitor_overrides(sim, cfg)
        _assert_name_map(sim.fdtd, spec)
        freqs = np.linspace(C0 / sim.lam_max, C0 / sim.lam_min, sim.n_wl_points)
        sim.fdtd.save(out_path)
    finally:
        sim.close()
    return (C0 / freqs) / NM      # nm, ascending in frequency (descending in λ)


def _assert_name_map(fdtd, spec):
    """Every object the func will touch must exist in the scene — never trust
    the name formula (the job-130145 class of bug)."""
    names, cavity = tooth_names(spec.n_periods_side)
    wanted = [n for quad in names.values() for n in quad] + [cavity]
    if not spec.bare:
        wanted += [n for pair in scatterer_names() for n in pair]
    missing = [n for n in wanted if not fdtd.getnamednumber(n)]
    if missing:
        raise RuntimeError(f"object-name map mismatch, missing {len(missing)}: "
                           f"{missing[:8]} ...")


# ═══════════════════════════════════════════════════════════════════════════
# lumopt2 wiring
# ═══════════════════════════════════════════════════════════════════════════

def import_lumopt2():
    """Import lumopt2 from the same install tree as lumapi (works on all 3 envs).

    Also applies the SlurmRunner fix unconditionally (user 2026-08-13, "fix it
    for always"): the shipped R1.3 SlurmRunner imports lumopt2.utils.lumslurm,
    which does not exist — the real module is api/python/lumslurm.py one level
    up. Registering it in sys.modules makes SlurmRunner constructible without
    touching the container image.
    """
    api_dir = os.path.dirname(config.LUMAPI_PATH)
    if api_dir not in sys.path:
        sys.path.insert(0, api_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")     # never open windows (§5 silent rule)
    import lumopt2
    try:
        import lumslurm
        sys.modules["lumopt2.utils.lumslurm"] = lumslurm
    except ImportError:
        pass                     # no lumslurm on this install → LocalRunner only
    return lumopt2


def make_project(spec, out_dir, lmpt=None):
    """Base .fsp + Parametrization + soft-max PortFom + GPU runner → Project.

    Returns (project, wl_nm) with the width penalty already attached.
    """
    lmpt = lmpt or import_lumopt2()
    os.makedirs(out_dir, exist_ok=True)
    fsp = os.path.join(out_dir, f"{spec.label}_base.fsp")
    wl_nm = build_base_fsp(spec, fsp)

    dp = np.full(N_PARAMS, spec.dp_tooth_nm)       # dp lives in PARAM units (nm)
    dp[SL_R] = dp[SL_X] = dp[I_DCOMB] = spec.dp_comb_nm
    region = lmpt.Box(
        x_span=2.0 * (COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + 240.0 + 500.0) * NM,
        y_span=2.0 * (spec.box_y_um * 500.0 - 900.0) * NM,
        z_span=0.8e-6,
    )
    parametrization = lmpt.Parametrization(
        func=make_func(spec), bounds=param_bounds(spec),
        optimization_region=region, initial_params=seed_params(spec),
        dp=list(dp),
    )
    fom = lmpt.Fom(
        lmpt.PortResults("Port_2", "transmission", [w * NM for w in wl_nm]),
        fct=make_fct(wl_nm),
    )
    project = lmpt.Project(setup=fsp, parametrization=parametrization,
                           fom=fom, runner=lmpt.LocalRunner(resource="GPU"),
                           project_name=os.path.join(out_dir, spec.label))
    attach_penalty(project)
    return project, wl_nm


# ═══════════════════════════════════════════════════════════════════════════
# Per-evaluation diagnostics + guards (one callback, one jsonl)
# ═══════════════════════════════════════════════════════════════════════════

def make_log_callback(spec, out_dir, sigma0_um=None, lmpt=None):
    """Log (T, λ, Q_L, Q_i, R, 1−T−R, σ, ρ, 2κL, params) each eval; enforce
    the recenter, 2κL, and σ-deadband guards (raise RecenterNeeded/WidthTrip)."""
    lmpt = lmpt or import_lumopt2()
    from lumopt2.utils.callbacks import BaseCallback
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")

    class CampaignLog(BaseCallback):
        def on_function_eval(self, project, eval_num, params, fom_value,
                             gradient=None, **kw):
            p = np.asarray(params, dtype=float)
            row = {"eval": int(eval_num), "t": time.time(), "fom": float(fom_value),
                   "rho": float(np.mean(p[SL_CORR]) / CORR_NM),
                   "two_kL": two_kappa_L(p, spec.n_periods_side),
                   "params": p.tolist()}
            try:
                project.load_forward_results()
                fdtd = project.fdtd_session.fdtd
                t2 = fdtd.getresult("FDTD::ports::Port_2", "expansion for port monitor")
                s11 = np.abs(np.squeeze(
                    fdtd.getresult("FDTD::ports::Port_1", "expansion for port monitor")["S"])) ** 2
                T = np.abs(np.squeeze(t2["T"]))
                wl = np.squeeze(t2["lambda"]) / NM
                lam_pk, t_pk, fwhm = measure_peak(wl, T)
                i_pk = int(np.argmin(np.abs(wl - lam_pk)))
                q_l = lam_pk / fwhm if fwhm else None
                row.update(lam_pk_nm=lam_pk, t_pk=t_pk, fwhm_nm=fwhm, q_loaded=q_l,
                           q_i=(q_l / (1.0 - np.sqrt(t_pk)) if q_l else None),
                           r_pk=float(s11[i_pk]),
                           loss=float(1.0 - t_pk - s11[i_pk]),
                           sigma_um=sigma_from_profile(fdtd, lam_pk))
            except Exception as e:                # diagnostics must never kill an eval
                row["diag_error"] = repr(e)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if row["two_kL"] < TWO_KL_FLOOR:
                raise RuntimeError(f"2kL {row['two_kL']:.3f} < floor {TWO_KL_FLOOR}")
            if row.get("lam_pk_nm") and abs(row["lam_pk_nm"] - spec.scan_center_nm) > RECENTER_NM:
                raise RecenterNeeded(f"peak {row['lam_pk_nm']:.3f} vs center {spec.scan_center_nm}")
            s = row.get("sigma_um")
            if sigma0_um and s and not (RHO_DN <= s / sigma0_um <= RHO_UP):
                raise WidthTrip(f"sigma {s:.2f} µm vs ctrl {sigma0_um:.2f} "
                                f"(ratio {s / sigma0_um:.3f})")

    return CampaignLog()


def sigma_from_profile(fdtd, lam_pk_nm):
    """Mode half-width (µm): sqrt of the centered 2nd moment of the |E|²
    x-envelope from the builder's field_profile monitor, at the λ nearest
    the resonance. Compared only as a RATIO to the same-N control."""
    try:
        res = fdtd.getresult("field_profile", "E")
        wl = np.squeeze(res["lambda"]) / NM
        i = int(np.argmin(np.abs(wl - lam_pk_nm)))
        I = np.sum(np.abs(res["E"]) ** 2, axis=-1)   # sum the 3 field components
        I = np.squeeze(I)
        if I.ndim > 1:                                # (x, ..., λ) → x-line at λ_i
            I = I.reshape(I.shape[0], -1)[:, i]
        x = np.squeeze(res["x"]) * 1e6
        w = np.trapz(I, x)
        mu = np.trapz(x * I, x) / w
        return float(np.sqrt(np.trapz((x - mu) ** 2 * I, x) / w))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Entries: campaign (with recenter/restart loop) and server-side validation
# ═══════════════════════════════════════════════════════════════════════════

def run_campaign(spec, out_dir, sigma0_um=None):
    """The main loop. Restarts from the best params on a recenter or width
    trip (rebuilding the base scene at the new λ) — also the crash-recovery
    path, since lumopt2 has no checkpointing. Returns the best (fom, params)."""
    lmpt = import_lumopt2()
    best = {"fom": -np.inf, "params": seed_params(spec)}
    for attempt in range(1 + MAX_RESTARTS):
        project, _ = make_project(spec, out_dir, lmpt)
        cb = make_log_callback(spec, out_dir, sigma0_um, lmpt)
        optimizer = lmpt.ScipyOptimizer(method="L-BFGS-B", max_iter=spec.max_iter,
                                        max_feval=spec.max_feval, ftol=1e-8,
                                        gtol=1e-6, max_line_search=4,
                                        bounds=param_bounds(spec))
        opt = lmpt.Optimization(project, optimizer, callbacks=[cb],
                                store_all_simulations=True)
        try:
            result = opt.run(initial_params=np.asarray(best["params"], dtype=float))
            if result.final_fom > best["fom"]:
                best = {"fom": float(result.final_fom),
                        "params": np.asarray(result.optimal_params)}
            break                                   # converged / budget exhausted
        except RecenterNeeded as e:
            best["params"], new_center = _best_from_log(spec, out_dir, best)
            print(f"[recenter {attempt}] {e} -> new center {new_center:.2f} nm")
            spec.scan_center_nm = round(new_center, 2)
        except WidthTrip as e:
            best["params"], _ = _best_from_log(spec, out_dir, best)
            spec.corr_max_nm *= 0.95
            best["params"][SL_CORR] = np.minimum(best["params"][SL_CORR], spec.corr_max_nm)
            print(f"[width trip {attempt}] {e} -> corr cap {spec.corr_max_nm:.0f} nm")
    out = {"label": spec.label, "best_fom": best["fom"],
           "best_params_nm": np.asarray(best["params"]).tolist(),
           "scan_center_nm": spec.scan_center_nm}
    with open(os.path.join(out_dir, f"{spec.label}_best.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] best FOM {best['fom']:.5f} -> {spec.label}_best.json")
    return best


def _best_from_log(spec, out_dir, fallback):
    """Best-so-far params + latest peak λ from the eval log (restart source)."""
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    best_fom = fallback["fom"]
    best_p = np.asarray(fallback["params"], dtype=float)
    lam = spec.scan_center_nm
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get("lam_pk_nm"):
                    lam = row["lam_pk_nm"]
                if row["fom"] > best_fom:
                    best_fom, best_p = row["fom"], np.asarray(row["params"])
    return best_p, lam


def run_canary(spec, out_dir):
    """Gate B2: ONE forward through the full lumopt2 stack at the seed.

    Run it with spec.bare=True to compare against the STORED bare N=100
    anchor (IGUM 51736: T 0.9104 / λ 1559.01 / Q 1760 / mode 19.24 µm —
    never re-run the control itself), and once with the comb to calibrate
    σ0 for the campaign tripwire. Returns the logged diagnostics row.
    """
    lmpt = import_lumopt2()
    project, _ = make_project(spec, out_dir, lmpt)
    cb = make_log_callback(spec, out_dir, lmpt=lmpt)
    p0 = seed_params(spec)
    fom = project.compute_fom(p0)
    try:
        cb.on_function_eval(project, 0, p0, fom)
    except (RecenterNeeded, WidthTrip):
        pass                                       # canary only reports, gates decide
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    with open(path) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[canary {spec.label}] FOM {fom:.5f}  T {row.get('t_pk')}  "
          f"λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"σ {row.get('sigma_um')} µm")
    return row


def run_validate_gradient(spec, out_dir, indices, perturbation=1e-2):
    """Gate B3: lumopt2's built-in adjoint-vs-FD check on chosen params."""
    lmpt = import_lumopt2()
    project, _ = make_project(spec, out_dir, lmpt)
    res = lmpt.validate_gradient(project, seed_params(spec), indices, perturbation)
    print(f"[validate_gradient] {res}")
    return res
