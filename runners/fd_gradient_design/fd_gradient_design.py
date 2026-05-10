"""
Finite-difference gradient inverse design for the pi-shift Bragg grating.

Drives `scipy.optimize.minimize(method='L-BFGS-B')` with central-difference
gradients on peak T(λ). Reuses the proven PSO FOM evaluator from
`runners.gradient_free_design.gradient_free_design._evaluate_particle`, so
each FOM call is one FDTD via lumapi on the parametric `freed_group`.

Why this exists: the lumopt-adjoint runner (`runners.inverse_design`) produces
gradients 10-100x off from finite-difference at every starting point tested
(check_gradient vec_error 12-44, healthy is <0.1). Until that pipeline is
patched, this module is the working gradient-based path. At N=5 parameters
the cost ratio is roughly 11 FDTDs/iter (1 base + 2*5 perturbed) vs. lumopt's
~7 FDTDs/iter, so the speed penalty is small while correctness is guaranteed.

Layout of the parameter vector p (length 2*N+1, same as PSO/lumopt):
    p = [dw_1, ..., dw_N,           # full corrugation depth (W_wide-W_narrow), nm
         shift_1, ..., shift_N,      # gap shift, nm
         cavity_width_nm]
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import config as _cfg
from simulation_config import SimulationConfig

from runners.inverse_design.inverse_design import (
    measure_baseline,
    params_to_kwargs,
    regular_grating_start,
)
from runners.gradient_free_design.gradient_free_design import (
    _build_parametric_fsp,
    _set_freed_group_params,
)


def _smoothed_peak_T(fdtd, params_nm, k_around: int = 1) -> float:
    """FOM evaluator: T at the CAVITY RESONANCE index (via find_bragg_resonance),
    averaged over ±k_around samples for FDTD-noise smoothing.

    Uses the same peak-finding logic as the post-processing analysis pipeline
    (`sim_helpers.find_bragg_resonance`): score every local max by
    `sharpness × dip_depth`, pick the highest-scoring peak. The cavity
    resonance wins because it is simultaneously the sharpest feature AND
    sits inside the deepest bandgap dip.

    Critical difference from `max(T)` or `mean(top-k T)`:
      - `max(T)`: can pick a bandgap edge artifact or a Fabry-Perot ripple,
        not the cavity resonance. WRONG FOM for resonance optimization.
      - `mean(top-k T)`: smooth but still corrupted by non-cavity peaks.
      - This function: locks onto the cavity peak consistently across
        iterations even if its absolute T is not the global max — same
        physical feature the user cares about, same feature the analysis
        pipeline reports as `resonance_transmission`.

    Smoothing: average T across ±k_around samples around the resonance
    index. With fom_n_points=201 over 10 nm, k_around=1 gives a 3-sample
    window of ~0.15 nm.

    NO retry on failure: production run 79298 showed that retrying after a
    transient lumapi error CORRUPTS the freed_group structure (subsequent
    setnamed calls fail with "no items matching the name 'freed_group'
    can be found"), making every subsequent eval return NaN. Better to
    return NaN once and let Powell treat that point as worse than any
    finite value (it'll explore another direction).
    """
    from sim_helpers import find_bragg_resonance

    if True:
        try:
            _set_freed_group_params(fdtd, params_nm)
            fdtd.run()
            # Use getresult(port, "T") which on a Lumerical port monitor
            # returns the MODAL transmission |S21|² for the source mode
            # (NOT integrated flux). Production code uses
            # getresult(port, "expansion for port monitor")["S"] then
            # |S|², but on our parametric-freed_group .fsp that result
            # wasn't available on the first call after fdtd.run() (job
            # 79298 failed every eval with "Can not find result
            # 'expansion for port monitor'"). The `T` result is the same
            # modal |S21|² but more robustly available immediately.
            res = fdtd.getresult("FDTD::ports::Port_2", "T")
            T = np.asarray(np.squeeze(res["T"])).flatten()
            if T.size == 0 or not np.all(np.isfinite(T)):
                raise RuntimeError(f"empty or non-finite T result (size={T.size})")
            T_abs = np.abs(T)
            # Get the wavelength array — port `T` result carries it under
            # "lambda"; fall back to reconstructing from the global monitor.
            try:
                wl = np.asarray(np.squeeze(res.get("lambda"))).flatten()
                if wl.size != T_abs.size:
                    raise ValueError("wl/T size mismatch")
            except Exception:
                wl_min = float(fdtd.getnamed("FDTD", "global monitor minimum wavelength"))
                wl_max = float(fdtd.getnamed("FDTD", "global monitor maximum wavelength"))
                wl = np.linspace(wl_min, wl_max, T_abs.size)
            # Find the cavity-resonance index via the production scorer.
            idx = int(find_bragg_resonance(wl, T_abs))
            # ±k_around smoothing window.
            i_lo = max(0, idx - k_around)
            i_hi = min(T_abs.size, idx + k_around + 1)
            peak = float(np.mean(T_abs[i_lo:i_hi]))
            fdtd.switchtolayout()
            return peak
        except Exception as exc:
            print(f"[smoothed_peak_T] FOM failed: {exc} — returning NaN (no retry; "
                  f"job 79298 showed retries corrupt the freed_group state).")
            try:
                fdtd.switchtolayout()
            except Exception:
                pass
            return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spec dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FDGradientSpec:
    """Configuration for one finite-difference L-BFGS-B inverse-design study."""

    # ── Free parameters and bounds ──────────────────────────────────────────
    n_free_inner_teeth: int = 2
    dw_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(60.0, 400.0), (60.0, 400.0)]
    )
    shift_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 200.0), (0.0, 200.0)]
    )
    cavity_width_bounds_nm: Tuple[float, float] = (500.0, 1100.0)

    # ── Initial point. None → regular_grating_start() (uniform pi-shift). ──
    initial_points: Optional[List[List[float]]] = None
    n_starts: int = 1
    cavity_width_start_nm: float = 800.0

    # ── Optimizer options ────────────────────────────────────────────────────
    # method='Powell' (default): gradient-free, robust to FDTD/mesh noise.
    #   1D direction searches; doesn't need a jac. Uses ~50-200 FDTDs total
    #   for N=5 to converge.
    # method='L-BFGS-B': uses central-diff jac (2N+1 FDTDs/iter). FAILS on
    #   our problem — Wolfe-condition line search is too strict for the
    #   FDTD-rebuild noise (~1e-3 in T even after top-3 smoothing).
    #   Job 79124, 79131, 79132 all stalled at p0 with
    #   ABNORMAL_TERMINATION_IN_LNSRCH despite parameter scaling and the
    #   smoothed FOM — concluded L-BFGS-B is the wrong tool here.
    method: str = "Powell"
    max_iter: int = 12
    ftol: float = 1e-5
    gtol: float = 1e-5
    # Powell-specific: initial step size in scaled [0,1] space. ~0.1 means
    # the first directional search probes ±10% of each parameter range.
    powell_xtol: float = 1e-3      # parameter convergence tolerance
    powell_initial_simplex_rel: float = 0.1   # ±10% of range

    # ── Finite-difference probe ─────────────────────────────────────────────
    # Central-difference half-step. 50 nm matches the device-wide mesh so each
    # probe moves at least one Yee cell boundary. Keep equal across all params
    # — scipy expects a single jac function over the whole vector.
    fd_step_nm: float = 50.0

    # ── FOM band (FDTD wavelength scan over the bandgap) ────────────────────
    fom_window_nm: float = 10.0
    fom_n_points: int = 201

    # ── Mesh override on the freed region (0 = device-wide mesh only) ───────
    mesh_override_dxyz_nm: float = 0.0

    # ── Geometry constraints ────────────────────────────────────────────────
    enforce_mirror_symmetry: bool = True
    lengthen_cavity: bool = True

    # ── Study metadata ──────────────────────────────────────────────────────
    label: str = ""

    # ─────────────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return 2 * self.n_free_inner_teeth + 1

    def all_bounds_nm(self) -> List[Tuple[float, float]]:
        return list(self.dw_bounds_nm) + list(self.shift_bounds_nm) + [self.cavity_width_bounds_nm]

    def validate(self) -> None:
        n = self.n_free_inner_teeth
        if n != 2:
            raise NotImplementedError(
                "FDGradientSpec currently only supports n_free_inner_teeth=2 "
                "(matches gradient_free_design's freed_group .lsf script)."
            )
        if len(self.dw_bounds_nm) != n:
            raise ValueError(
                f"dw_bounds_nm length ({len(self.dw_bounds_nm)}) must equal n_free_inner_teeth ({n})."
            )
        if len(self.shift_bounds_nm) != n:
            raise ValueError(
                f"shift_bounds_nm length ({len(self.shift_bounds_nm)}) must equal n_free_inner_teeth ({n})."
            )

    def get_starts(self, cfg: Optional[SimulationConfig] = None) -> List[List[float]]:
        """Return the list of starting parameter vectors.

        With cfg supplied: uses regular_grating_start which reads
        cfg.geometry.corrugation_depth_m (canonical full-DW value).
        Without cfg (e.g. build_sweep_list.py at deploy time): falls back
        to the hardcoded uniform pi-shift Bragg grating [300,300,0,0,W],
        matching GradientFreeDesignSpec.get_starts() convention.
        """
        self.validate()
        if self.initial_points is not None:
            return [list(p) for p in self.initial_points]
        if cfg is not None:
            return [regular_grating_start(cfg, self.n_free_inner_teeth, self.cavity_width_start_nm)]
        # cfg-less fallback. 300 nm matches the project default
        # (cfg.geometry.corrugation_depth_m = 300e-9).
        return [[300.0] * self.n_free_inner_teeth
                + [0.0] * self.n_free_inner_teeth
                + [float(self.cavity_width_start_nm)]]

    def describe(self) -> str:
        return (f"FDGradientSpec(label={self.label!r})\n"
                f"  optimizer        = {self.method}\n"
                f"  max_iter         = {self.max_iter}\n"
                f"  fd_step_nm       = {self.fd_step_nm} (used only if method='L-BFGS-B')\n"
                f"  n_free_inner     = {self.n_free_inner_teeth}\n"
                f"  dw_bounds_nm     = {self.dw_bounds_nm}\n"
                f"  shift_bounds_nm  = {self.shift_bounds_nm}\n"
                f"  cavity_bounds_nm = {self.cavity_width_bounds_nm}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cost-function and gradient closures
# ─────────────────────────────────────────────────────────────────────────────


class _FDBudgetExceeded(Exception):
    """Raised inside the FOM closure when the FDTD-eval budget is exhausted.
    Caught at the driver level so we still write the partial summary."""


def _round_key(p: np.ndarray, precision_nm: float = 0.5) -> Tuple[float, ...]:
    """Cache key for parameter vectors — round to nearest 0.5 nm (well below
    fd_step_nm=50). Production run 79124 showed scipy passes p with sub-nm
    floating-point drift between fun/jac calls at the same logical point —
    sub-nm rounding (0.001) was too tight and missed cache. 0.5 nm is well
    below dx=50 nm and well above any drift scipy introduces."""
    return tuple(round(float(v) / precision_nm) * precision_nm for v in p)


def _make_objective_and_grad(
    fdtd,
    spec: FDGradientSpec,
    bounds_nm: np.ndarray,
    history: List[dict],
    incremental_path: Optional[str],
    eval_budget: int,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Build cached `(fun, jac)` closures for scipy.optimize.minimize.

    `fun` returns -peak_T (scipy minimizes; user wants max peak_T).
    `jac` returns -grad(peak_T) via central differences with bound guards.
    Cache hits avoid duplicate FDTDs when scipy calls fun/jac at the same p.
    """
    cache: Dict[Tuple[float, ...], float] = {}
    eval_count = [0]   # mutable counter through closure
    lo = bounds_nm[:, 0]
    hi = bounds_nm[:, 1]
    dx = float(spec.fd_step_nm)

    def _save_incremental() -> None:
        if incremental_path:
            with open(incremental_path, "w") as fp:
                json.dump({"history": history,
                           "n_evals": eval_count[0],
                           "eval_budget": eval_budget}, fp, indent=2)

    def _eval_peak_T(p: np.ndarray, kind: str) -> float:
        key = _round_key(p)
        if key in cache:
            return cache[key]
        if eval_count[0] >= eval_budget:
            raise _FDBudgetExceeded(
                f"FDTD eval budget {eval_budget} exhausted; bailing out cleanly."
            )
        eval_count[0] += 1
        peak_T = _smoothed_peak_T(fdtd, p, k_around=1)
        cache[key] = peak_T
        history.append({
            "eval": eval_count[0],
            "kind": kind,
            "p_nm": list(map(float, p)),
            "peak_T": peak_T,
        })
        print(f"[fdgrad] eval {eval_count[0]:3d} ({kind:>5s}): peak_T = {peak_T:.6f}  "
              f"p = {[f'{v:.2f}' for v in p]}")
        _save_incremental()
        return peak_T

    def fun(p: np.ndarray) -> float:
        return -_eval_peak_T(p, "fun")

    def jac(p: np.ndarray) -> np.ndarray:
        # Make sure base point is cached (scipy almost always calls fun(p)
        # before jac(p), but be robust to any order).
        f0 = _eval_peak_T(p, "fun")
        n = len(p)
        grad_T = np.zeros(n, dtype=float)
        for i in range(n):
            # Central difference with bound guards. If p+dx/2 crosses hi,
            # use one-sided (p, p-dx); symmetric for lo.
            hi_i = hi[i]
            lo_i = lo[i]
            p_plus = p.copy()
            p_minus = p.copy()
            half = 0.5 * dx
            if p[i] + half > hi_i and p[i] - dx >= lo_i:
                # One-sided: backward difference (p, p-dx)
                p_minus[i] = p[i] - dx
                f_minus = _eval_peak_T(p_minus, "jac")
                grad_T[i] = (f0 - f_minus) / dx
            elif p[i] - half < lo_i and p[i] + dx <= hi_i:
                # One-sided: forward difference (p, p+dx)
                p_plus[i] = p[i] + dx
                f_plus = _eval_peak_T(p_plus, "jac")
                grad_T[i] = (f_plus - f0) / dx
            else:
                # Standard central difference
                p_plus[i] = min(p[i] + half, hi_i)
                p_minus[i] = max(p[i] - half, lo_i)
                f_plus = _eval_peak_T(p_plus, "jac")
                f_minus = _eval_peak_T(p_minus, "jac")
                grad_T[i] = (f_plus - f_minus) / (p_plus[i] - p_minus[i])
        print(f"[fdgrad] grad(peak_T) = {[f'{g:+.3e}' for g in grad_T]}")
        return -grad_T   # minimize -peak_T → grad of -peak_T

    return fun, jac


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main driver
# ─────────────────────────────────────────────────────────────────────────────


def run_fd_gradient_design(
    cfg: SimulationConfig,
    spec: FDGradientSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run one finite-difference L-BFGS-B optimization."""
    import sys as _sys
    from scipy.optimize import minimize as _minimize  # type: ignore

    _lum_dir = os.path.dirname(_cfg.LUMAPI_PATH)
    if _lum_dir and _lum_dir not in _sys.path:
        _sys.path.insert(0, _lum_dir)
    import lumapi  # type: ignore
    from runners.single.run_simulation import run_single_sim

    spec.validate()
    starts = spec.get_starts(cfg)
    if not (0 <= start_idx < len(starts)):
        raise IndexError(f"start_idx {start_idx} out of range (have {len(starts)} starts).")
    p0 = list(starts[start_idx])
    print(f"[fd_gradient_design] start[{start_idx}] = {p0}")

    # Push p0 into cfg (so baseline measurement uses the right starting geom).
    cfg = copy.deepcopy(cfg)
    cfg.grating.n_free_inner_teeth = spec.n_free_inner_teeth
    cfg.grating.lengthen_cavity = spec.lengthen_cavity
    init_kwargs = params_to_kwargs(p0, spec.n_free_inner_teeth)
    cfg.grating.inner_dw_nm = list(init_kwargs["inner_dw_nm"])
    cfg.grating.inner_shift_nm = list(init_kwargs["inner_shift_nm"])
    cfg.grating.cavity_width_m = init_kwargs["cavity_width_m"]

    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "fd_gradient_design",
                                   spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    initial_peak_T: Optional[float] = None
    if baseline_lambda_m is None:
        # Retry the baseline measurement on transient lumapi races (license
        # contention or "Can not find result 'expansion for port monitor'"
        # when multiple jobs hit the FlexLM at startup; observed in 79259,
        # crashed at 22 s before any optimization). Up to 3 attempts with
        # 30 s back-off.
        import time as _time
        baseline_err = None
        for attempt in range(3):
            try:
                baseline_lambda_m, initial_peak_T = measure_baseline(cfg)
                baseline_err = None
                break
            except Exception as exc:
                baseline_err = exc
                print(f"[fd_gradient_design] baseline attempt {attempt + 1}/3 failed: {exc}")
                if attempt < 2:
                    print(f"[fd_gradient_design] sleeping 30s before retry ...")
                    _time.sleep(30)
        if baseline_err is not None:
            raise RuntimeError(
                f"baseline measurement failed after 3 attempts: {baseline_err}"
            )

    # Build the parametric .fsp once. _build_parametric_fsp reads
    # n_free_inner_teeth, fom_window_nm, fom_n_points, mesh_override_dxyz_nm
    # off the spec — supply a SimpleNamespace shim so we don't subclass
    # GradientFreeDesignSpec.
    psf_shim = types.SimpleNamespace(
        n_free_inner_teeth=spec.n_free_inner_teeth,
        fom_window_nm=spec.fom_window_nm,
        fom_n_points=spec.fom_n_points,
        mesh_override_dxyz_nm=spec.mesh_override_dxyz_nm,
    )

    print("[fd_gradient_design] building parametric .fsp ...")
    fdtd = lumapi.FDTD(hide=True)
    fsp_path = os.path.join(out_dir, "fdgrad_design.fsp")
    history: List[dict] = []
    incremental_path = os.path.join(out_dir, "fdgrad_incremental.json")
    p_final: List[float] = list(p0)
    fom_final = float("nan")
    convergence_msg = "not_run"
    try:
        _build_parametric_fsp(fdtd, cfg, psf_shim, p0, baseline_lambda_m)
        fdtd.save(fsp_path)
        print(f"[fd_gradient_design] saved .fsp: {fsp_path}")

        bounds_nm = np.asarray(spec.all_bounds_nm(), dtype=float)
        n = spec.n_params()
        # Hard cap on FDTD evaluations: max_iter * (1 base + 2N perturbed) plus
        # a small slack for line-search reruns. Prevents L-BFGS-B from spiraling.
        eval_budget = int(spec.max_iter * (1 + 2 * n) + 5)

        fun, jac = _make_objective_and_grad(
            fdtd, spec, bounds_nm, history, incremental_path, eval_budget
        )

        # Parameter rescaling to [0, 1] internally (see note below).
        # Production run 79130 confirmed L-BFGS-B's first-step heuristic
        # (`step = -g`, raw, in parameter units) gives sub-mesh-cell
        # displacements when |g| ≈ 1e-3 in a parameter range of 100s of nm.
        # Line search then never leaves p0 → ABNORMAL_TERMINATION_IN_LNSRCH.
        # Rescaling to [0,1] makes |g| ~ 1, so L-BFGS-B's defaults work.
        lo_arr = bounds_nm[:, 0]
        hi_arr = bounds_nm[:, 1]
        rng_arr = hi_arr - lo_arr

        def _to_scaled(p_nm: np.ndarray) -> np.ndarray:
            return (np.asarray(p_nm, dtype=float) - lo_arr) / rng_arr

        def _from_scaled(p_s: np.ndarray) -> np.ndarray:
            return lo_arr + np.asarray(p_s, dtype=float) * rng_arr

        def fun_scaled(p_s):
            return fun(_from_scaled(p_s))

        def jac_scaled(p_s):
            # chain rule: d/d(p_s) = d/d(p_nm) · d(p_nm)/d(p_s) = jac_nm · range
            return jac(_from_scaled(p_s)) * rng_arr

        scaled_bounds = [(0.0, 1.0)] * n
        p0_scaled = _to_scaled(np.asarray(p0, dtype=float))

        method = getattr(spec, "method", "Powell")
        print(f"[fd_gradient_design] {method} max_iter={spec.max_iter}, "
              f"eval_budget={eval_budget}, params rescaled to [0,1]")
        try:
            if method == "Powell":
                # Gradient-free: pass only fun, no jac.
                # direc parameter sets initial search directions; identity matrix
                # scaled by powell_initial_simplex_rel gives sensible first probes.
                n_params = spec.n_params()
                init_step = float(getattr(spec, "powell_initial_simplex_rel", 0.1))
                direc = np.eye(n_params) * init_step
                result = _minimize(
                    fun=fun_scaled,
                    x0=p0_scaled,
                    method="Powell",
                    bounds=scaled_bounds,
                    options={"maxiter": int(spec.max_iter),
                             "xtol": float(getattr(spec, "powell_xtol", 1e-3)),
                             "ftol": float(spec.ftol),
                             "direc": direc,
                             "disp": True},
                )
            else:
                # Gradient-based path (L-BFGS-B). Known to fail on our noisy
                # FOM but kept for reference.
                result = _minimize(
                    fun=fun_scaled,
                    x0=p0_scaled,
                    jac=jac_scaled,
                    method=method,
                    bounds=scaled_bounds,
                    options={"maxiter": int(spec.max_iter),
                             "ftol": float(spec.ftol),
                             "gtol": float(spec.gtol),
                             "disp": True},
                )
            p_final = list(map(float, _from_scaled(result.x)))
            fom_final = -float(result.fun)   # convert back to peak_T (we minimized -peak_T)
            convergence_msg = str(result.message) if hasattr(result.message, '__str__') else "unknown"
            print(f"[fd_gradient_design] L-BFGS-B done. peak_T = {fom_final:.6f}  "
                  f"params = {[f'{v:.2f}' for v in p_final]}  "
                  f"msg = {convergence_msg!r}")
        except _FDBudgetExceeded as exc:
            # Pull the best evaluation we've seen so far from the history.
            print(f"[fd_gradient_design] {exc}")
            best = max(history, key=lambda h: h["peak_T"])
            p_final = list(map(float, best["p_nm"]))
            fom_final = float(best["peak_T"])
            convergence_msg = "fd_budget_exceeded"
            print(f"[fd_gradient_design] best-of-history peak_T = {fom_final:.6f}")
    finally:
        fdtd.close()

    # Post-opt verification at higher mesh accuracy.
    print("[fd_gradient_design] post-opt verification at converged params (accurate mesh) ...")
    cfg_final = copy.deepcopy(cfg)
    cfg_final.mesh.simulation_mode = "accurate"
    fk = params_to_kwargs(list(p_final), spec.n_free_inner_teeth)
    cfg_final.grating.inner_dw_nm = list(fk["inner_dw_nm"])
    cfg_final.grating.inner_shift_nm = list(fk["inner_shift_nm"])
    cfg_final.grating.cavity_width_m = fk["cavity_width_m"]
    final_result = run_single_sim(cfg_final)
    lam_peak = float(final_result["resonance_wavelength_nm"]) * 1e-9
    T_peak   = float(final_result["resonance_transmission"])

    summary = {
        "start_idx":          int(start_idx),
        "label":              spec.label,
        "p_initial":          list(map(float, p0)),
        "p_final":            list(map(float, p_final)),
        "fom_final":          float(fom_final),
        "baseline_lambda_m":  float(baseline_lambda_m),
        "initial_peak_T":     (None if initial_peak_T is None else float(initial_peak_T)),
        "true_peak_lambda_m": float(lam_peak),
        "true_peak_T":        float(T_peak),
        "delta_peak_T":       (None if initial_peak_T is None else float(T_peak - initial_peak_T)),
        "n_free_inner_teeth": spec.n_free_inner_teeth,
        "algorithm":          "L-BFGS-B (central-diff jac)",
        "fd_step_nm":         float(spec.fd_step_nm),
        "max_iter":           int(spec.max_iter),
        "n_evals":            len(history),
        "convergence":        convergence_msg,
        "optimizer_history":  history,
    }
    with open(os.path.join(out_dir, "final_params.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if initial_peak_T is not None:
        print(f"[fd_gradient_design] done. peak T: initial = {initial_peak_T:.4f}  "
              f"-> final = {T_peak:.4f}  (delta = {T_peak - initial_peak_T:+.4f}) "
              f"at lambda = {lam_peak * 1e9:.3f} nm")
    else:
        print(f"[fd_gradient_design] done. true peak T = {T_peak:.4f} "
              f"at lambda = {lam_peak * 1e9:.3f} nm")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────


def _load_spec(spec_module: str) -> Tuple[FDGradientSpec, SimulationConfig]:
    mod = importlib.import_module(spec_module)
    if not hasattr(mod, "SPEC"):
        raise AttributeError(f"{spec_module} must define a top-level SPEC: FDGradientSpec.")
    return mod.SPEC, getattr(mod, "BASE", None) or SimulationConfig()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Finite-difference L-BFGS-B inverse design")
    ap.add_argument("--spec", required=True,
                    help="Module with SPEC (e.g. runners.fd_gradient_design.optimize_transmission)")
    ap.add_argument("--start", type=int, default=0, help="Starting-point index (default 0)")
    ap.add_argument("--baseline-lambda-nm", type=float, default=None,
                    help="Pre-measured baseline lambda_resonance in nm (skips the baseline run)")
    ap.add_argument("--output-root", default=None, help="Override output directory root")
    args = ap.parse_args(argv)

    spec, base = _load_spec(args.spec)
    base_lambda_m = args.baseline_lambda_nm * 1e-9 if args.baseline_lambda_nm else None
    run_fd_gradient_design(
        cfg=base,
        spec=spec,
        start_idx=args.start,
        output_root=args.output_root,
        baseline_lambda_m=base_lambda_m,
    )


if __name__ == "__main__":
    main()
