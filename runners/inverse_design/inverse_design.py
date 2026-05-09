"""
Inverse-design machinery for the pi-shift Bragg grating.

This single module contains everything needed to run a lumopt-based continuous-
adjoint shape optimization:

  - InverseDesignSpec dataclass (mirrors SweepSpec for sweeps)
  - regular_grating_start() helper for the deterministic starting point
  - params_to_kwargs() decomposition of the parameter vector
  - polygon vertex generator for lumopt's FunctionDefinedPolygon
  - run_inverse_design() entry point + CLI

A per-study config file (e.g. runners/inverse_design/peak_t.py) declares a
top-level SPEC: InverseDesignSpec and (optional) BASE: SimulationConfig that
this module consumes.

Layout of the optimization parameter vector p (length 2*N+1):
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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import config as _cfg
from simulation_config import SimulationConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spec dataclass and start-point helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InverseDesignSpec:
    """Configuration for one inverse-design study (per-study analog of SweepSpec)."""

    # ── Free parameters and bounds ──────────────────────────────────────────
    n_free_inner_teeth: int = 2
    # Per-tooth bounds on full corrugation depth (DW = W_wide - W_narrow), in nm.
    dw_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(60.0, 400.0), (60.0, 400.0)]
    )
    # Per-tooth bounds on gap shift, in nm.
    shift_bounds_nm: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 200.0), (0.0, 200.0)]
    )
    cavity_width_bounds_nm: Tuple[float, float] = (500.0, 1100.0)

    # ── Initial points (one per multi-start). When None and n_starts > 1,
    # Latin-hypercube samples are drawn. With n_starts = 1 (default) and
    # initial_points = None, the regular-grating point is used.
    initial_points: Optional[List[List[float]]] = None
    n_starts: int = 1
    seed: int = 0

    # ── Optimizer ───────────────────────────────────────────────────────────
    max_iter: int = 30
    optimizer_method: str = "L-BFGS-B"
    optimizer_pgtol: float = 1e-6
    optimizer_ftol: float = 1e-6
    use_concurrent_adjoint_solves: bool = True

    # ── Geometry constraints ────────────────────────────────────────────────
    enforce_mirror_symmetry: bool = True
    lengthen_cavity: bool = True

    # ── Study metadata ──────────────────────────────────────────────────────
    label: str = ""

    # ─────────────────────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return 2 * self.n_free_inner_teeth + 1

    def all_bounds_nm(self) -> List[Tuple[float, float]]:
        """Flat list of bounds in parameter-vector order: [dw..., shift..., cavity]."""
        return list(self.dw_bounds_nm) + list(self.shift_bounds_nm) + [self.cavity_width_bounds_nm]

    def validate(self) -> None:
        n = self.n_free_inner_teeth
        if n < 1:
            raise ValueError(f"n_free_inner_teeth must be >= 1, got {n}.")
        if len(self.dw_bounds_nm) != n:
            raise ValueError(
                f"dw_bounds_nm length ({len(self.dw_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if len(self.shift_bounds_nm) != n:
            raise ValueError(
                f"shift_bounds_nm length ({len(self.shift_bounds_nm)}) "
                f"must equal n_free_inner_teeth ({n})."
            )
        if self.initial_points is not None:
            for i, p in enumerate(self.initial_points):
                if len(p) != self.n_params():
                    raise ValueError(
                        f"initial_points[{i}] has length {len(p)}, "
                        f"expected {self.n_params()} (= 2*N + 1 with N={n})."
                    )

    def latin_hypercube_starts(self) -> List[List[float]]:
        rng = np.random.default_rng(self.seed)
        bounds = self.all_bounds_nm()
        d = len(bounds)
        u = (rng.random((self.n_starts, d)) +
             np.array([rng.permutation(self.n_starts) for _ in range(d)]).T) / self.n_starts
        return [
            [bounds[j][0] + u[i, j] * (bounds[j][1] - bounds[j][0]) for j in range(d)]
            for i in range(self.n_starts)
        ]

    def get_starts(self) -> List[List[float]]:
        self.validate()
        if self.initial_points is not None:
            return [list(p) for p in self.initial_points]
        return self.latin_hypercube_starts()

    def describe(self) -> str:
        lines = [f"InverseDesignSpec(label={self.label!r})"]
        lines.append(f"  n_free_inner_teeth = {self.n_free_inner_teeth}")
        lines.append(f"  parameters         = {self.n_params()}")
        lines.append(f"  dw_bounds_nm       = {self.dw_bounds_nm}")
        lines.append(f"  shift_bounds_nm    = {self.shift_bounds_nm}")
        lines.append(f"  cavity_bounds_nm   = {self.cavity_width_bounds_nm}")
        lines.append(f"  n_starts           = {self.n_starts}")
        lines.append(f"  max_iter           = {self.max_iter}")
        lines.append(f"  optimizer          = {self.optimizer_method}")
        return "\n".join(lines)


def regular_grating_start(
    cfg: SimulationConfig, n_free_inner_teeth: int, cavity_width_nm: float = 800.0,
) -> List[float]:
    """Parameter vector for the unmodified regular grating (recommended starting point).

    All N freed teeth use the same full corrugation depth as the rest of the
    grating (cfg.geometry.corrugation_depth_m), zero inner-tooth shift, and
    the supplied cavity width. Peak T at the optimum must exceed peak T at
    this start, otherwise the optimizer hasn't found an improvement.
    """
    full_depth_nm = cfg.geometry.corrugation_depth_m * 1e9
    return (
        [float(full_depth_nm)] * n_free_inner_teeth      # dw_d for d=1..N
        + [0.0] * n_free_inner_teeth                      # shift_d for d=1..N
        + [float(cavity_width_nm)]                        # cavity_width_nm
    )


def params_to_kwargs(p, n_free_inner_teeth: int) -> dict:
    """Decompose the flat parameter vector into bragg_device-style kwargs."""
    n = n_free_inner_teeth
    expected = 2 * n + 1
    if len(p) != expected:
        raise ValueError(
            f"params_to_kwargs: expected vector of length {expected} "
            f"(2*N+1 with N={n}), got {len(p)}."
        )
    return {
        "inner_dw_nm":     list(p[0:n]),
        "inner_shift_nm":  list(p[n:2 * n]),
        "cavity_width_m":  float(p[2 * n]) * 1e-9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Polygon vertex generator for lumopt's FunctionDefinedPolygon
#
# The freed (optimization) region spans the n_free innermost teeth on each
# side of the cavity plus the cavity itself plus one outer-shifted tooth on
# the right (d = n_free + 1, whose narrow gap absorbs shift_N). Its outer
# boundaries are at fixed x positions independent of the parameter vector
# (the cavity absorbs Σ shifts). Static outer teeth and access waveguides
# stay in a lumopt base_script; lumopt mutates only the freed-region polygon.
# ─────────────────────────────────────────────────────────────────────────────


def _envelope_mod_depth(d: int, cfg: SimulationConfig) -> float:
    """Apodization-envelope full corrugation depth for tooth d (in meters).
    Mirrors bragg_device._add_bragg_core's get_mod_depth() for d > n_free."""
    n_total = cfg.grating.n_periods_each_side
    n_apod = cfg.apodization.n_apod_periods_each_side if cfg.apodization.enabled else 0
    full_depth_edge = cfg.geometry.corrugation_depth_m
    full_depth_center = (cfg.apodization.center_mod_depth_nm * 1e-9) if cfg.apodization.enabled else full_depth_edge
    if cfg.apodization.enabled and d <= n_apod and n_total > 1:
        denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
        if denom == 0:
            return full_depth_center
        frac = (d - 1) / float(denom)
        if cfg.apodization.method == "tanh":
            steep = cfg.apodization.tanh_steepness
            frac = float(np.tanh(steep * 2.0 * frac) / np.tanh(2.0 * steep))
        return full_depth_center + (full_depth_edge - full_depth_center) * frac
    return full_depth_edge


def freed_region_x_bounds(cfg: SimulationConfig, n_free: int) -> Tuple[float, float]:
    """Fixed x-extent of the freed region (independent of the parameter vector)."""
    n_total = cfg.grating.n_periods_each_side
    pitch = cfg.grating.pitch_m
    if cfg.grating.cavity_neg_detuning_nm != 0.0:
        cavity_length_baseline = (pitch * 0.5) - cfg.grating.cavity_neg_detuning_nm * 1e-9
    else:
        cavity_length_baseline = pitch * 0.5
    x_grating_end = n_total * pitch + cavity_length_baseline / 2.0
    x_left_freed_start = -x_grating_end + (n_total - n_free) * pitch
    x_right_freed_end = -x_grating_end + (n_total + n_free + 1) * pitch + cavity_length_baseline
    return x_left_freed_start, x_right_freed_end


def freed_region_segments(cfg: SimulationConfig, p) -> List[Tuple[float, float, float]]:
    """Enumerate the freed-region segments (x_start, x_end, width_m) left-to-right."""
    n_free = cfg.grating.n_free_inner_teeth
    kw = params_to_kwargs(p, n_free)
    inner_dw_nm = kw["inner_dw_nm"]
    inner_shift_nm = kw["inner_shift_nm"]
    cavity_width_m = kw["cavity_width_m"]

    pitch = cfg.grating.pitch_m
    half_pitch = pitch * 0.5
    avg_w = cfg.geometry.avg_corrugation_width_m

    def tooth_widths(d: int) -> Tuple[float, float]:
        if 1 <= d <= n_free:
            mod_depth = float(inner_dw_nm[d - 1]) * 1e-9
        else:
            mod_depth = _envelope_mod_depth(d, cfg)
        delta = mod_depth * 0.5
        return avg_w - delta, avg_w + delta

    shift_m = [float(s) * 1e-9 for s in inner_shift_nm]
    sum_shifts = sum(shift_m)

    if cfg.grating.cavity_neg_detuning_nm != 0.0:
        cavity_length_baseline = half_pitch - cfg.grating.cavity_neg_detuning_nm * 1e-9
    else:
        cavity_length_baseline = half_pitch
    cavity_extra = 2.0 * sum_shifts if cfg.grating.lengthen_cavity else 0.0
    cavity_length_eff = cavity_length_baseline + cavity_extra

    x_start, _ = freed_region_x_bounds(cfg, n_free)
    x = x_start
    segs: List[Tuple[float, float, float]] = []

    # Left freed teeth: d = n_free, ..., 1
    for d in range(n_free, 0, -1):
        Wn, Ww = tooth_widths(d)
        nlen = half_pitch - shift_m[d - 1]
        segs.append((x, x + nlen, Wn));         x += nlen
        segs.append((x, x + half_pitch, Ww));   x += half_pitch

    # Cavity
    segs.append((x, x + cavity_length_eff, cavity_width_m));  x += cavity_length_eff

    # Right freed teeth: d = 1, ..., n_free+1
    # narrow_d_R is shortened by shift_{d-1} for d>=2 (full for d=1).
    for d in range(1, n_free + 2):
        Wn, Ww = tooth_widths(d)
        s_prev = shift_m[d - 2] if d >= 2 else 0.0
        nlen = half_pitch - s_prev
        if d == 1 and cfg.grating.cavity_width_option == "avg_ext":
            Wn = avg_w
        segs.append((x, x + nlen, Wn));         x += nlen
        segs.append((x, x + half_pitch, Ww));   x += half_pitch

    return segs


def polygon_vertices(cfg: SimulationConfig, p) -> np.ndarray:
    """Closed polygon vertex array (CCW) for the freed region, mirror-symmetric about y=0."""
    segs = freed_region_segments(cfg, p)
    top: List[Tuple[float, float]] = []
    for (x0, x1, w) in segs:
        y_top = +0.5 * w
        if not top or top[-1][1] != y_top:
            top.append((x0, y_top))
        top.append((x1, y_top))
    bot: List[Tuple[float, float]] = []
    for (x0, x1, w) in reversed(segs):
        y_bot = -0.5 * w
        if not bot or bot[-1][1] != y_bot:
            bot.append((x1, y_bot))
        bot.append((x0, y_bot))
    return np.array(top + bot, dtype=float)


def make_polygon_callback(cfg: SimulationConfig):
    """Closure for lumopt's FunctionDefinedPolygon `func` argument."""
    def _callback(p):
        return polygon_vertices(cfg, p)
    return _callback


def make_lumopt_geometry(cfg: SimulationConfig, spec: InverseDesignSpec, initial_p):
    """Construct a lumopt FunctionDefinedPolygon for the freed region.

    Lazy-imports lumopt — this module can be inspected/tested without a Lumerical install.
    """
    from lumopt.geometries.polygon import FunctionDefinedPolygon  # type: ignore

    eps_in = cfg.material.n_core_const ** 2
    eps_out = cfg.material.n_clad_const ** 2

    return FunctionDefinedPolygon(
        func=make_polygon_callback(cfg),
        initial_params=np.array(initial_p, dtype=float),
        bounds=spec.all_bounds_nm(),
        z=0.0,
        depth=cfg.geometry.core_height_m,
        eps_out=eps_out,
        eps_in=eps_in,
        edge_precision=5,
        dx=1.0,   # FD step for jacobian, in nm (parameter units)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. lumopt orchestration: baseline λ, base_script, and run_inverse_design
# ─────────────────────────────────────────────────────────────────────────────


def measure_baseline(cfg: SimulationConfig) -> Tuple[float, float]:
    """Run a single FDTD at the cfg's baseline geometry; return (λ_resonance, peak_T).

    Uses the resonance scalars that run_single_sim's post_processing pipeline
    already computes — no need to re-run the peak finder here.
    """
    from runners.single.run_simulation import run_single_sim

    print("[inverse_design] measuring baseline λ_resonance and peak T ...")
    result = run_single_sim(cfg)
    lam_peak = float(result["resonance_wavelength_nm"]) * 1e-9
    T_peak   = float(result["resonance_transmission"])
    print(f"[inverse_design] baseline λ_resonance = {lam_peak * 1e9:.3f} nm  (T = {T_peak:.4f})")
    return lam_peak, T_peak


def _build_static_skeleton(sim, cfg: SimulationConfig, n_free: int) -> None:
    """Add only the STATIC core rectangles to `sim.fdtd` (skip the freed region).

    Mirrors the relevant slice of bragg_device._add_bragg_core: builds the
    access waveguides + the outer teeth (d > n_free on the left, d > n_free+1
    on the right). The freed region (n_free innermost teeth on each side +
    cavity + d=n_free+1 right) is intentionally omitted — lumopt's
    FunctionDefinedPolygon fills it in each iteration.
    """
    fdtd = sim.fdtd
    pitch = sim.pitch
    half_pitch = pitch / 2.0
    seg_id = 0

    avg_w = 0.5 * (sim.width_narrow + sim.width_wide)
    full_depth_edge = sim.width_wide - sim.width_narrow
    full_depth_center = sim.center_mod_depth if sim.use_apodization else full_depth_edge
    n_total = sim.n_periods_each_side
    n_apod = sim.n_apod_periods_each_side
    apod_method = sim.apod_method
    tanh_steepness = sim.tanh_steepness

    def get_mod_depth(d):
        if d <= n_apod and n_total > 1:
            denom = (n_apod - 1) if (n_apod > 1 and n_apod == n_total) else n_apod
            if denom == 0:
                return full_depth_center
            frac = (d - 1) / float(denom)
            if apod_method == "tanh":
                frac = float(np.tanh(tanh_steepness * 2.0 * frac) / np.tanh(2.0 * tanh_steepness))
            return full_depth_center + (full_depth_edge - full_depth_center) * frac
        return full_depth_edge

    def add_seg(x1, x2, width, name):
        nonlocal seg_id
        seg_id += 1
        fdtd.addrect()
        fdtd.set("name", f"{name}_{seg_id:d}")
        fdtd.set("material", sim.core_material)
        fdtd.set("y", 0)
        fdtd.set("y span", width)
        fdtd.set("z", 0.0)
        fdtd.set("z span", sim.core_height)
        fdtd.set("x min", x1)
        fdtd.set("x max", x2)

    # Left access waveguide
    x_grating_start = -sim.x_grating_end
    add_seg(-sim.x_sim_boundary - 1e-6, x_grating_start, sim.width_port, "wg_left_inf")

    # Left static teeth: d = n_total ... n_free+1  (full pitch each, no shift)
    x = x_grating_start
    for d in range(n_total, n_free, -1):
        mod_depth = get_mod_depth(d)
        Wn = avg_w - mod_depth / 2.0
        Ww = avg_w + mod_depth / 2.0
        add_seg(x, x + half_pitch, Wn, f"L_narrow_{d}");  x += half_pitch
        add_seg(x, x + half_pitch, Ww, f"L_wide_{d}");    x += half_pitch
    # x now equals x_left_freed_start — the freed region starts here.
    # Skip to x_right_freed_end (handled by lumopt's polygon).

    # Right static teeth: d = n_free+2 ... n_total  (full pitch each, no shift).
    # Their starting position is x_right_freed_end (closed-form, see freed_region_x_bounds).
    cavity_length_baseline = sim.cavity_length  # detuning already applied in __init__
    x = -sim.x_grating_end + (n_total + n_free + 1) * pitch + cavity_length_baseline
    for d in range(n_free + 2, n_total + 1):
        mod_depth = get_mod_depth(d)
        Wn = avg_w - mod_depth / 2.0
        Ww = avg_w + mod_depth / 2.0
        add_seg(x, x + half_pitch, Wn, f"R_narrow_{d}");  x += half_pitch
        add_seg(x, x + half_pitch, Ww, f"R_wide_{d}");    x += half_pitch

    # Right access waveguide
    add_seg(x, sim.x_sim_boundary + 1e-6, sim.width_port, "wg_right_inf")


def make_base_script(cfg: SimulationConfig, n_free: int):
    """Callback that builds the static portion of the FDTD simulation.

    Builds the FDTD region, mesh override, materials, ports/source, and
    transmission monitor, plus all STATIC core rectangles. The freed region
    is intentionally NOT built here — lumopt's FunctionDefinedPolygon fills
    it in each iteration with the current parameter vector.
    """
    from bragg_device import PiShiftBraggFDTD

    def _callback(fdtd):
        kwargs = cfg.to_device_kwargs()
        # n_free is structural (controls freed-region x bounds); inner_*_nm
        # are irrelevant in the skeleton (no freed rectangles built here).
        kwargs["n_free_inner_teeth"] = n_free
        kwargs["inner_dw_nm"] = None
        kwargs["inner_shift_nm"] = [0.0] * n_free
        sim = PiShiftBraggFDTD(**kwargs)
        sim.fdtd = fdtd  # use lumopt's CAD session
        sim._setup_materials()
        sim._add_fdtd_region()
        sim._add_aligned_mesh_override()
        _build_static_skeleton(sim, cfg, n_free)
        sim._add_source_and_monitors()
        sim.update_scan(
            cfg.spectral.center_wavelength_m,
            cfg.spectral.scan_width_nm,
            cfg.spectral.n_wl_points,
        )

    return _callback


def run_inverse_design(
    cfg: SimulationConfig,
    spec: InverseDesignSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run the lumopt continuous-adjoint optimization for one starting point.

    With n_starts = 1 (the recommended default), start_idx = 0 picks the
    regular-grating starting point.
    """
    # lumopt lives next to lumapi.py inside the Lumerical install. Plain `import
    # lumopt` doesn't find it unless that directory is on sys.path. Mirror the
    # bragg_device.py / athena_run_one.py pattern that already sets up lumapi.
    import sys as _sys
    _lum_dir = os.path.dirname(_cfg.LUMAPI_PATH)
    if _lum_dir and _lum_dir not in _sys.path:
        _sys.path.insert(0, _lum_dir)

    from lumopt.optimization import Optimization                          # type: ignore
    from lumopt.figures_of_merit.PortTransmission import porttransmission # type: ignore
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers       # type: ignore
    from sim_helpers import peak_t_diagnostic
    from runners.single.run_simulation import run_single_sim

    spec.validate()
    starts = spec.get_starts() if spec.initial_points or spec.n_starts > 1 else [
        regular_grating_start(cfg, spec.n_free_inner_teeth)
    ]
    if not (0 <= start_idx < len(starts)):
        raise IndexError(f"start_idx {start_idx} out of range (have {len(starts)} starts).")
    p0 = starts[start_idx]

    # Configure cfg for the inverse-design path.
    cfg = copy.deepcopy(cfg)
    cfg.grating.n_free_inner_teeth = spec.n_free_inner_teeth
    cfg.grating.lengthen_cavity = spec.lengthen_cavity
    init_kwargs = params_to_kwargs(p0, spec.n_free_inner_teeth)
    cfg.grating.inner_dw_nm = list(init_kwargs["inner_dw_nm"])
    cfg.grating.inner_shift_nm = list(init_kwargs["inner_shift_nm"])
    cfg.grating.cavity_width_m = init_kwargs["cavity_width_m"]

    # Output directory
    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "inverse_design", spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline λ_resonance + peak T at the initial (regular-grating) geometry.
    # Both numbers come from the same FDTD run — λ feeds the optimizer's FOM
    # wavelength, T_peak is kept for the final initial-vs-optimum comparison.
    initial_peak_T: Optional[float] = None
    if baseline_lambda_m is None:
        baseline_lambda_m, initial_peak_T = measure_baseline(cfg)

    # FOM: single-wavelength PortTransmission on Port_2.
    # This is the modal-overlap transmission (= |S21|²), evaluated at the
    # baseline λ_resonance — same number as your existing T(λ) curves, but
    # adjoint-differentiable and read directly from the FDTD port object.
    fom = porttransmission(
        monitor_port="Port_2",
        mode_number=1,
        direction="Forward",
        target_T_fwd=lambda wl: np.ones_like(wl),
        norm_p=1,
        target_fom=0.0,
    )

    geometry = make_lumopt_geometry(cfg, spec, p0)

    optimizer = ScipyOptimizers(
        max_iter=spec.max_iter,
        method=spec.optimizer_method,
        scaling_factor=1.0,
        pgtol=spec.optimizer_pgtol,
        ftol=spec.optimizer_ftol,
    )

    base_script = make_base_script(cfg, spec.n_free_inner_teeth)
    opt = Optimization(
        base_script=base_script,
        wavelengths=np.array([baseline_lambda_m]),
        fom=fom,
        geometry=geometry,
        optimizer=optimizer,
        use_var_fdtd=False,
        hide_fdtd_cad=True,
        use_deps=True,
        store_all_simulations=True,
        save_global_index=False,
        label=f"{spec.label}_start{start_idx}",
        source_name="source",
    )

    print(f"[inverse_design] running lumopt for start{start_idx}, p0={p0} ...")
    fom_final, params_final = opt.run(working_dir=out_dir)

    # Post-optimization: re-simulate optimum and read the resonance scalars
    # that run_single_sim's post_processing pipeline already computes.
    print(f"[inverse_design] post-opt verification at converged params ...")
    cfg_final = copy.deepcopy(cfg)
    fk = params_to_kwargs(list(params_final), spec.n_free_inner_teeth)
    cfg_final.grating.inner_dw_nm = list(fk["inner_dw_nm"])
    cfg_final.grating.inner_shift_nm = list(fk["inner_shift_nm"])
    cfg_final.grating.cavity_width_m = fk["cavity_width_m"]
    final_result = run_single_sim(cfg_final)
    lam_peak = float(final_result["resonance_wavelength_nm"]) * 1e-9
    T_peak   = float(final_result["resonance_transmission"])

    summary = {
        "start_idx": int(start_idx),
        "label": spec.label,
        "p_initial": list(map(float, p0)),
        "p_final": list(map(float, params_final)),
        "fom_final": float(fom_final),
        "baseline_lambda_m": float(baseline_lambda_m),
        "initial_peak_T":     (None if initial_peak_T is None else float(initial_peak_T)),
        "true_peak_lambda_m": float(lam_peak),
        "true_peak_T":        float(T_peak),
        "delta_peak_T":       (None if initial_peak_T is None else float(T_peak - initial_peak_T)),
        "n_free_inner_teeth": spec.n_free_inner_teeth,
    }
    with open(os.path.join(out_dir, "final_params.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if initial_peak_T is not None:
        print(f"[inverse_design] done. peak T: initial = {initial_peak_T:.4f}  "
              f"→ final = {T_peak:.4f}  (Δ = {T_peak - initial_peak_T:+.4f}) "
              f"at λ = {lam_peak * 1e9:.3f} nm")
    else:
        print(f"[inverse_design] done. true peak T = {T_peak:.4f} at λ = {lam_peak * 1e9:.3f} nm "
              f"(initial peak T not available — baseline was pre-supplied)")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _load_spec(spec_module: str) -> Tuple[InverseDesignSpec, SimulationConfig]:
    mod = importlib.import_module(spec_module)
    if not hasattr(mod, "SPEC"):
        raise AttributeError(f"{spec_module} must define a top-level SPEC: InverseDesignSpec.")
    return mod.SPEC, getattr(mod, "BASE", None) or SimulationConfig()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="lumopt adjoint inverse design driver")
    ap.add_argument("--spec", required=True,
                    help="Module with SPEC (e.g. runners.inverse_design.peak_t)")
    ap.add_argument("--start", type=int, default=0,
                    help="Starting-point index (default 0; only the regular-grating start with n_starts=1)")
    ap.add_argument("--baseline-lambda-nm", type=float, default=None,
                    help="Pre-measured baseline λ_resonance in nm (skips the baseline run)")
    ap.add_argument("--output-root", default=None, help="Override output directory root")
    args = ap.parse_args(argv)

    spec, base = _load_spec(args.spec)
    base_lambda_m = args.baseline_lambda_nm * 1e-9 if args.baseline_lambda_nm else None
    run_inverse_design(
        cfg=base,
        spec=spec,
        start_idx=args.start,
        output_root=args.output_root,
        baseline_lambda_m=base_lambda_m,
    )


if __name__ == "__main__":
    main()
