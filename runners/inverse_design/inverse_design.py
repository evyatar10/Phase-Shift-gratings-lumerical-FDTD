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
    max_iter: int = 16
    optimizer_method: str = "L-BFGS-B"
    optimizer_pgtol: float = 1e-6
    optimizer_ftol: float = 1e-6
    use_concurrent_adjoint_solves: bool = True

    # ── FOM (Gaussian-weighted modal transmission across the bandgap) ───────
    # The Gaussian weight σ=1 nm is wide enough that the resonance stays in
    # the FOM band even with ±2 nm drift. No outer-loop recentering needed.
    fom_window_nm: float = 10.0
    fom_n_points: int = 201
    fom_weight_sigma_nm: float = 1.0

    # ── Mesh & adjoint perturbation knobs ──────────────────────────────────
    # Set mesh_override_dxyz_nm = 0 to skip the fine override and use the
    # device-wide periodic-aligned 50 nm mesh. param_dx_nm should be ≥ mesh
    # cell so finite-difference perturbations actually change the discretised
    # eps (gradient nonzero).
    mesh_override_dxyz_nm: float = 0.0
    param_dx_nm: float = 50.0

    # ── Geometry constraints ────────────────────────────────────────────────
    enforce_mirror_symmetry: bool = True
    lengthen_cavity: bool = True

    # ── Run mode ────────────────────────────────────────────────────────────
    # "optimize"        : standard L-BFGS-B optimization (default)
    # "check_gradient"  : compare adjoint vs FD gradient at p0 (no optimization)
    # "scan_landscape"  : evaluate FOM along cavity_width across its bounds
    #                     (other params held at p0); diagnoses local plateaus
    mode: str = "optimize"
    check_gradient_dx_nm: float = 50.0
    scan_n_points: int = 9

    # ── Outer-loop active-set re-centering (lumopt only) ────────────────────
    # When n_outer_iters > 1, run_inverse_design_outer_loop alternates inner
    # adjoint optimization (single-λ FOM, GPU-correct) with re-measuring the
    # resonance λ between outer iters. This is the GPU-friendly fix for the
    # broadband-port-profile bug: at fom_n_points=1 the adjoint is
    # mathematically consistent with `frequency_dependent_profile=0`.
    # Each outer iter: max_iter inner iters → re-measure λ → restart.
    # Total inner iterations = n_outer_iters * max_iter.
    n_outer_iters: int = 1

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
    """Closed polygon vertex array (CCW) for the freed region, mirror-symmetric about y=0.

    Retained for `test_geometry.py` and visualization. The optimizer no longer
    consumes this — it uses `render_freed_rectangles` via ParameterizedGeometry
    (Phase-2 fix #1: polygons and rectangles render to different Yee meshes;
    the official Ansys grating-coupler example was migrated to
    ParameterizedGeometry for exactly this reason).
    """
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


def freed_region_named_rects(cfg: SimulationConfig, p) -> List[Tuple[str, float, float, float]]:
    """Return the freed-region rectangles as `(name, x_center, x_span, y_span)` tuples.

    Names are stable across iterations (positional index) so lumopt's
    `only_update=True` path can `setnamed(name, "x", ...)` without rebuilding.
    """
    segs = freed_region_segments(cfg, p)
    out: List[Tuple[str, float, float, float]] = []
    for i, (x0, x1, w) in enumerate(segs):
        x_center = 0.5 * (x0 + x1)
        x_span = x1 - x0
        out.append((f"freed_seg_{i:02d}", float(x_center), float(x_span), float(w)))
    return out


def make_render_callback(cfg: SimulationConfig, n_free: int):
    """Closure for lumopt's ParameterizedGeometry `func` argument.

    Signature is `(params, fdtd, only_update)` — lumopt enforces this via
    `inspect.signature`. On the first call (only_update=False), each rectangle
    is created via `addrect`. Subsequent calls (only_update=True) just update
    x, x span, y span via `setnamed`.

    Each freed segment is a separate named `addrect`. With n_free=2:
      Left:  freed_seg_00 (narrow_2_outer), freed_seg_01 (wide_2),
             freed_seg_02 (narrow_2_inner), freed_seg_03 (wide_1),
             freed_seg_04 (narrow_1)
      Cavity: freed_seg_05
      Right: freed_seg_06 (narrow_1), freed_seg_07 (wide_1),
             freed_seg_08 (narrow_2_inner), freed_seg_09 (wide_2),
             freed_seg_10 (narrow_2_outer)
    """
    core_material = None  # captured lazily on first call from the live FDTD object
    core_height = float(cfg.geometry.core_height_m)
    z_center = 0.0

    def _resolve_core_material(fdtd) -> str:
        """Find the core material name as it was registered by _setup_materials."""
        # bragg_device.py:_setup_materials may install a custom Si3N4 dispersion;
        # `core_material` on the device gets a name like 'const_sin' or
        # 'custom_sin'. The static skeleton is built before this callback runs,
        # so we can read the material name from one of the existing static rects.
        try:
            return str(fdtd.getnamed("wg_left_inf_1", "material"))
        except Exception:
            # Fall back to the const-index core material.
            return cfg.material.core_material if hasattr(cfg.material, "core_material") else "Si3N4 (Silicon Nitride) - Luke"

    def _render(params, fdtd, only_update: bool):
        nonlocal core_material
        rects = freed_region_named_rects(cfg, params)
        if not only_update:
            fdtd.switchtolayout()
            if core_material is None:
                core_material = _resolve_core_material(fdtd)
        for name, xc, xs, ys in rects:
            if not only_update:
                fdtd.addrect()
                fdtd.set("name", name)
                fdtd.set("material", core_material)
                fdtd.set("z", z_center)
                fdtd.set("z span", core_height)
            fdtd.setnamed(name, "x",      xc)
            fdtd.setnamed(name, "x span", xs)
            fdtd.setnamed(name, "y",      0.0)
            fdtd.setnamed(name, "y span", ys)

    return _render


def make_polygon_callback(cfg: SimulationConfig):
    """Closure for lumopt's FunctionDefinedPolygon `func` argument (legacy path)."""
    def _callback(p):
        return polygon_vertices(cfg, p)
    return _callback


def make_lumopt_geometry(cfg: SimulationConfig, spec: InverseDesignSpec, initial_p):
    """Construct a lumopt ParameterizedGeometry for the freed region.

    Phase-2 fix #1: rectangles instead of one polygon — eliminates the
    polygon-vs-rectangle Yee-mesh staircasing discrepancy that was causing
    the polygon-rendered geometry to give T=0.752 vs baseline T=0.866 at
    the same wavelength, and the corresponding zero-gradient stall.

    ParameterizedGeometry uses finite-difference gradients (no analytic shape
    derivative). `dx` here is the parameter perturbation in nm — must be
    larger than the override mesh cell size so the perturbation actually
    changes the discretised eps. With mesh_override_dxyz_nm=10 nm, dx=1 nm
    is the canonical choice (mesh/10).

    Lazy-imports lumopt — this module can be inspected/tested without a
    Lumerical install.
    """
    from lumopt.geometries.parameterized_geometry import ParameterizedGeometry  # type: ignore

    return ParameterizedGeometry(
        func=make_render_callback(cfg, spec.n_free_inner_teeth),
        initial_params=np.array(initial_p, dtype=float),
        bounds=spec.all_bounds_nm(),
        dx=float(spec.param_dx_nm),
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


def make_base_script(cfg: SimulationConfig, n_free: int, target_lambda_m: float,
                     fom_window_nm: float = 10.0, fom_n_points: int = 601,
                     mesh_override_dxyz_nm: float = 10.0):
    """Callback that builds the static portion of the FDTD simulation.

    Builds the FDTD region, mesh override, materials, ports/source, and
    transmission monitor, plus all STATIC core rectangles. The freed region
    is intentionally NOT built here — lumopt's ParameterizedGeometry fills
    it in each iteration with the current parameter vector.

    Wavelength sampling (Phase-2 fix #4): `fom_window_nm`=10 nm (full bandgap)
    sampled at `fom_n_points`=601 (≥18 samples per resonance FWHM). The FOM
    weight is centered on `target_lambda_m` with σ=fom_weight_sigma_nm
    (Phase-2 fix #5) so the weight has support even if the resonance drifts.

    Override-mesh box (Phase-2 fix #2,#3): a fine mesh of cell size
    `mesh_override_dxyz_nm` (default 10 nm) is added over the freed region
    plus a 2-cell margin. This is on top of the device-wide override and
    is what lets parameter perturbations actually change the discretised eps.
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
        # Wide scan over the entire bandgap; the FOM weight (set elsewhere via
        # target_T_fwd_weights) restricts the integral to the resonance
        # neighbourhood while keeping the simulation band wide enough that
        # the resonance never escapes during optimization.
        sim.update_scan(
            target_lambda_m,
            fom_window_nm,
            fom_n_points,
        )
        # Rename ports to match lumopt's PortTransmission convention
        # (it looks specifically for ports named "source" and "fom").
        fdtd.setnamed("FDTD::ports::Port_1", "name", "source")
        fdtd.setnamed("FDTD::ports::Port_2", "name", "fom")

        # GPU FDTD does not support frequency-dependent profiles. bragg_device
        # turns this on by default (needed for some non-GPU paths), but with
        # GPU + lumopt the simulation hangs/aborts. Explicitly disable on both
        # ports for the inverse-design path.
        fdtd.setnamed("FDTD::ports::source", "frequency dependent profile", 0)
        fdtd.setnamed("FDTD::ports::fom", "frequency dependent profile", 0)

        # opt_fields monitor: covers the freed region. Lumopt records E,H here
        # on every iteration to compute dFOM/dε for adjoint. The adjoint
        # integration in lumopt's `spatial_gradient_integral_on_cad` performs
        # `integrate2(integrand, [1,2,3], x, y, z)` — a real 3D volume integral
        # weighted per-wavelength. The monitor MUST be 3D (z-axis must have
        # multiple sample points covering the core slab) and MUST have
        # frequency points matching the FOM sampling, otherwise lumopt
        # silently produces wildly mis-scaled gradients (off by ~10⁸ in our
        # 5-DOF Bragg case — confirmed via check_gradient diagnostic 78914
        # against finite-difference). z span = core_height × 3 covers the
        # mode profile (core + ~core-height of evanescent tail in cladding).
        x_lo, x_hi = freed_region_x_bounds(cfg, n_free)
        # SiN-on-SiO2 is weakly guiding (n_core=1.977, n_clad=1.44 → 1/e mode-tail
        # decay length ~180 nm at 1560 nm). Capturing the full mode requires
        # opt_fields y/z spans matching the FDTD region (cfg.z_span ≈ 3.1 µm,
        # cfg.y_span ≈ 3.6 µm). Truncating the mode tail under-counts the field
        # integral and biases adjoint gradients high.
        # Phase-3 fix: was 0.9× — now 0.99× (just inside FDTD region to avoid
        # PML overlap warnings). The 0.9× truncation may have contributed to
        # the residual FD-vs-adjoint mismatch after the target_T_fwd_weights
        # patch (vec_error 11.79 → 11.40 → ?).
        fdtd.addpower()
        fdtd.set("name", "opt_fields")
        fdtd.set("monitor type", "3D")
        fdtd.set("x min", x_lo)
        fdtd.set("x max", x_hi)
        fdtd.set("y", 0.0)
        fdtd.set("y span", cfg.y_span * 0.99)
        fdtd.set("z", 0.0)
        fdtd.set("z span", cfg.z_span * 0.99)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("use source limits", 1)
        fdtd.set("frequency points", int(fom_n_points))

        # Phase-2 fix #2,#3: fine override mesh over the freed region + margin.
        # Per the Y-splitter canonical pattern the mesh box must be LARGER
        # than opt_fields by 2·mesh_x so adjoint integration sees a clean
        # mesh at the boundary.
        # Skip if mesh_override_dxyz_nm <= 0 — at n_periods=80, fine override
        # makes FDTD 8x slower and doesn't fit in walltime. Use the device-
        # wide 50 nm mesh + dx_param=50 nm instead.
        if mesh_override_dxyz_nm and mesh_override_dxyz_nm > 0:
            mesh_dx = mesh_override_dxyz_nm * 1e-9
            margin = 2.0 * mesh_dx
            fdtd.addmesh()
            fdtd.set("name", "mesh_override_freed")
            fdtd.set("x min", x_lo - margin)
            fdtd.set("x max", x_hi + margin)
            fdtd.set("y", 0.0)
            fdtd.set("y span", cfg.geometry.width_wide_m * 1.4 + 2 * margin)
            fdtd.set("z", 0.0)
            fdtd.set("z span", cfg.geometry.core_height_m * 1.4 + 2 * margin)
            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 1)
            fdtd.set("dx", mesh_dx)
            fdtd.set("dy", mesh_dx)
            fdtd.set("dz", mesh_dx)

    return _callback


def _patch_porttransmission_weights():
    """Monkey-patch lumopt's `porttransmission` to fix the dropped Gaussian
    weight in the adjoint integral kernel.

    Bug (verified by check_gradient diagnostics 78712, 78715, fine_mesh, smalldx
    — all gave vec_error 12-44 vs healthy <0.1):

        Forward FOM:    error_term = (∫ w(λ)·|T-target|^p / range)^{1/p}
        ∂/∂T(λ):        ∂error_term/∂T(λ) = error_term^{1-p} · w(λ)·|err|^{p-1}·sign(err) / range

    The current lumopt code pre-multiplies the error by w (`T_fwd_error =
    w·(T-target)`) then takes `|T_fwd_error|^{p-1}·sign(T_fwd_error)`. For p=1
    (our case) this collapses to `sign(T-target)` and the w(λ) factor is lost
    entirely. Result: the adjoint sees a uniformly weighted gradient (over the
    full 10 nm scan) while the forward FOM is Gaussian-weighted (σ=1 nm) —
    explains the observed 10-100× gradient inflation.

    This patch replaces both gradient methods (`fom_gradient_wavelength_integral_on_cad`
    and the static `fom_gradient_wavelength_integral_impl`) with versions that
    keep the weight inside the kernel for any p ≥ 1.
    """
    from lumopt.figures_of_merit.PortTransmission import porttransmission  # type: ignore
    import lumapi  # type: ignore

    if getattr(porttransmission, "_weights_patch_applied", False):
        return  # idempotent

    def fom_gradient_wavelength_integral_on_cad(self, sim, grad_var_name, wl):
        assert np.allclose(wl, self.wavelengths)

        target_T_fwd_vs_wavelength = self.target_T_fwd(wl).flatten()
        weights_vs_wavelength = self.target_T_fwd_weights(wl).flatten()
        # NB: do NOT pre-multiply T_fwd_error by weights — keep error pure
        # so that |err|^{p-1}·sign(err) carries the right magnitude, then
        # multiply by w(λ) explicitly in the kernel.
        T_fwd_error = self.T_fwd_vs_wavelength - target_T_fwd_vs_wavelength

        if wl.size > 1:
            wavelength_range = wl.max() - wl.min()
            # Forward error_term integrand (matches forward FOM, NOT |w·err|^p):
            T_fwd_error_integrand = np.multiply(
                weights_vs_wavelength, np.power(np.abs(T_fwd_error), self.norm_p)
            ) / wavelength_range
            const_factor = -1.0 * np.power(
                np.trapz(y=T_fwd_error_integrand, x=wl), 1.0 / self.norm_p - 1.0
            )
            # Per-wavelength derivative: w(λ) · |err|^{p-1} · sign(err) / range
            integral_kernel = (
                weights_vs_wavelength
                * np.power(np.abs(T_fwd_error), self.norm_p - 1)
                * np.sign(T_fwd_error)
                / wavelength_range
            )

            d = np.diff(wl)
            quad_weight = np.append(np.append(d[0], d[0:-1] + d[1:]), d[-1]) / 2
            v = const_factor * integral_kernel * quad_weight

            lumapi.putMatrix(sim.fdtd.handle, "wl_scaled_integral_kernel", v)
            sim.fdtd.eval((
                "dF_dp_s=size({0});"
                "dF_dp2 = reshape(permute({0},[3,2,1]),[dF_dp_s(3),dF_dp_s(2)*dF_dp_s(1)]);"
                "T_fwd_partial_derivs=real(mult(transpose(wl_scaled_integral_kernel),dF_dp2));"
            ).format(grad_var_name))
            T_fwd_partial_derivs_on_cad = sim.fdtd.getv("T_fwd_partial_derivs")
        else:
            sim.fdtd.eval(("T_fwd_partial_derivs=real({0});").format(grad_var_name))
            T_fwd_partial_derivs_on_cad = sim.fdtd.getv("T_fwd_partial_derivs")
            T_fwd_partial_derivs_on_cad *= -1.0 * weights_vs_wavelength * np.sign(T_fwd_error)

        return T_fwd_partial_derivs_on_cad.flatten()

    @staticmethod
    def fom_gradient_wavelength_integral_impl(
        T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, target_T_fwd_vs_wavelength,
        wl, norm_p, target_T_fwd_weights,
    ):
        if wl.size > 1:
            assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
            wavelength_range = wl.max() - wl.min()
            T_fwd_error = T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
            T_fwd_error_integrand = np.multiply(
                target_T_fwd_weights, np.power(np.abs(T_fwd_error), norm_p)
            ) / wavelength_range
            const_factor = -1.0 * np.power(
                np.trapz(y=T_fwd_error_integrand, x=wl), 1.0 / norm_p - 1.0
            )
            integral_kernel = (
                target_T_fwd_weights
                * np.power(np.abs(T_fwd_error), norm_p - 1)
                * np.sign(T_fwd_error)
                / wavelength_range
            )

            num_opt_param = T_fwd_partial_derivs_vs_wl.shape[0]
            T_fwd_partial_derivs = np.zeros(num_opt_param, dtype="complex")
            for i in range(num_opt_param):
                T_fwd_partial_derivs[i] = const_factor * np.trapz(
                    y=integral_kernel * T_fwd_partial_derivs_vs_wl[i], x=wl
                )
        else:
            T_fwd_partial_derivs = (
                -1.0 * target_T_fwd_weights
                * np.sign(T_fwd_vs_wavelength - target_T_fwd_vs_wavelength)
                * T_fwd_partial_derivs_vs_wl.flatten()
            )

        return T_fwd_partial_derivs.real, T_fwd_partial_derivs_vs_wl.real

    porttransmission.fom_gradient_wavelength_integral_on_cad = (
        fom_gradient_wavelength_integral_on_cad
    )
    porttransmission.fom_gradient_wavelength_integral_impl = (
        fom_gradient_wavelength_integral_impl
    )
    porttransmission._weights_patch_applied = True
    print("[inverse_design] applied target_T_fwd_weights adjoint patch to porttransmission.")


def _gaussian_weight_fn(lam_target_m: float, sigma_nm: float):
    """Wide Gaussian weight on T(λ), centered on `lam_target_m`.

    Phase-2 fix #5: replaces the prior Lorentzian HWHM=0.15 nm. With
    sigma_nm=1.0 the weight has significant support out to ~3 nm so the
    optimizer still gets gradient signal even when the resonance drifts
    by up to a few FWHM during inner-loop iterations.
    """
    sigma_m = sigma_nm * 1e-9

    def _w(wl):
        wl = np.asarray(wl).flatten()
        return np.exp(-0.5 * ((wl - lam_target_m) / sigma_m) ** 2)

    return _w


def run_inverse_design(
    cfg: SimulationConfig,
    spec: InverseDesignSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Run lumopt's continuous-adjoint optimization for one starting point.

    Single-loop L-BFGS-B with `spec.max_iter` iterations. The Gaussian-weighted
    PortTransmission FOM is centered on the BASELINE λ_resonance — wide enough
    (σ=1 nm) that the moving resonance stays inside its support throughout.
    Post-opt verification re-simulates the converged geometry at a finer mesh
    (`simulation_mode='accurate'`) for a physics-precise true_peak_T.
    """
    # lumopt lives next to lumapi.py inside the Lumerical install.
    import sys as _sys
    _lum_dir = os.path.dirname(_cfg.LUMAPI_PATH)
    if _lum_dir and _lum_dir not in _sys.path:
        _sys.path.insert(0, _lum_dir)

    from lumopt.optimization import Optimization                           # type: ignore
    from lumopt.figures_of_merit.PortTransmission import porttransmission  # type: ignore
    from lumopt.optimizers.generic_optimizers import ScipyOptimizers        # type: ignore
    from runners.single.run_simulation import run_single_sim

    # Apply the target_T_fwd_weights adjoint patch BEFORE constructing any
    # porttransmission instance. Idempotent — safe to call multiple times.
    _patch_porttransmission_weights()

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

    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "inverse_design", spec.label or "study")
    out_dir = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir, exist_ok=True)

    initial_peak_T: Optional[float] = None
    if baseline_lambda_m is None:
        baseline_lambda_m, initial_peak_T = measure_baseline(cfg)
    lam_target_m = float(baseline_lambda_m)

    # Build the lumopt optimization.
    fom = porttransmission(
        monitor_port="fom",
        mode_number=1,
        direction="Forward",
        target_T_fwd=lambda wl: np.ones_like(np.asarray(wl).flatten()),
        target_T_fwd_weights=_gaussian_weight_fn(lam_target_m, spec.fom_weight_sigma_nm),
        norm_p=1,
        target_fom=0.0,
    )
    geometry = make_lumopt_geometry(cfg, spec, p0)
    optimizer = ScipyOptimizers(
        max_iter=int(spec.max_iter),
        method=spec.optimizer_method,
        pgtol=spec.optimizer_pgtol,
        ftol=spec.optimizer_ftol,
    )
    half_w_m = 0.5 * spec.fom_window_nm * 1e-9
    fom_wavelengths = np.linspace(
        lam_target_m - half_w_m,
        lam_target_m + half_w_m,
        int(spec.fom_n_points),
    )
    base_script = make_base_script(
        cfg,
        spec.n_free_inner_teeth,
        lam_target_m,
        fom_window_nm=spec.fom_window_nm,
        fom_n_points=spec.fom_n_points,
        mesh_override_dxyz_nm=spec.mesh_override_dxyz_nm,
    )
    opt = Optimization(
        base_script=base_script,
        wavelengths=fom_wavelengths,
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

    # ── Diagnostic modes (no optimization) ────────────────────────────────
    if spec.mode == "check_gradient":
        print(f"[inverse_design] check_gradient mode: dx={spec.check_gradient_dx_nm} nm "
              f"at p0={p0}")
        p0_arr = np.array(p0, dtype=float)
        fd_grad, adj_grad, vec_err = opt.check_gradient(
            test_params=p0_arr,
            dx=float(spec.check_gradient_dx_nm),
            working_dir=out_dir,
        )
        print(f"[check_gradient] vec_error = {vec_err:.4f}")
        summary = {
            "mode": "check_gradient",
            "p0": list(map(float, p0)),
            "dx_nm": float(spec.check_gradient_dx_nm),
            "fd_grad": fd_grad.tolist(),
            "adj_grad": adj_grad.tolist(),
            "vec_error": float(vec_err),
            "baseline_lambda_m": float(baseline_lambda_m),
            "initial_peak_T": (None if initial_peak_T is None else float(initial_peak_T)),
        }
        with open(os.path.join(out_dir, "check_gradient.json"), "w") as fp:
            json.dump(summary, fp, indent=2)
        return summary

    if spec.mode == "scan_landscape":
        print(f"[inverse_design] scan_landscape mode: scanning cavity_width "
              f"across {spec.cavity_width_bounds_nm} at {spec.scan_n_points} points")
        opt.initialize(out_dir)
        cw_lo, cw_hi = spec.cavity_width_bounds_nm
        cw_values = np.linspace(cw_lo, cw_hi, int(spec.scan_n_points))
        fom_values: List[float] = []
        for cw in cw_values:
            p_test = list(p0)
            p_test[-1] = float(cw)
            fom_here = float(opt.callable_fom(np.array(p_test, dtype=float)))
            print(f"[scan] cavity_width = {cw:7.2f} nm  →  FOM = {fom_here:.6e}")
            fom_values.append(fom_here)
        summary = {
            "mode": "scan_landscape",
            "p0": list(map(float, p0)),
            "scanned_param": "cavity_width_nm",
            "scan_values_nm": cw_values.tolist(),
            "fom_values": fom_values,
            "baseline_lambda_m": float(baseline_lambda_m),
            "initial_peak_T": (None if initial_peak_T is None else float(initial_peak_T)),
        }
        with open(os.path.join(out_dir, "scan_landscape.json"), "w") as fp:
            json.dump(summary, fp, indent=2)
        return summary

    # ── Default: optimize ────────────────────────────────────────────────
    print(f"[inverse_design] running L-BFGS-B (max_iter={spec.max_iter}) "
          f"at λ_target={lam_target_m*1e9:.4f} nm, p0={p0}")
    fom_final, params_final = opt.run(working_dir=out_dir)
    p_final = list(map(float, params_final))
    print(f"[inverse_design] optimizer done. fom={fom_final:.4f} | "
          f"p={['%.2f' % v for v in p_final]}")

    # Post-opt verification at higher mesh accuracy.
    print(f"[inverse_design] post-opt verification at converged params (accurate mesh) ...")
    cfg_final = copy.deepcopy(cfg)
    cfg_final.mesh.simulation_mode = "accurate"
    fk = params_to_kwargs(p_final, spec.n_free_inner_teeth)
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
        "p_final": p_final,
        "fom_final": float(fom_final),
        "baseline_lambda_m": float(baseline_lambda_m),
        "initial_peak_T":     (None if initial_peak_T is None else float(initial_peak_T)),
        "true_peak_lambda_m": float(lam_peak),
        "true_peak_T":        float(T_peak),
        "delta_peak_T":       (None if initial_peak_T is None else float(T_peak - initial_peak_T)),
        "n_free_inner_teeth": spec.n_free_inner_teeth,
        "max_iter":           int(spec.max_iter),
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


def run_inverse_design_outer_loop(
    cfg: SimulationConfig,
    spec: InverseDesignSpec,
    start_idx: int = 0,
    output_root: Optional[str] = None,
    baseline_lambda_m: Optional[float] = None,
) -> dict:
    """Active-set outer loop: alternate inner adjoint optimization with
    re-centering of the FOM wavelength on the resonance.

    Why this exists: lumopt's adjoint pipeline silently relies on
    `frequency_dependent_profile=1` on the fom port to produce per-wavelength
    correct mode shapes. GPU FDTD does NOT support that setting, and our
    inverse_design path explicitly disables it (line 583-584). The result:
    the broadband-FOM adjoint computes E_adj at the wrong spatial shape for
    every λ ≠ λ_center, contaminating the gradient (vec_error 11.40 at
    fom_n_points=201).

    Workaround: at fom_n_points=1, single-frequency adjoint is mathematically
    consistent regardless of frequency_dependent_profile, so GPU stays valid.
    But single-λ FOM has no drift handling (resonance moves out of band as
    parameters change). The outer loop adds drift handling: every K=max_iter
    inner iters, re-measure the resonance via run_single_sim (broadband, GPU),
    update the FOM wavelength, restart the inner adjoint optimization from
    the current best params.

    Cost: K × n_outer_iters total inner iters; plus n_outer_iters baseline
    re-measurements (~3 min each on n_periods=80 GPU).
    """
    if spec.n_outer_iters <= 1:
        raise ValueError(
            "run_inverse_design_outer_loop requires spec.n_outer_iters > 1; "
            "for n_outer_iters=1 use the standard run_inverse_design directly."
        )

    n_outer = int(spec.n_outer_iters)
    base_inner_iters = int(spec.max_iter)

    # Get the starting params.
    starts = spec.get_starts()
    if not (0 <= start_idx < len(starts)):
        raise IndexError(f"start_idx {start_idx} out of range (have {len(starts)} starts).")
    p_curr = list(starts[start_idx])

    # Set up output dir.
    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "inverse_design", spec.label or "study")
    out_dir_outer = os.path.join(output_root, f"start{start_idx}")
    os.makedirs(out_dir_outer, exist_ok=True)

    outer_history: List[dict] = []
    lam_target_m = baseline_lambda_m

    for outer_i in range(n_outer):
        print(f"\n{'='*60}")
        print(f"[outer_loop] outer iter {outer_i + 1}/{n_outer}")
        print(f"[outer_loop] starting params: {p_curr}")

        # 1. Re-measure resonance at current params (skip first iter if
        #    baseline_lambda_m was pre-supplied; otherwise always measure).
        cfg_curr = copy.deepcopy(cfg)
        cfg_curr.grating.n_free_inner_teeth = spec.n_free_inner_teeth
        cfg_curr.grating.lengthen_cavity = spec.lengthen_cavity
        kw = params_to_kwargs(p_curr, spec.n_free_inner_teeth)
        cfg_curr.grating.inner_dw_nm = list(kw["inner_dw_nm"])
        cfg_curr.grating.inner_shift_nm = list(kw["inner_shift_nm"])
        cfg_curr.grating.cavity_width_m = kw["cavity_width_m"]

        if outer_i > 0 or lam_target_m is None:
            print(f"[outer_loop] re-measuring resonance at current geometry ...")
            lam_target_m, peak_T_at_p = measure_baseline(cfg_curr)
            print(f"[outer_loop] λ_target = {lam_target_m * 1e9:.3f} nm  (T = {peak_T_at_p:.4f})")

        # 2. Build inner spec: single-λ FOM at current λ_target, p_curr as start.
        inner_spec = copy.deepcopy(spec)
        inner_spec.fom_n_points = 1                 # ← key: single λ → GPU adjoint correct
        inner_spec.fom_window_nm = 0.0              # 0 width is OK for single point
        inner_spec.max_iter = base_inner_iters
        inner_spec.n_outer_iters = 1                # don't recurse
        inner_spec.initial_points = [list(p_curr)]
        inner_spec.n_starts = 1
        inner_spec.label = f"{spec.label or 'outer'}_inner{outer_i}"

        # 3. Run inner adjoint optimization.
        inner_out_dir = os.path.join(out_dir_outer, f"inner{outer_i}")
        try:
            result = run_inverse_design(
                cfg=cfg, spec=inner_spec, start_idx=0,
                output_root=inner_out_dir,
                baseline_lambda_m=lam_target_m,
            )
            # Defensive float coercion — result fields may be numpy scalars
            # (np.float64) or 1-element arrays from various code paths.
            def _to_py_float(x, default=float("nan")):
                try:
                    return float(np.asarray(x).item()) if x is not None else default
                except (TypeError, ValueError):
                    return default

            def _to_py_list(x):
                try:
                    return list(map(float, np.asarray(x).flatten()))
                except (TypeError, ValueError):
                    return list(x) if x is not None else []

            p_curr = _to_py_list(result.get("p_final", p_curr))
            true_peak_T = _to_py_float(result.get("true_peak_T"))
            true_peak_lambda_m = _to_py_float(result.get("true_peak_lambda_m"))
            fom_final_inner = _to_py_float(result.get("fom_final"))

            outer_history.append({
                "outer_iter": outer_i,
                "lam_target_nm": float(lam_target_m * 1e9),
                "p_start": _to_py_list(result.get("p_initial", [])),
                "p_final": list(p_curr),
                "fom_final": fom_final_inner,
                "true_peak_T": true_peak_T,
                "true_peak_lambda_nm": float(true_peak_lambda_m * 1e9) if not np.isnan(true_peak_lambda_m) else None,
            })
            print(f"[outer_loop] iter {outer_i + 1} done: peak_T = "
                  f"{true_peak_T:.4f}  →  p = {[round(v, 2) for v in p_curr]}")

            # Update lam_target for next outer iter using the actual measured
            # resonance from this iter's verification (better than re-measuring).
            if not np.isnan(true_peak_lambda_m) and true_peak_lambda_m > 0:
                lam_target_m = true_peak_lambda_m
        except Exception as exc:
            import traceback
            print(f"[outer_loop] iter {outer_i + 1} FAILED: {exc}")
            traceback.print_exc()
            outer_history.append({
                "outer_iter": outer_i,
                "lam_target_nm": (float(lam_target_m * 1e9) if lam_target_m else None),
                "error": str(exc),
            })
            break

    # Final summary.
    summary = {
        "label": spec.label,
        "start_idx": int(start_idx),
        "n_outer_iters": n_outer,
        "max_inner_iter": base_inner_iters,
        "p_initial": list(map(float, starts[start_idx])),
        "p_final": list(map(float, p_curr)),
        "outer_history": outer_history,
        "final_lam_target_nm": (lam_target_m * 1e9) if lam_target_m else None,
    }
    with open(os.path.join(out_dir_outer, "outer_loop_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[outer_loop] DONE. {n_outer} outer iters × {base_inner_iters} inner iters.")
    print(f"  p_final = {p_curr}")
    if outer_history:
        last = outer_history[-1]
        if "true_peak_T" in last:
            print(f"  final peak T = {last.get('true_peak_T'):.4f}")
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
