"""
Geometry callback for lumopt's FunctionDefinedPolygon.

The freed (optimization) region spans the n_free innermost teeth on each side
of the cavity plus the cavity itself. Its outer boundaries are at fixed
x positions independent of the parameter vector p (the cavity absorbs the
total of all gap shifts). This means we can put the static outer teeth and
access waveguides in a lumopt base_script and let the optimizer mutate only
the freed-region polygon.

Parameter vector layout (see `spec.py`):
    p = [dw_1..dw_N, shift_1..shift_N, cavity_width_nm]    length = 2N+1

Polygon convention:
- Single closed polygon traced as top-edge (left→right) then bottom-edge
  (right→left). Mirror-symmetric about y=0.
- z = 0 (core center), depth = core_height (3D slab extrusion).
- eps_in = core (SiN), eps_out = cladding (SiO2). lumopt assigns these.

Segment enumeration in the freed region (left → right):
    0:           narrow_N_L    (W_narrow[N], length = pitch/2 - shift_N)
    1:           wide_N_L      (W_wide[N],   length = pitch/2)
    ...
    2N-2:        narrow_1_L    (W_narrow[1], length = pitch/2 - shift_1)
    2N-1:        wide_1_L      (W_wide[1],   length = pitch/2)
    2N:          cavity        (W_cavity,    length = cavity_baseline + 2*Σ shifts)
    2N+1:        narrow_1_R    (W_narrow[1], length = pitch/2)              [full]
    2N+2:        wide_1_R      (W_wide[1],   length = pitch/2)
    2N+3:        narrow_2_R    (W_narrow[2], length = pitch/2 - shift_1)
    2N+4:        wide_2_R      (W_wide[2],   length = pitch/2)
    ...
    4N+1:        narrow_{N+1}_R (W_narrow[N+1] from envelope, length = pitch/2 - shift_N)
    4N+2:        wide_{N+1}_R   (W_wide[N+1] from envelope,   length = pitch/2)

Total: 4N+3 segments.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation_config import SimulationConfig


# ── Mod-depth envelope (mirrors bragg_device.get_mod_depth) ──────────────────

def _envelope_mod_depth(d: int, cfg: SimulationConfig) -> float:
    """Apodization envelope value for tooth d (in meters). Returns full
    corrugation depth (W_wide - W_narrow). Mirrors bragg_device._add_bragg_core's
    get_mod_depth() for d > n_free_inner_teeth."""
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


# ── Freed-region geometry (parameter-dependent) ──────────────────────────────

def freed_region_x_bounds(cfg: SimulationConfig, n_free: int) -> Tuple[float, float]:
    """Fixed x-extent of the freed region (independent of the parameter vector).

    Returns (x_left_freed_start, x_right_freed_end) in meters. The freed
    region spans n_free freed teeth on the left, the cavity, n_free freed
    teeth on the right, and one additional outer-shifted tooth (d=n_free+1)
    on the right whose narrow gap absorbs shift_N.
    """
    n_total = cfg.grating.n_periods_each_side
    pitch = cfg.grating.pitch_m
    # Cavity length without extras (extras are absorbed via cavity_extra)
    if cfg.grating.cavity_neg_detuning_nm != 0.0:
        cavity_length_baseline = (pitch * 0.5) - cfg.grating.cavity_neg_detuning_nm * 1e-9
    else:
        cavity_length_baseline = pitch * 0.5
    x_grating_end = n_total * pitch + cavity_length_baseline / 2.0
    x_left_freed_start = -x_grating_end + (n_total - n_free) * pitch
    # Right boundary: end of wide_{n_free+1}_R = -x_grating_end + (n_total + n_free + 1) * pitch + cavity_length_baseline
    x_right_freed_end = -x_grating_end + (n_total + n_free + 1) * pitch + cavity_length_baseline
    return x_left_freed_start, x_right_freed_end


def freed_region_segments(cfg: SimulationConfig, p) -> List[Tuple[float, float, float]]:
    """Enumerate the freed-region segments (x_start, x_end, width_m) in left-to-right order.

    p is the optimization vector: [dw_1..dw_N, shift_1..shift_N, cavity_width_nm].
    """
    from runners.inverse_design.spec import params_to_kwargs

    n_free = cfg.grating.n_free_inner_teeth
    kw = params_to_kwargs(p, n_free)
    inner_dw_nm = kw["inner_dw_nm"]
    inner_shift_nm = kw["inner_shift_nm"]
    cavity_width_m = kw["cavity_width_m"]

    pitch = cfg.grating.pitch_m
    half_pitch = pitch * 0.5
    avg_w = cfg.geometry.avg_corrugation_width_m

    def tooth_widths(d: int) -> Tuple[float, float]:
        """(W_narrow, W_wide) for tooth d (1-indexed)."""
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
        segs.append((x, x + nlen, Wn));  x += nlen
        segs.append((x, x + half_pitch, Ww));  x += half_pitch

    # Cavity
    segs.append((x, x + cavity_length_eff, cavity_width_m));  x += cavity_length_eff

    # Right freed teeth: d = 1, ..., n_free + 1
    # narrow_d_R length = half_pitch - shift_{d-1} for d >= 2 (full for d=1)
    for d in range(1, n_free + 2):
        Wn, Ww = tooth_widths(d)
        s_prev = shift_m[d - 2] if d >= 2 else 0.0
        nlen = half_pitch - s_prev
        # cavity_width_option="avg_ext" widens R_narrow_1 to avg_w (legacy)
        if d == 1 and cfg.grating.cavity_width_option == "avg_ext":
            Wn = avg_w
        segs.append((x, x + nlen, Wn));  x += nlen
        segs.append((x, x + half_pitch, Ww));  x += half_pitch

    return segs


def polygon_vertices(cfg: SimulationConfig, p) -> np.ndarray:
    """Closed polygon vertex array (CCW) for the freed region.

    Top edge traced left → right, bottom edge right → left. Returns shape (V, 2).
    """
    segs = freed_region_segments(cfg, p)
    top: List[Tuple[float, float]] = []
    for (x0, x1, w) in segs:
        y_top = +0.5 * w
        if not top or top[-1][1] != y_top:
            # vertical step at x = x0 (last point already there)
            top.append((x0, y_top))
        top.append((x1, y_top))
    bot: List[Tuple[float, float]] = []
    for (x0, x1, w) in reversed(segs):
        y_bot = -0.5 * w
        if not bot or bot[-1][1] != y_bot:
            bot.append((x1, y_bot))
        bot.append((x0, y_bot))
    verts = np.array(top + bot, dtype=float)
    return verts


# ── lumopt integration ───────────────────────────────────────────────────────

def make_polygon_callback(cfg: SimulationConfig):
    """Return a closure suitable for lumopt's FunctionDefinedPolygon `func` arg.

    The closure captures cfg; lumopt will call it as `func(p) -> (V, 2)`.
    """
    def _callback(p):
        return polygon_vertices(cfg, p)
    return _callback


def make_lumopt_geometry(cfg: SimulationConfig, spec, initial_p):
    """Construct a lumopt FunctionDefinedPolygon for the freed region.

    Lazy-imports lumopt so this module can be inspected/used without a Lumerical
    install. Caller is expected to have lumopt available at runtime.

    Args:
        cfg: SimulationConfig with grating.n_free_inner_teeth set.
        spec: InverseDesignSpec — supplies bounds.
        initial_p: starting parameter vector (length 2N+1).
    """
    from lumopt.geometries.polygon import FunctionDefinedPolygon  # type: ignore

    # eps from constant-material indices (the only adjoint-friendly path here)
    eps_in = cfg.material.n_core_const ** 2
    eps_out = cfg.material.n_clad_const ** 2

    bounds = spec.all_bounds_nm()
    geom = FunctionDefinedPolygon(
        func=make_polygon_callback(cfg),
        initial_params=np.array(initial_p, dtype=float),
        bounds=bounds,
        z=0.0,
        depth=cfg.geometry.core_height_m,
        eps_out=eps_out,
        eps_in=eps_in,
        edge_precision=5,
        dx=1.0,   # FD step for jacobian, in nm (parameter units)
    )
    return geom
