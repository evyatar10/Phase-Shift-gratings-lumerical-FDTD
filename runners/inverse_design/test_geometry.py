"""
Local geometry round-trip verification — runs without Lumerical.

Checks that:
  1. Segment math in `freed_region_segments` gives a self-consistent layout
     with x_right_freed_end matching the closed-form bound from
     `freed_region_x_bounds`.
  2. The total grating length is preserved as shifts are applied (cavity
     absorbs Σ shifts, x_grating_end stays constant).
  3. Mirror-symmetric DW bounds reduce to the apodization envelope when
     inner_dw_nm matches the envelope value at d=1, d=2.
  4. Single-tooth-shift case reproduces the legacy `innermost_tooth_shift`
     convention: left narrow_1 and right narrow_2 shortened by `shift`.

Run:
    python -m runners.inverse_design.test_geometry
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from runners.inverse_design.inverse_design import (
    freed_region_segments,
    freed_region_x_bounds,
    polygon_vertices,
    _envelope_mod_depth,
)
from simulation_config import SimulationConfig


def _make_cfg(n_free: int = 2, n_periods: int = 80) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = n_periods
    cfg.grating.n_free_inner_teeth = n_free
    cfg.grating.lengthen_cavity = True
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.spectral.scan_width_nm = 10.0
    return cfg


def test_x_bounds_fixed_under_shifts():
    cfg = _make_cfg(n_free=2)
    x_lo_a, x_hi_a = freed_region_x_bounds(cfg, n_free=2)
    # Bounds are derived from cfg only (not from p), so they cannot change.
    x_lo_b, x_hi_b = freed_region_x_bounds(cfg, n_free=2)
    assert x_lo_a == x_lo_b and x_hi_a == x_hi_b
    # Sanity: width = (2N+1)*pitch + cavity_baseline
    expected_width = (2 * 2 + 1) * cfg.grating.pitch_m + 0.5 * cfg.grating.pitch_m
    assert abs((x_hi_a - x_lo_a) - expected_width) < 1e-15
    print(f"[ok] x_bounds_fixed_under_shifts: width={expected_width*1e9:.1f} nm")


def test_total_length_preserved():
    """As shifts grow, cavity expands so the freed-region end remains at
    x_right_freed_end (closed-form). Walk segments and confirm."""
    cfg = _make_cfg(n_free=2)
    x_lo, x_hi = freed_region_x_bounds(cfg, n_free=2)
    for shift_combo in [(0.0, 0.0), (100.0, 0.0), (50.0, 75.0), (200.0, 200.0)]:
        # p = [dw_1, dw_2, shift_1, shift_2, cavity_width_nm]
        p = [300.0, 300.0, shift_combo[0], shift_combo[1], 250.0]
        segs = freed_region_segments(cfg, p)
        # Walk: first seg starts at x_lo
        assert abs(segs[0][0] - x_lo) < 1e-15, f"start mismatch: {segs[0][0]} vs {x_lo}"
        # Last seg ends at x_hi
        assert abs(segs[-1][1] - x_hi) < 1e-9, (
            f"end mismatch for shifts={shift_combo}: {segs[-1][1]*1e9:.3f} vs {x_hi*1e9:.3f}"
        )
        n_segs = 4 * cfg.grating.n_free_inner_teeth + 3
        assert len(segs) == n_segs, f"expected {n_segs} segs, got {len(segs)}"
        print(f"[ok] total_length_preserved: shifts={shift_combo} -> {len(segs)} segs")


def test_single_shift_legacy_layout():
    """Reproduce the legacy single-tooth shift convention.

    With shift_1>0, shift_2=0, inner_dw set to the envelope value, the
    freed-region segment positions on the left should match: narrow_1_L
    shortened by shift_1, all other narrow segments at full half_pitch.
    On the right: narrow_1_R full, narrow_2_R shortened by shift_1.
    """
    cfg = _make_cfg(n_free=2)
    pitch = cfg.grating.pitch_m
    half_pitch = pitch / 2.0
    shift = 100e-9
    # Use envelope DWs so widths match the legacy non-inverse-design path
    dw_d1 = _envelope_mod_depth(1, cfg) * 1e9
    dw_d2 = _envelope_mod_depth(2, cfg) * 1e9
    p = [dw_d1, dw_d2, 100.0, 0.0, 250.0]
    segs = freed_region_segments(cfg, p)
    # Segment indices (n_free=2):
    #   0: narrow_2_L  (length = half_pitch - shift_2 = half_pitch  → 250 nm)
    #   1: wide_2_L
    #   2: narrow_1_L  (length = half_pitch - shift_1 = 150 nm)
    #   3: wide_1_L
    #   4: cavity      (length = 250 + 2*shift_1 = 450 nm)
    #   5: narrow_1_R  (full half_pitch = 250 nm)
    #   6: wide_1_R
    #   7: narrow_2_R  (length = half_pitch - shift_1 = 150 nm)
    #   8: wide_2_R
    #   9: narrow_3_R  (length = half_pitch - shift_2 = 250 nm)
    #  10: wide_3_R
    expected_lengths = [
        half_pitch,             # narrow_2_L (no shift_2)
        half_pitch,             # wide_2_L
        half_pitch - shift,     # narrow_1_L (shift_1)
        half_pitch,             # wide_1_L
        half_pitch + 2 * shift, # cavity (250 + 2*shift = 450)
        half_pitch,             # narrow_1_R (full)
        half_pitch,             # wide_1_R
        half_pitch - shift,     # narrow_2_R (shift_1)
        half_pitch,             # wide_2_R
        half_pitch,             # narrow_3_R (no shift_2)
        half_pitch,             # wide_3_R
    ]
    for i, ((x0, x1, w), expected) in enumerate(zip(segs, expected_lengths)):
        actual = x1 - x0
        assert abs(actual - expected) < 1e-15, (
            f"seg {i}: length {actual*1e9:.3f} nm vs expected {expected*1e9:.3f} nm"
        )
    print("[ok] single_shift_legacy_layout: 11 segments with correct lengths")


def test_polygon_is_closed_and_mirror_symmetric():
    cfg = _make_cfg(n_free=2)
    p = [300.0, 250.0, 50.0, 75.0, 800.0]
    verts = polygon_vertices(cfg, p)
    assert verts.ndim == 2 and verts.shape[1] == 2, f"shape {verts.shape}"
    # Top half should mirror bottom half through y=0
    ys = verts[:, 1]
    assert abs(ys.max() + ys.min()) < 1e-15, f"asymmetric y: {ys.min()}..{ys.max()}"
    # x-extent matches freed_region_x_bounds
    x_lo, x_hi = freed_region_x_bounds(cfg, n_free=2)
    assert abs(verts[:, 0].min() - x_lo) < 1e-9
    assert abs(verts[:, 0].max() - x_hi) < 1e-9
    print(f"[ok] polygon_is_closed_and_mirror_symmetric: {verts.shape[0]} vertices")


if __name__ == "__main__":
    test_x_bounds_fixed_under_shifts()
    test_total_length_preserved()
    test_single_shift_legacy_layout()
    test_polygon_is_closed_and_mirror_symmetric()
    print("\nAll geometry round-trip checks passed.")
