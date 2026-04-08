"""
Mesh convergence testing — Pi-Shift Bragg Grating FDTD.

Tests two mesh parameters sequentially (coordinate descent).  The convergence
metric and threshold are configurable (see user-editable settings below):
  "Q"      — Q factor (lambda/FWHM); threshold 2.0% (typical published FDTD standard)
  "lambda" — resonance wavelength;   threshold 1.0% (Ansys/Lumerical standard)

Sweep order:
  Phase A — cells_per_half_period  [4, 5, 6, 7, 8, 10]  controls dx
  Phase B — dz_divisor             [5, 7, 9, 12, 15]     controls dz = core_height / divisor
  (dy_divisor not swept — dy=50 nm / 13 cells across narrow width is well-resolved)

Each phase fixes the best-converged value from the previous phase.
Early stopping: if |Δmetric/metric| < threshold for two consecutive mesh values,
the phase stops and recommends the coarser of those two.

Results are checkpointed to a JSON file named checkpoint_p{N_PERIODS}_{METRIC}.json
after every run.  Different (periods, metric) combinations write separate files and
never overwrite each other.

Run from the project root:
    python convergence_testing/run_mesh_convergence.py
"""

import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bragg_device import PiShiftBraggFDTD
from post_processing import extract_s_parameters, find_resonance
from sim_helpers import apply_monitor_overrides
from simulation_config import SimulationConfig

# ═════════════════════════════════════════════════════════════════════════════
# User-editable settings
# ═════════════════════════════════════════════════════════════════════════════

# ── Convergence metric ───────────────────────────────────────────────────────
# Set CONVERGENCE_METRIC to "Q" or "lambda".
# Each metric has its own threshold based on published FDTD standards:
#   lambda — 1.0%: Ansys/Lumerical stated standard for resonance wavelength
#   Q      — 2.0%: typical published FDTD standard for mesh convergence of Q.
#             Note: Lumerical's 0.2% criterion refers to auto-shutoff convergence
#             (field decay within a single run), NOT mesh-to-mesh convergence.
CONVERGENCE_METRIC   = "Q"      # "Q" or "lambda"
N_PERIODS_EACH_SIDE  = 80       # device length; also encoded in checkpoint filename

THRESHOLD_LAMBDA = 0.010        # 1.0% — for lambda convergence metric
THRESHOLD_Q      = 0.020        # 2.0% — for Q convergence metric

# Sweep values for each phase (coarse → fine).
# cells=3 removed from Phase A: too coarse, guaranteed non-converged, wastes a run.
# Phase C (dy_divisor) dropped: dy=50 nm is already 13 cells across the 650 nm
# narrow width — well-resolved and not a convergence concern.
PHASE_A_VALUES = [4, 5, 6, 7, 8, 10]        # cells_per_half_period (controls dx)
PHASE_B_VALUES = [5, 7, 9, 12, 15]          # dz_divisor            (controls dz)

# Default (current production) values — used as fixed values in other phases
DEFAULT_CELLS  = 5
DEFAULT_DZ_DIV = 7
DEFAULT_DY_DIV = 13

# ═════════════════════════════════════════════════════════════════════════════
# Output paths
# ═════════════════════════════════════════════════════════════════════════════

CONV_DIR     = os.path.join(config.BASE_SAVE_DIR, "mesh_convergence")
LAYOUTS_DIR  = os.path.join(CONV_DIR, "layouts")
RESULTS_DIR  = os.path.join(CONV_DIR, "results")
# Checkpoint filename encodes periods + metric so different run configurations
# never overwrite each other (e.g. checkpoint_p80_Q.json vs checkpoint_p60_lambda.json).
CHECKPOINT   = os.path.join(CONV_DIR,
    f"checkpoint_p{N_PERIODS_EACH_SIDE}_{CONVERGENCE_METRIC}.json")

os.makedirs(LAYOUTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Subclass: tunable mesh divisors, zero changes to bragg_device.py
# ═════════════════════════════════════════════════════════════════════════════

class _ConvergenceBraggFDTD(PiShiftBraggFDTD):
    """
    PiShiftBraggFDTD with cells_per_half_period, dy_divisor, dz_divisor exposed
    as constructor arguments.  Only _add_aligned_mesh_override is overridden;
    all other behaviour (geometry, source, monitors, S-parameters) is identical
    to the parent class.
    """

    def __init__(self, *args,
                 cells_per_half_period=DEFAULT_CELLS,
                 dy_divisor=DEFAULT_DY_DIV,
                 dz_divisor=DEFAULT_DZ_DIV,
                 **kwargs):
        self._cells  = int(cells_per_half_period)
        self._dy_div = float(dy_divisor)
        self._dz_div = float(dz_divisor)
        super().__init__(*args, **kwargs)

    def _add_aligned_mesh_override(self, cells_per_half_period=5):
        """
        3-box mesh override (left periodic, cavity, right periodic)
        with tunable dy/dz divisors and cells_per_half_period.
        """
        fdtd = self.fdtd
        half_pitch = 0.5 * self.pitch
        n_cells_half = max(1, self._cells)
        dx_grating = half_pitch / float(n_cells_half)
        dy = self.width_narrow / self._dy_div
        dz = self.core_height / self._dz_div

        # Cavity and grating boundaries
        x_grating_start = -self.x_grating_end
        x_grating_end   =  self.x_grating_end
        x_cav_left  = -self.cavity_length / 2.0
        x_cav_right =  self.cavity_length / 2.0

        # Cavity dx: snap to nearest integer cells
        n_cav = max(1, round(self.cavity_length / dx_grating))
        dx_cav = self.cavity_length / float(n_cav)
        if abs(dx_cav - dx_grating) / dx_grating > 0.05:
            print(f"  WARNING: cavity dx={dx_cav*1e9:.1f}nm deviates >5% from "
                  f"dx_grating={dx_grating*1e9:.1f}nm")

        # Y/Z extent: waveguide + evanescent margin (1 tail for optimization, 2 for accurate)
        _dn_sq = max(self.n_eff_guess**2 - self.n_clad_const**2, 0.01)
        _decay_len = self.lambda_B / (2.0 * math.pi * math.sqrt(_dn_sq))
        _n_tails = 2.0 if self.simulation_mode == "accurate" else 1.0
        y_span_override = self.width_wide  + 2.0 * _n_tails * _decay_len
        z_span_override = self.core_height + 2.0 * _n_tails * _decay_len

        def add_mesh_box(name, x_left, x_right, dx_val):
            span = x_right - x_left
            fdtd.addmesh()
            fdtd.set("name", name)
            fdtd.set("x", x_left + span / 2.0)
            fdtd.set("x span", span)
            fdtd.set("y", 0.0)
            fdtd.set("y span", y_span_override)
            fdtd.set("z", 0.0)
            fdtd.set("z span", z_span_override)
            fdtd.set("override x mesh", 1)
            fdtd.set("override y mesh", 1)
            fdtd.set("override z mesh", 1)
            fdtd.set("dx", dx_val)
            fdtd.set("dy", dy)
            fdtd.set("dz", dz)

        # Left periodic: exact dx_grating (span = n_periods*pitch, always divisible)
        add_mesh_box("mesh_left_periodic", x_grating_start, x_cav_left, dx_grating)
        # Cavity: snap_dx to fit cavity_length exactly
        add_mesh_box("mesh_cavity", x_cav_left, x_cav_right, dx_cav)
        # Right periodic: exact dx_grating
        add_mesh_box("mesh_right_periodic", x_cav_right, x_grating_end, dx_grating)


# ═════════════════════════════════════════════════════════════════════════════
# Simulation config
# ═════════════════════════════════════════════════════════════════════════════

def _make_cfg() -> SimulationConfig:
    """SimulationConfig tuned for convergence testing: no 2D/3D/far-field monitors."""
    cfg = SimulationConfig()
    cfg.apodization.enabled        = False
    cfg.monitors.record_2d_fields  = False
    cfg.monitors.record_3d_fields  = False
    cfg.farfield.enabled           = False
    cfg.run.cleanup_lumerical_data = True
    # Fewer periods: shorter device → faster sim + faster auto-shutoff (lower Q
    # decays quicker). 40 periods is enough to see a clear resonance and get a
    # stable Q ratio for convergence comparison purposes.
    cfg.grating.n_periods_each_side = N_PERIODS_EACH_SIDE
    # Narrow scan: 20 nm is sufficient to capture the cavity resonance peak
    # and avoids wasting DFT points outside the stopband.
    cfg.spectral.scan_width_nm = 20.0
    # Fewer DFT monitor points: 3001 is overkill for just locating the resonance.
    cfg.spectral.n_wl_points = 1001
    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {}


def _save_checkpoint(data: dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# Single simulation runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_one(cfg: SimulationConfig,
             cells: int, dy_div: float, dz_div: float,
             tag: str) -> dict:
    """
    Build and run one FDTD simulation, extract Q factor from S-parameters.

    Parameters
    ----------
    cfg     : SimulationConfig (monitors already disabled)
    cells   : cells_per_half_period
    dy_div  : dy = width_narrow / dy_div
    dz_div  : dz = core_height  / dz_div
    tag     : short string used in layout / results filenames

    Returns
    -------
    dict with keys: cells, dy_div, dz_div, dx_nm, dy_nm, dz_nm,
                    Q, resonance_nm, fwhm_pm, sim_time_s
    """
    half_pitch = cfg.grating.pitch_m / 2.0
    dx_nm = half_pitch / cells * 1e9
    dy_nm = cfg.geometry.width_narrow_m / dy_div * 1e9
    dz_nm = cfg.geometry.core_height_m  / dz_div * 1e9

    print(f"\n{'─' * 62}")
    print(f"  [{tag}]")
    print(f"  cells={cells}  dy_div={dy_div}  dz_div={dz_div}")
    print(f"  dx={dx_nm:.1f} nm   dy={dy_nm:.1f} nm   dz={dz_nm:.1f} nm")
    print(f"{'─' * 62}")

    device_kwargs = cfg.to_device_kwargs()
    sim = _ConvergenceBraggFDTD(
        **device_kwargs,
        cells_per_half_period=cells,
        dy_divisor=dy_div,
        dz_divisor=dz_div,
    )

    layout_path  = os.path.join(LAYOUTS_DIR, f"layout_{tag}.fsp")

    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )
    apply_monitor_overrides(sim, cfg)
    sim.fdtd.save(layout_path)

    t0 = time.perf_counter()
    sim.fdtd.run()
    elapsed = time.perf_counter() - t0
    print(f"  Simulation done in {elapsed:.1f} s")

    try:
        s_params  = extract_s_parameters(sim, cfg)
        resonance = find_resonance(s_params)
        # find_resonance computes FWHM as peak_width_samples * dw, where
        # dw = wl[1] - wl[0].  Lumerical returns wavelengths in descending
        # order (sampled uniformly in frequency), so dw < 0 and FWHM comes
        # out negative.  abs() corrects the sign without touching the
        # original post_processing.py.
        fwhm_m = abs(resonance.spectral_fwhm_m)
        Q = resonance.wavelength_m / fwhm_m
        print(f"  lambda_res = {resonance.wavelength_m * 1e9:.4f} nm  "
              f"FWHM = {fwhm_m * 1e12:.2f} pm  "
              f"Q = {Q:.2f}  "
              f"T_res = {resonance.transmission:.4f}")
    finally:
        plt.close("all")
        sim.close()

    return {
        "cells":        cells,
        "dy_div":       dy_div,
        "dz_div":       dz_div,
        "dx_nm":        round(dx_nm,  2),
        "dy_nm":        round(dy_nm,  2),
        "dz_nm":        round(dz_nm,  2),
        "Q":            Q,
        "resonance_nm": resonance.wavelength_m * 1e9,
        "fwhm_pm":      fwhm_m * 1e12,
        "transmission": float(resonance.transmission),
        "sim_time_s":   round(elapsed, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# One sweep phase
# ═════════════════════════════════════════════════════════════════════════════

def _run_phase(phase: str,
               param: str,
               values: list,
               fixed_cells: int,
               fixed_dy_div: float,
               fixed_dz_div: float,
               cfg: SimulationConfig,
               checkpoint: dict,
               metric: str,
               threshold: float) -> tuple:
    """
    Sweep one parameter while fixing the others.

    metric    : "Q" or "lambda" — which quantity drives convergence checking.
    threshold : fractional threshold (e.g. 0.02 = 2%) for the chosen metric.

    Returns (results_list, best_value) where best_value is the coarser of
    the two values that first showed |Delta_metric/metric| < threshold for
    two consecutive steps.  Falls back to the finest value if no convergence.
    """
    print(f"\n{'=' * 62}")
    print(f"  Phase {phase}: sweeping {param}  [metric={metric}, threshold={threshold*100:.1f}%]")
    print(f"  Fixed: cells={fixed_cells}, dy_div={fixed_dy_div}, dz_div={fixed_dz_div}")
    print(f"{'=' * 62}")

    results         = []
    prev_wl         = None
    prev_Q          = None
    converged_count = 0
    best_value      = values[-1]   # fallback: finest mesh

    for i, val in enumerate(values):
        # Build (cells, dy_div, dz_div) for this step
        cells  = int(val)   if param == "cells_per_half_period" else fixed_cells
        dz_div = float(val) if param == "dz_divisor"           else fixed_dz_div
        dy_div = float(val) if param == "dy_divisor"           else fixed_dy_div

        ck_key = f"ph{phase}_{val}"

        if ck_key in checkpoint:
            print(f"\n  [SKIP] {param}={val} — loaded from checkpoint")
            rec = checkpoint[ck_key]
        else:
            tag = f"ph{phase}_{param.replace('_', '')}_{val}"
            rec = _run_one(cfg, cells, dy_div, dz_div, tag)
            checkpoint[ck_key] = rec
            _save_checkpoint(checkpoint)

        results.append(rec)
        wl = rec["resonance_nm"]
        Q  = rec["Q"]

        # Convergence check on the selected metric
        if prev_wl is not None:
            if metric == "Q":
                delta = abs(Q - prev_Q) / max(abs(prev_Q), 1e-12)
                label = f"|ΔQ/Q| = {delta * 100:.3f}%"
            else:
                delta = abs(wl - prev_wl) / max(abs(prev_wl), 1e-12)
                label = f"|Δλ/λ| = {delta * 100:.3f}%"

            if delta < threshold:
                converged_count += 1
                if converged_count >= 2:
                    best_value = values[i - 1]   # coarser value, already stable
                    print(f"\n  Converged at {param}={val}  ({label})")
                    print(f"  Using {param} = {best_value}  (coarser, already stable)")
                    break
            else:
                converged_count = 0

        prev_wl = wl
        prev_Q  = Q
    else:
        print(f"\n  No convergence within sweep range. "
              f"Using finest value: {param} = {best_value}")

    return results, best_value


# ═════════════════════════════════════════════════════════════════════════════
# Results table
# ═════════════════════════════════════════════════════════════════════════════

def _print_table(phase: str, param_label: str, results: list) -> None:
    col = 12
    sep = "─" * 118
    header = (f"  {'Param val':<{col}} {'dx(nm)':<8} {'dy(nm)':<8} {'dz(nm)':<8}"
              f" {'lambda(nm)':>12}  {'FWHM(pm)':>10}  {'Q':>10}"
              f"  {'T_res':>7}  {'Δλ%':>10}  {'ΔQ%':>10}  {'Time(s)':>8}")
    print(f"\n{sep}")
    print(f"  Phase {phase} — {param_label}")
    print(sep)
    print(header)
    print(sep)

    prev_wl = None
    prev_Q  = None
    for r in results:
        val = r.get("cells") if param_label == "cells_per_half_period" else \
              r.get("dz_div") if param_label == "dz_divisor" else \
              r.get("dy_div")
        dlam_str = "         —"
        dQ_str   = "         —"
        if prev_wl is not None:
            dlam = (r["resonance_nm"] - prev_wl) / abs(prev_wl) * 100
            dlam_str = f"{dlam:+.4f}%"
        if prev_Q is not None:
            dQ = (r["Q"] - prev_Q) / abs(prev_Q) * 100
            dQ_str = f"{dQ:+.4f}%"
        t_res = r.get("transmission", float("nan"))
        print(f"  {str(val):<{col}} {r['dx_nm']:<8.1f} {r['dy_nm']:<8.1f} "
              f"{r['dz_nm']:<8.1f} {r['resonance_nm']:>12.4f}  {r['fwhm_pm']:>10.2f}"
              f"  {r['Q']:>10.2f}  {t_res:>7.4f}  {dlam_str:>10}  {dQ_str:>10}"
              f"  {r['sim_time_s']:>8.1f}")
        prev_wl = r["resonance_nm"]
        prev_Q  = r["Q"]
    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg        = _make_cfg()
    checkpoint = _load_checkpoint()

    half_pitch = cfg.grating.pitch_m / 2.0
    threshold  = THRESHOLD_Q if CONVERGENCE_METRIC == "Q" else THRESHOLD_LAMBDA

    print("\n" + "=" * 62)
    print("  MESH CONVERGENCE")
    print(f"  Convergence metric    : {CONVERGENCE_METRIC}  "
          f"(threshold {threshold * 100:.1f}%)")
    print(f"  Periods each side     : {cfg.grating.n_periods_each_side}")
    print(f"  Current defaults      : cells={DEFAULT_CELLS} "
          f"(dx={half_pitch / DEFAULT_CELLS * 1e9:.0f} nm), "
          f"dz_div={DEFAULT_DZ_DIV} "
          f"(dz={cfg.geometry.core_height_m / DEFAULT_DZ_DIV * 1e9:.0f} nm), "
          f"dy_div={DEFAULT_DY_DIV} "
          f"(dy={cfg.geometry.width_narrow_m / DEFAULT_DY_DIV * 1e9:.0f} nm)")
    print(f"  Checkpoint            : {CHECKPOINT}")
    print("=" * 62)

    # ── Phase A: cells_per_half_period (controls dx) ──────────────────────
    res_A, best_cells = _run_phase(
        phase="A", param="cells_per_half_period", values=PHASE_A_VALUES,
        fixed_cells=DEFAULT_CELLS, fixed_dy_div=DEFAULT_DY_DIV,
        fixed_dz_div=DEFAULT_DZ_DIV,
        cfg=cfg, checkpoint=checkpoint,
        metric=CONVERGENCE_METRIC, threshold=threshold,
    )
    _print_table("A", "cells_per_half_period", res_A)

    # ── Phase B: dz_divisor (controls dz = core_height / divisor) ────────
    res_B, best_dz_div = _run_phase(
        phase="B", param="dz_divisor", values=PHASE_B_VALUES,
        fixed_cells=best_cells, fixed_dy_div=DEFAULT_DY_DIV,
        fixed_dz_div=DEFAULT_DZ_DIV,
        cfg=cfg, checkpoint=checkpoint,
        metric=CONVERGENCE_METRIC, threshold=threshold,
    )
    _print_table("B", "dz_divisor", res_B)

    # ── Final recommendation ──────────────────────────────────────────────
    dx_rec = half_pitch / best_cells * 1e9
    dz_rec = cfg.geometry.core_height_m  / best_dz_div * 1e9

    print(f"\n{'=' * 62}")
    print("  RECOMMENDED MESH SETTINGS")
    print(f"{'=' * 62}")
    print(f"  cells_per_half_period : {best_cells}  ->  dx = {dx_rec:.1f} nm")
    print(f"  dz_divisor            : {best_dz_div}  ->  dz = {dz_rec:.1f} nm")
    print(f"  dy_divisor            : {DEFAULT_DY_DIV}  ->  dy = {cfg.geometry.width_narrow_m / DEFAULT_DY_DIV * 1e9:.1f} nm  (not swept — already well-resolved)")
    print(f"\n  (Current defaults: dx=50 nm, dz=50 nm, dy=50 nm)")
    print(f"  Results saved to: {CONV_DIR}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
