"""
Inner tooth size sweep for Pi-Shift Bragg Grating FDTD.

Sweeps the corrugation depth of the innermost grating tooth on both sides of
the cavity over INNER_SIZE_VALUES_NM, for each predefined tooth shift value in
TOOTH_SHIFT_VALUES_NM.

The conceptual maximum of the size sweep equals the global corrugation depth
(cfg.geometry.corrugation_depth_m, default 300 nm). This is not hard-coded —
use that config field if you need to reference it programmatically.

Sweep configuration (modify here):
    INNER_SIZE_VALUES_NM  — corrugation depth values for the innermost tooth [nm]
    TOOTH_SHIFT_VALUES_NM — tooth shift values to run each size sweep for [nm]

Total simulations = len(TOOTH_SHIFT_VALUES_NM) * len(INNER_SIZE_VALUES_NM).

Usage:
    python run_sweep_inner_tooth_size.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import gc
import time

import matplotlib.pyplot as plt

import config
import post_processing
from bragg_device_inner_size import PiShiftBraggFDTDWithInnerSize
from sim_helpers import apply_monitor_overrides, generate_file_tag
from simulation_config import SimulationConfig

# ── Sweep configuration ────────────────────────────────────────────────────────

# Innermost tooth corrugation depth values to sweep [nm].
# 0 = flat (no corrugation); max conceptually = cfg.geometry.corrugation_depth_m.
INNER_SIZE_VALUES_NM =  [100,125,150,200]

# Tooth shift values (nm) for which the full size sweep is repeated.
TOOTH_SHIFT_VALUES_NM = [90, 130]

# Cavity shortening from pitch/2 reference for cavity_width_option="avg" at λ=1560.1 nm,
# computed by python_tools/recommend_cavity_length.py (n_narrow/n_avg from FDE).
# Set to None to use the default pitch/2 with no correction.
PHASE_MATCHED_CAVITY_DETUNING_NM = 5.76   # = pitch/2 − 244.24 nm

# ── Core simulation function ───────────────────────────────────────────────────


def run_single_sim_with_inner_size(
    cfg: SimulationConfig,
    shift_m: float,
    inner_size_nm: float,
    lengthen_cavity: bool = True,
) -> dict:
    """Build, run, and analyze one simulation for a given shift + inner tooth size."""
    kwargs = cfg.to_device_kwargs()
    kwargs['innermost_tooth_shift_m'] = shift_m
    kwargs['lengthen_cavity'] = lengthen_cavity
    kwargs['innermost_corrugation_depth_m'] = inner_size_nm * 1e-9

    sim = PiShiftBraggFDTDWithInnerSize(**kwargs)

    shift_nm = round(shift_m * 1e9, 1)
    base_tag = generate_file_tag(sim)
    if cfg.farfield.enabled:
        base_tag += "_ff"
    tag = f"{base_tag}_shift_{shift_nm}nm_innersize_{inner_size_nm:.0f}nm"
    if not lengthen_cavity:
        tag += "_fixed_cav"

    shift_subfolder = f"shift_{shift_nm:.0f}nm"
    layouts_dir = os.path.join(config.LAYOUTS_DIR, shift_subfolder)
    results_dir = os.path.join(config.RESULTS_DIR,  shift_subfolder)
    os.makedirs(layouts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    layout_path  = os.path.join(layouts_dir, f"layout_{tag}.fsp")
    results_path = os.path.join(results_dir,  f"result_{tag}.mat")

    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )
    apply_monitor_overrides(sim, cfg)
    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.3f} s")

    try:
        results = post_processing.analyze_simulation(sim, cfg, results_path, tag)
        results['innermost_tooth_shift_nm'] = shift_nm
        results['innermost_corrugation_depth_nm'] = inner_size_nm
    finally:
        plt.close("all")
        sim.close()

    return results


# ── Sweep runner ───────────────────────────────────────────────────────────────


def run_sweep_inner_tooth_size(
    base_cfg: SimulationConfig,
    lengthen_cavity: bool = True,
) -> list:
    """
    Run the full 2D sweep: for each shift in TOOTH_SHIFT_VALUES_NM, iterate
    over all values in INNER_SIZE_VALUES_NM.

    Returns a flat list of result dicts, one entry per simulation.
    """
    n_total = len(TOOTH_SHIFT_VALUES_NM) * len(INNER_SIZE_VALUES_NM)
    run_idx = 0
    all_results = []

    print("=" * 60)
    print("INNER TOOTH SIZE SWEEP")
    print(f"  Shift values:      {TOOTH_SHIFT_VALUES_NM} nm")
    print(f"  Inner size values: {INNER_SIZE_VALUES_NM} nm")
    print(f"  Lengthen cavity:   {lengthen_cavity}")
    print(f"  Total simulations: {n_total}")
    print("=" * 60)

    for shift_nm in TOOTH_SHIFT_VALUES_NM:
        shift_m = shift_nm * 1e-9
        for size_nm in INNER_SIZE_VALUES_NM:
            run_idx += 1
            print(
                f"\n>>> RUN {run_idx}/{n_total}: "
                f"shift = {shift_nm} nm, inner_size = {size_nm} nm <<<\n"
            )
            iter_cfg = copy.deepcopy(base_cfg)
            try:
                results = run_single_sim_with_inner_size(
                    iter_cfg, shift_m, size_nm, lengthen_cavity=lengthen_cavity
                )
                all_results.append(results)
            except Exception as e:
                print(f"ERROR in run {run_idx}: {e}")
                raise
            finally:
                gc.collect()
                print("Memory cleared.\n")

    print("=" * 60)
    print(f"SWEEP COMPLETE: {n_total} runs finished.")
    print("=" * 60)
    return all_results


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_sweep_results(all_results: list) -> None:
    """
    Plot transmission spectra grouped by shift value.
    One subplot per shift, one line per inner tooth size.
    """
    n_shifts = len(TOOTH_SHIFT_VALUES_NM)
    fig, axes = plt.subplots(1, n_shifts, figsize=(7 * n_shifts, 5), squeeze=False)

    for col, shift_nm in enumerate(TOOTH_SHIFT_VALUES_NM):
        ax = axes[0][col]
        for res in all_results:
            if res['innermost_tooth_shift_nm'] == float(shift_nm):
                size_nm = res['innermost_corrugation_depth_nm']
                ax.plot(res['wl_nm'], res['T'], label=f"inner size = {size_nm:.0f} nm")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Transmission")
        ax.set_title(f"Shift = {shift_nm} nm")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Transmission vs Wavelength — Inner Tooth Size Sweep")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = SimulationConfig()

    cfg.grating.n_periods_each_side = 80
    cfg.mesh.simulation_mode = "optimization"
    cfg.spectral.scan_width_nm = 16.0

    if PHASE_MATCHED_CAVITY_DETUNING_NM is not None:
        cfg.grating.cavity_detuning_nm = PHASE_MATCHED_CAVITY_DETUNING_NM

    SHOW_PLOTS = False

    all_results = run_sweep_inner_tooth_size(cfg, lengthen_cavity=True)
    if SHOW_PLOTS:
        plot_sweep_results(all_results)
