"""
Innermost tooth shift sweep for Pi-Shift Bragg Grating FDTD.

Sweeps the innermost grating tooth shift over [0, 35, 70] nm,
plots all three transmission spectra on one figure, and saves
a separate .mat file per shift value.

Usage:
    python run_sweep_innermost_shift.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import gc
import time

import numpy as np
import matplotlib.pyplot as plt

import config
import post_processing
from bragg_device_shifted import PiShiftBraggFDTDWithShift
from sim_helpers import apply_monitor_overrides, generate_file_tag
from simulation_config import SimulationConfig

#SHIFT_VALUES_M = np.array([0, 60, 90, 110, 130, 150, 170]) * 1e-9  # nm
SHIFT_VALUES_M = np.array([100]) * 1e-9  # nm

# Cavity shortening from pitch/2 reference for cavity_width_option="avg" at λ=1560.1 nm,
# computed by python_tools/recommend_cavity_length.py (n_narrow/n_avg from FDE).
# Set to None to use the default pitch/2 with no correction.
PHASE_MATCHED_CAVITY_NEG_DETUNING_NM = 5.76   # = pitch/2 − 244.24 nm

def run_single_sim_with_shift(cfg: SimulationConfig, shift_m: float, lengthen_cavity: bool = True) -> dict:
    """Build, run, and analyze one simulation with the given innermost tooth shift."""
    kwargs = cfg.to_device_kwargs()
    kwargs['innermost_tooth_shift_m'] = shift_m
    kwargs['lengthen_cavity'] = lengthen_cavity
    sim = PiShiftBraggFDTDWithShift(**kwargs)

    # Build file tag — append FF, shift, and cavity-mode suffixes as needed
    base_tag = generate_file_tag(sim)
    if cfg.farfield.enabled:
        base_tag += "_ff"
    if shift_m > 0.0:
        shift_nm = round(shift_m * 1e9, 1)
        tag = f"{base_tag}_shift_{shift_nm}nm"
    else:
        tag = base_tag
    if shift_m > 0.0 and not lengthen_cavity:
        tag += "_fixed_cav"

    layout_path  = os.path.join(config.LAYOUTS_DIR, f"layout_{tag}.fsp")
    results_path = os.path.join(config.RESULTS_DIR,  f"result_{tag}.mat")
    os.makedirs(os.path.dirname(layout_path),  exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

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
        results['innermost_tooth_shift_nm'] = shift_m * 1e9
    finally:
        plt.close("all")
        sim.close()

    return results


def run_sweep_innermost_shift(base_cfg: SimulationConfig, lengthen_cavity: bool = True) -> list:
    all_results = []

    print("=" * 60)
    print("INNERMOST TOOTH SHIFT SWEEP")
    print(f"  Values: {[v * 1e9 for v in SHIFT_VALUES_M]} nm")
    print(f"  Lengthen cavity: {lengthen_cavity}")
    print("=" * 60)

    for i, shift_m in enumerate(SHIFT_VALUES_M):
        print(f"\n>>> RUN {i + 1}/{len(SHIFT_VALUES_M)}: shift = {shift_m * 1e9:.1f} nm <<<\n")
        iter_cfg = copy.deepcopy(base_cfg)
        try:
            results = run_single_sim_with_shift(iter_cfg, shift_m, lengthen_cavity=lengthen_cavity)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR in iteration {i + 1}: {e}")
            raise
        finally:
            gc.collect()
            print("Memory cleared.\n")

    print("=" * 60)
    print(f"SWEEP COMPLETE: {len(SHIFT_VALUES_M)} runs finished.")
    print("=" * 60)
    return all_results


def plot_sweep_results(all_results: list) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for res in all_results:
        shift_nm = res['innermost_tooth_shift_nm']
        ax.plot(res['wl_nm'], res['T'], label=f"shift = {shift_nm:.0f} nm")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Transmission")
    ax.set_title("Transmission vs Wavelength — Innermost Tooth Shift Sweep")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = SimulationConfig()

    cfg.grating.n_periods_each_side = 80

    # Simulation mode: "accurate" (dx≈35nm, cells=7) or "optimization" (dx=50nm, cells=5)
    cfg.mesh.simulation_mode = "optimization"
    cfg.spectral.scan_width_nm = 20.0

    if PHASE_MATCHED_CAVITY_NEG_DETUNING_NM is not None:
        cfg.grating.cavity_neg_detuning_nm = PHASE_MATCHED_CAVITY_NEG_DETUNING_NM

    # Optional overrides (uncomment to use):
    # cfg.spectral.center_wavelength_m = 1.5625e-6
    # cfg.monitors.record_2d_fields = False
    # cfg.farfield.enabled = False

    all_results = run_sweep_innermost_shift(cfg, lengthen_cavity=True)
    plot_sweep_results(all_results)
