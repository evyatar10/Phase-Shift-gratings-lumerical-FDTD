"""
Core simulation pipeline for Pi-Shift Bragg Grating FDTD.

Responsibilities of this module:
  1. Build the FDTD device from config
  2. Run the simulation
  3. Delegate all post-simulation work to post_processing.analyze_simulation()

Post-simulation analysis (resonance, fields, far-field, saving, plotting)
lives in post_processing.py. See that module for details on each stage.

Usage:
    # With default parameters (defined in simulation_config.py):
    python -m runners.single.run_simulation         # from project root
    python runners/single/run_simulation.py         # also works

    # To override parameters, edit the __main__ block below or import:
    from simulation_config import SimulationConfig
    from runners.single.run_simulation import run_single_sim
    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = 120
    run_single_sim(cfg)
"""

import glob
import os
import sys
import time

# Make the project root importable when invoked as `python runners/single/run_simulation.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt

import config
import post_processing
from bragg_device import PiShiftBraggFDTD
from sim_helpers import apply_monitor_overrides, generate_file_tag
from simulation_config import SimulationConfig


def run_single_sim(cfg: SimulationConfig) -> dict:
    """
    Build, run, and analyze a single FDTD simulation.

    Returns the assembled results dict (also saved to disk as a .mat file).
    """

    # ── 1. Create device and resolve file paths ───────────────────────────
    sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())

    tag = generate_file_tag(sim)
    if cfg.monitors.record_3d_fields:
        tag += "_3D"
    if cfg.farfield.enabled:
        tag += "_ff"

    layout_path  = os.path.join(config.LAYOUTS_DIR, f"layout_{tag}.fsp")
    results_path = os.path.join(config.RESULTS_DIR,  f"result_{tag}.mat")

    # ── 2. Build device, configure scan, apply monitor settings ──────────
    sim.build()
    sim.update_scan(
        center_lambda_m=cfg.spectral.center_wavelength_m,
        width_nm=cfg.spectral.scan_width_nm,
        n_points=cfg.spectral.n_wl_points,
    )
    apply_monitor_overrides(sim, cfg)

    sim.fdtd.save(layout_path)
    print(f"Saved layout to: {layout_path}")

    # ── 3. Run simulation ─────────────────────────────────────────────────
    start = time.perf_counter()
    sim.fdtd.run()
    print(f"Simulation time: {time.perf_counter() - start:.3f} seconds")

    # ── 4. Post-simulation analysis (extraction → processing → saving) ────
    try:
        results = post_processing.analyze_simulation(sim, cfg, results_path, tag)
    finally:
        # ── 5. Cleanup temp files ─────────────────────────────────────────
        if cfg.run.cleanup_lumerical_data:
            cleanup_lumerical_temp_files(layout_path)

        plt.show()        # display figures when running locally
        plt.close("all")
        sim.close()       # free Lumerical session memory

    return results


def cleanup_lumerical_temp_files(layout_path: str) -> None:
    """Delete the .h5 scratch files Lumerical writes alongside the .fsp layout."""
    data_dir = layout_path.replace(".fsp", "")
    if not os.path.exists(data_dir):
        return
    try:
        for h5_file in glob.glob(os.path.join(data_dir, "*.h5")):
            os.remove(h5_file)
            print(f"Cleaned up: {h5_file}")
    except Exception as e:
        print(f"Warning: Could not clean up Lumerical temp files: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create config with defaults from simulation_config.py.
    # Override any parameters below before running:
    cfg = SimulationConfig()

    # Simulation mode: "accurate" (dx≈35nm, cells=7) or "optimization" (dx=50nm, cells=5)
    cfg.mesh.simulation_mode = "optimization"

    # Example overrides (uncomment to use):
    # cfg.grating.n_periods_each_side = 120
    # cfg.apodization.enabled = False
    # cfg.apodization.center_mod_depth_nm = 20.0
    # cfg.spectral.center_wavelength_m = 1.560e-6
    # cfg.monitors.record_3d_fields = True
    # cfg.farfield.enabled = False

    run_single_sim(cfg)
