"""
Parameter sweep runner for Pi-Shift Bragg Grating FDTD.

Runs run_simulation.run_single_sim() once per value in sweep.values,
overriding the parameter specified by sweep.parameter each iteration.
Post-simulation analysis for each run is handled by post_processing.py.

Usage:
    python run_sweep.py

Configure the sweep in the __main__ block below, or import and call:
    from simulation_config import SimulationConfig
    from run_sweep import run_sweep
    cfg = SimulationConfig()
    cfg.sweep.parameter = "grating.n_periods_each_side"
    cfg.sweep.values = [80, 100, 120]
    run_sweep(cfg)
"""

import copy
import gc

import matplotlib.pyplot as plt

from simulation_config import SimulationConfig, set_nested_attr
from run_simulation import run_single_sim


def run_sweep(base_cfg: SimulationConfig) -> None:
    """
    Run the simulation once per sweep value, overriding the target parameter.

    Uses dot notation from base_cfg.sweep.parameter to identify the config field.
    Each iteration receives a deep copy of the config so changes don't leak.
    """
    param_path = base_cfg.sweep.parameter
    values = base_cfg.sweep.values

    print(f"{'=' * 60}")
    print(f"PARAMETER SWEEP")
    print(f"  Parameter : {param_path}")
    print(f"  Values    : {values}")
    print(f"  INTERCONNECT export: {base_cfg.run.export_interconnect}")
    print(f"{'=' * 60}\n")

    for i, value in enumerate(values):
        print(f"\n>>> RUN {i + 1}/{len(values)}: {param_path} = {value} <<<\n")

        iter_cfg = copy.deepcopy(base_cfg)
        set_nested_attr(iter_cfg, param_path, value)

        try:
            run_single_sim(iter_cfg)
        except Exception as e:
            print(f"ERROR in sweep iteration {i + 1}: {e}")
            raise
        finally:
            plt.close("all")  # prevent figure accumulation across sweep iterations
            gc.collect()
            print("Memory cleared.\n")

    print(f"\n{'=' * 60}")
    print(f"SWEEP COMPLETE: {len(values)} runs finished.")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = SimulationConfig()

    # ── Configure the sweep ──────────────────────────────────────────────
    cfg.sweep.parameter = "grating.n_periods_each_side"
    cfg.sweep.values = [80, 100, 120]

    # ── Runtime options ──────────────────────────────────────────────────
    # cfg.run.export_interconnect = False   # Skip INTERCONNECT .txt export
    # cfg.run.cleanup_lumerical_data = True # Delete .h5 temp files after each run

    # ── Override base config for all sweep runs (optional) ───────────────
    # cfg.spectral.center_wavelength_m = 1.562673e-6
    # cfg.spectral.scan_width_nm = 30.0
    # cfg.apodization.n_apod_periods_each_side = 5
    # cfg.monitors.record_3d_fields = True
    # cfg.farfield.enabled = True

    run_sweep(cfg)
