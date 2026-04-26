"""
Experiment card examples for Pi-Shift Bragg Grating FDTD.

Three usage patterns:
  1. Single device run
  2. Multi-device comparison (run_cards)
  3. Parameter sweep via experiment card

Uncomment the example you want to run, or import and adapt in your own script.

Usage:
    python -m runners.single.run_experiment         # from project root
    python runners/single/run_experiment.py         # also works
"""

import os
import sys

# Make the project root importable when invoked as `python runners/single/run_experiment.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_card import ExperimentCard, run_card, run_cards
from simulation_config import SimulationConfig
from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec


# ═══════════════════════════════════════════════════════════════════════════════
# Example 1: Single device run
# ═══════════════════════════════════════════════════════════════════════════════

def example_single_run(base=None):
    """Run a single experiment card — the simplest workflow."""
    card = ExperimentCard(
        n_periods_each_side=50,
        center_mod_depth_nm=15,
        apod_method='tanh',
        tanh_steepness=2.5,
        label="Sample A",
    )
    return run_card(card, base=base)


# ═══════════════════════════════════════════════════════════════════════════════
# Example 2: Compare multiple devices
# ═══════════════════════════════════════════════════════════════════════════════

def example_compare_devices(base=None):
    """Run several cards back-to-back for comparison."""
    cards = [
        ExperimentCard(
            n_periods_each_side=40,
            center_mod_depth_nm=10,
            label="Dev1 — shallow mod",
        ),
        ExperimentCard(
            n_periods_each_side=40,
            center_mod_depth_nm=20,
            label="Dev2 — deep mod",
        ),
        ExperimentCard(
            n_periods_each_side=60,
            center_mod_depth_nm=10,
            cavity_length_nm=300,           # override cavity length
            label="Dev3 — more periods, longer cavity",
        ),
    ]
    return run_cards(cards, base=base)


# ═══════════════════════════════════════════════════════════════════════════════
# Example 3: Sweep one card parameter
# ═══════════════════════════════════════════════════════════════════════════════

def example_sweep(base=None):
    """Sweep n_periods_each_side while holding other card values fixed."""
    card = ExperimentCard(
        center_mod_depth_nm=15,
        corrugation_depth_nm=200,
        label="Period sweep",
    )
    spec = SweepSpec(n_periods_each_side=[30, 40, 50, 60], label="period_sweep")
    run_sweep_spec(spec, target="local", base=card.to_config(base))


# ═══════════════════════════════════════════════════════════════════════════════
# Server entry point — used by Zeus/Athena dispatchers (RUN_SCRIPT=run_experiment)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg=None):
    """Default 'experiment' run for HPC dispatch — runs example_single_run.
    Edit this body to switch to example_compare_devices or example_sweep."""
    return example_single_run(base=cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — uncomment one example to run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simulation mode: "accurate" (dx≈35nm, cells=7) or "optimization" (dx=50nm, cells=5)
    # Pass as base config to example functions so all cards inherit this setting.
    base_cfg = SimulationConfig()
    base_cfg.mesh.simulation_mode = "optimization"

    # example_single_run(base=base_cfg)
    # example_compare_devices(base=base_cfg)
    # example_sweep(base=base_cfg)
    pass
