"""
Parameter-change comparison: record the full 3D field volume for two devices
that differ only in `grating.innermost_tooth_shift_m`, so the effect of the
shift on the cavity field can be inspected side-by-side.

The two .mat result files are then loaded by matlab_plotting/plot_field_3d.m,
which renders them as a true 2-structure comparison (paired XZ, XY, and YZ
panels with a shared dB color scale).

Run A: default SimulationConfig (innermost_tooth_shift_m = 0).
Run B: same config but innermost_tooth_shift_m = 100 nm.

Both runs share these overrides on top of the defaults:
  - monitors.record_3d_fields = True   (required by plot_field_3d.m)
  - monitors.record_2d_fields = False  (the MATLAB script slices everything from 3D)
  - farfield.enabled          = True   (also widens the simulation domain — span
                                        multiplier 5.0 instead of 1.8 wavelengths)
  - grating.cavity_width_option = "narrow"

Run from the repo root:
    python runners/single/compare_3d_field_default_vs_shift.py

Server entry point (Athena/Zeus dispatchers):
    RUN_SCRIPT=compare_3d_field_default_vs_shift  → run_compare_3d_field_default_vs_shift(cfg)
"""

import copy
import os
import sys

# Make the project root importable when invoked as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig


SHIFT_M = 100e-9   # second device's innermost-tooth shift


def _make_base_config(cfg: SimulationConfig | None = None) -> SimulationConfig:
    """Apply the overrides shared by both runs."""
    cfg = copy.deepcopy(cfg) if cfg is not None else SimulationConfig()
    cfg.monitors.record_3d_fields  = True
    cfg.monitors.record_2d_fields  = False
    cfg.farfield.enabled           = True
    cfg.grating.cavity_width_option = "narrow"
    return cfg


def run_compare_3d_field_default_vs_shift(cfg: SimulationConfig | None = None) -> list[dict]:
    """Run the two sims back-to-back. Returns the two result dicts in order."""
    base = _make_base_config(cfg)

    # --- Run A: default (no tooth shift) ---
    cfg_a = copy.deepcopy(base)
    cfg_a.grating.innermost_tooth_shift_m = 0.0
    print("\n" + "=" * 60)
    print("RUN A — default (innermost_tooth_shift = 0 nm)")
    print("=" * 60)
    res_a = run_single_sim(cfg_a)

    # --- Run B: 100 nm innermost-tooth shift ---
    cfg_b = copy.deepcopy(base)
    cfg_b.grating.innermost_tooth_shift_m = SHIFT_M
    print("\n" + "=" * 60)
    print(f"RUN B — innermost_tooth_shift = {SHIFT_M*1e9:.0f} nm")
    print("=" * 60)
    res_b = run_single_sim(cfg_b)

    print("\n" + "=" * 60)
    print("Both runs complete. Open MATLAB → run plot_field_3d.m → select")
    print("the two new result_*.mat files.")
    print("=" * 60)
    return [res_a, res_b]


if __name__ == "__main__":
    run_compare_3d_field_default_vs_shift()
