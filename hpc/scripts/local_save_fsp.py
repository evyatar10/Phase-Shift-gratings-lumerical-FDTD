"""
Run locally on Windows to build the device and save the .fsp layout file.
Does NOT run the simulation — that happens on Zeus via the FDTD engine.

Usage:
    python hpc/scripts/local_save_fsp.py
    python hpc/scripts/local_save_fsp.py --output-path C:/some/dir
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)

import config
from bragg_device import PiShiftBraggFDTD
from sim_helpers import apply_monitor_overrides, generate_file_tag
from simulation_config import SimulationConfig

parser = argparse.ArgumentParser()
parser.add_argument('--output-path', default=None,
                    help='Directory to save the .fsp file (default: config.LAYOUTS_DIR)')
args = parser.parse_args()

# ── Configure ────────────────────────────────────────────────────────────────
cfg = SimulationConfig()
cfg.mesh.simulation_mode = "optimization"

# Override parameters here if needed:
# cfg.grating.n_periods_each_side = 120
# cfg.apodization.enabled = True
# cfg.spectral.center_wavelength_m = 1.560e-6

# ── Build device ──────────────────────────────────────────────────────────────
sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())

tag = generate_file_tag(sim)
if cfg.monitors.record_3d_fields:
    tag += "_3D"
if cfg.farfield.enabled:
    tag += "_ff"

output_dir = args.output_path or config.LAYOUTS_DIR
os.makedirs(output_dir, exist_ok=True)
layout_path = os.path.join(output_dir, f"layout_{tag}.fsp")

sim.build()
sim.update_scan(
    center_lambda_m=cfg.spectral.center_wavelength_m,
    width_nm=cfg.spectral.scan_width_nm,
    n_points=cfg.spectral.n_wl_points,
)
apply_monitor_overrides(sim, cfg)

sim.fdtd.save(layout_path)
print(f"FSP_SAVED:{layout_path}")  # deploy.sh parses this line
sim.close()
