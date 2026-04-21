"""
Run locally on Windows to build and save .fsp layout file(s).
Does NOT run simulations — that happens on Zeus via the FDTD engine.

Prints FSP_SAVED:<path> for each file — deploy.sh collects and uploads all.

Usage:
    python hpc/scripts/local_save_fsp.py --preset single
    python hpc/scripts/local_save_fsp.py --preset sweep_shift
    python hpc/scripts/local_save_fsp.py --preset sweep_inner_size
    python hpc/scripts/local_save_fsp.py --preset single --output-path C:/some/dir
"""
import argparse
import copy
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_DIR)

import config
from bragg_device import PiShiftBraggFDTD
from sim_helpers import apply_monitor_overrides, generate_file_tag
from simulation_config import SimulationConfig

# ── Presets — edit values here to match your server scripts ──────────────────
PRESETS = {
    "single": dict(
        shift_values_nm=None,
        inner_size_values_nm=None,
        use_apodization=False,
        n_apod_periods=1,
        n_periods_each_side=None,        # None = config default
        scan_width_nm=None,              # None = config default
        detuning_nm=5.76,
    ),
    "sweep_shift": dict(
        shift_values_nm=[100],
        inner_size_values_nm=None,
        use_apodization=False,
        n_apod_periods=1,
        n_periods_each_side=80,
        scan_width_nm=20.0,
        detuning_nm=5.76,
    ),
    "sweep_inner_size": dict(
        shift_values_nm=[90, 130],
        inner_size_values_nm=[100, 125, 150, 200],
        use_apodization=True,
        n_apod_periods=1,
        n_periods_each_side=80,
        scan_width_nm=16.0,
        detuning_nm=5.76,
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--preset', default='single', choices=list(PRESETS),
                    help='Which layout configuration to generate')
parser.add_argument('--output-path', default=None)
args = parser.parse_args()

p = PRESETS[args.preset]

cfg = SimulationConfig()
cfg.mesh.simulation_mode = "optimization"
if p['n_periods_each_side'] is not None:
    cfg.grating.n_periods_each_side = p['n_periods_each_side']
if p['scan_width_nm'] is not None:
    cfg.spectral.scan_width_nm = p['scan_width_nm']
if p['detuning_nm'] is not None:
    cfg.grating.cavity_neg_detuning_nm = p['detuning_nm']

output_dir = args.output_path or config.LAYOUTS_DIR
os.makedirs(output_dir, exist_ok=True)

print(f"=== local_save_fsp preset: {args.preset} ===")
print(f"  shift sweep:      {p['shift_values_nm'] or 'none'}")
print(f"  inner-size sweep: {p['inner_size_values_nm'] or 'none'}")
print(f"  apodization:      {p['use_apodization']}")
print(f"  n_periods:        {p['n_periods_each_side'] or 'config default'}")
print(f"  scan_width_nm:    {p['scan_width_nm'] or 'config default'}")
print(f"  detuning_nm:      {p['detuning_nm'] or 'config default'}")
print(f"  output dir:       {output_dir}")
print("=" * 40)

shifts = [s * 1e-9 for s in p['shift_values_nm']] if p['shift_values_nm'] else [None]
sizes  = list(p['inner_size_values_nm']) if p['inner_size_values_nm'] else [None]

for shift_m in shifts:
    for size_nm in sizes:
        iter_cfg = copy.deepcopy(cfg)
        kwargs = iter_cfg.to_device_kwargs()

        if shift_m is not None:
            kwargs['innermost_tooth_shift_m'] = shift_m
        if size_nm is not None:
            kwargs['use_apodization'] = p['use_apodization']
            kwargs['n_apod_periods_each_side'] = p['n_apod_periods']
            kwargs['center_mod_depth_nm'] = float(size_nm)

        sim = PiShiftBraggFDTD(**kwargs)

        tag = generate_file_tag(sim)
        if iter_cfg.monitors.record_3d_fields:
            tag += "_3D"
        if iter_cfg.farfield.enabled:
            tag += "_ff"
        if shift_m is not None and shift_m > 0.0:
            tag += f"_S{shift_m * 1e9:.0f}"
        if size_nm is not None:
            tag += f"_I{size_nm:.0f}"

        layout_path = os.path.join(output_dir, f"layout_{tag}.fsp")

        sim.build()
        sim.update_scan(
            center_lambda_m=iter_cfg.spectral.center_wavelength_m,
            width_nm=iter_cfg.spectral.scan_width_nm,
            n_points=iter_cfg.spectral.n_wl_points,
        )
        apply_monitor_overrides(sim, iter_cfg)
        sim.fdtd.save(layout_path)
        sim.close()
        print(f"FSP_SAVED:{layout_path}")
