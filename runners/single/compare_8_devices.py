"""
8-device comparison for the TE-vs-TM bar chart (peak transmission + spatial
mode-width FWHM). Runs all 8 devices at IDENTICAL settings so the bars are
truly comparable:

  mesh         = accurate
  n_core/n_clad = 1.9963 / 1.444  (constant, IT11-calibrated)
  pitch        = 500 nm (TE) / 518.3 nm (TM)   [proper per-polarization]
  n_periods    = 80 each side
  scan         = 1565 ± 25 nm, 2501 pts (fine enough for high-Q TM)

Devices (×TE, ×TM):
  baseline   — regular grating (no apodization)
  linear10   — linear apodization, 10 periods/side, 4 nm cavity floor
  tanh10     — tanh apodization, 10 periods/side, steepness 2.0, 4 nm floor
  optimized  — the PSO-optimized device

TE optimized was tuned at n_core=1.977, so its geometry is INDEX-SCALED by
s = 1.977/1.9963 (Maxwell scale-invariance: n·d preserved → same behaviour at
1.9963). TM optimized was already optimized at 1.9963 → no scaling.

Each device writes result_<geomtag>_cmp_<label>.mat; plot_compare_8_devices.m
reads them by the _cmp_<label> suffix.

Invocation:
  bash athena/deploy_athena.sh --option2 --run=compare_8_devices --gpu=h200
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig

# Index scaling for the TE optimized device (tuned at 1.977 → run at 1.9963).
S = 1.977 / 1.9963

# Saved, verified TE PSO-optimized device (result_N80_W781.mat, T=0.9614).
TE_OPT = dict(dw=[69.09, 218.45], shift=[120.45, 62.05], cavity_nm=774.58, pitch_nm=500.0)
# TM PSO-optimized device (job 97316, accurate-verified T=0.9561). Already 1.9963.
TM_OPT = dict(dw=[95.18, 102.00], shift=[0.0, 0.0], cavity_nm=854.30, pitch_nm=518.3)


def _base(pol: str, pitch_nm: float) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = 80
    cfg.grating.lengthen_cavity     = True
    cfg.grating.pitch_m             = pitch_nm * 1e-9
    cfg.source.polarization         = pol
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const       = 1.9963
    cfg.material.n_clad_const       = 1.444
    cfg.mesh.simulation_mode        = "accurate"
    cfg.spectral.center_wavelength_m = 1.565e-6
    cfg.spectral.scan_width_nm       = 50.0
    cfg.spectral.n_wl_points         = 2501
    cfg.material.n_eff_guess        = cfg.spectral.center_wavelength_m / (2 * cfg.grating.pitch_m)
    cfg.monitors.record_2d_fields   = True   # need yx (top) + zx (side) for both-plane mode width
    cfg.monitors.record_3d_fields   = False
    cfg.farfield.enabled            = False
    return cfg


def _baseline(pol, pitch_nm):
    cfg = _base(pol, pitch_nm)
    cfg.apodization.enabled = False
    return cfg


def _apod(pol, pitch_nm, method):
    cfg = _base(pol, pitch_nm)
    cfg.apodization.enabled = True
    cfg.apodization.method  = method            # 'linear' or 'tanh'
    cfg.apodization.n_apod_periods_each_side = 10
    cfg.apodization.center_mod_depth_nm = 4.0   # apodization floor at the cavity
    cfg.apodization.tanh_steepness = 2.0
    return cfg


def _optimized(pol, opt, scale):
    cfg = _base(pol, opt["pitch_nm"] * scale)
    cfg.apodization.enabled = False
    cfg.grating.n_free_inner_teeth = 2
    cfg.grating.inner_dw_nm    = [d * scale for d in opt["dw"]]
    cfg.grating.inner_shift_nm = [s * scale for s in opt["shift"]]
    cfg.grating.cavity_width_m = opt["cavity_nm"] * scale * 1e-9
    return cfg


# (label, cfg) — order = chart order.
DEVICES = [
    ("TE_baseline",  _baseline("TE", 500.0)),
    ("TE_linear10",  _apod("TE", 500.0, "linear")),
    ("TE_tanh10",    _apod("TE", 500.0, "tanh")),
    ("TE_optimized", _optimized("TE", TE_OPT, S)),     # index-scaled 1.977→1.9963
    ("TM_baseline",  _baseline("TM", 518.3)),
    ("TM_linear10",  _apod("TM", 518.3, "linear")),
    ("TM_tanh10",    _apod("TM", 518.3, "tanh")),
    ("TM_optimized", _optimized("TM", TM_OPT, 1.0)),   # already 1.9963 → no scaling
]


def run(_unused_cfg=None) -> dict:
    last = None
    for label, cfg in DEVICES:
        print(f"\n===== device: {label} =====")
        last = run_single_sim(cfg, show_plots=False, save_figs=False,
                              tag_suffix=f"_cmp_{label}")
        print(f"  {label}: T={last.get('resonance_transmission')}  "
              f"lam={last.get('resonance_wavelength_nm')}  fwhm_um={last.get('fwhm_m', 0)*1e6:.3f}")
    return last


if __name__ == "__main__":
    run()
