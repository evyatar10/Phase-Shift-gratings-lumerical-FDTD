"""
Validate the index-scaling approach for the TE optimized device.

The TE PSO device was optimized at n_core=1.977 / n_clad=1.44. For the 8-device
comparison it must run at n_core=1.9963 / n_clad=1.444. We scale its geometry by
s = 1.977/1.9963 to (approximately) preserve the Bragg condition. This run tests
how good that approximation is by simulating the SAME device three ways at
identical mesh/scan:

  1. orig_old      — n=1.977/1.44,   unscaled geometry      (the reference optimum)
  2. unscaled_new  — n=1.9963/1.444, unscaled geometry      (naive reuse → detuned)
  3. scaled_new    — n=1.9963/1.444, geometry × s           (the scaling approach)

Validation = does (3) recover (1)'s resonance wavelength AND peak T? And does
(3) beat (2)? Scaling is approximate (n_core and n_clad changed by different
ratios → index contrast changed, which no geometric scale can fix), so we expect
(3) close to (1) but not exact.

Invocation:
  bash athena/deploy_athena.sh --option2 --run=validate_te_scaling --gpu=rtx6k
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig

S = 1.977 / 1.9963

# Saved verified TE PSO-optimized device.
DW = [69.09, 218.45]
SHIFT = [120.45, 62.05]
CAVITY_NM = 774.58
PITCH_NM = 500.0


def _cfg(n_core, n_clad, scale):
    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = 80
    cfg.grating.lengthen_cavity     = True
    cfg.grating.pitch_m             = PITCH_NM * scale * 1e-9
    cfg.grating.n_free_inner_teeth  = 2
    cfg.grating.inner_dw_nm         = [d * scale for d in DW]
    cfg.grating.inner_shift_nm      = [s * scale for s in SHIFT]
    cfg.grating.cavity_width_m      = CAVITY_NM * scale * 1e-9
    cfg.source.polarization         = "TE"
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const       = n_core
    cfg.material.n_clad_const       = n_clad
    cfg.mesh.simulation_mode        = "accurate"
    cfg.spectral.center_wavelength_m = 1.565e-6
    cfg.spectral.scan_width_nm       = 60.0          # [1535,1595] covers scaled (~1558) + detuned (~1573)
    cfg.spectral.n_wl_points         = 3001
    cfg.material.n_eff_guess        = cfg.spectral.center_wavelength_m / (2 * cfg.grating.pitch_m)
    cfg.apodization.enabled         = False
    cfg.monitors.record_2d_fields   = False
    cfg.monitors.record_3d_fields   = False
    cfg.farfield.enabled            = False
    return cfg


VARIANTS = [
    ("orig_old",     _cfg(1.977,  1.44,  1.0)),
    ("unscaled_new", _cfg(1.9963, 1.444, 1.0)),
    ("scaled_new",   _cfg(1.9963, 1.444, S)),
]


def run(_unused_cfg=None) -> dict:
    last = None
    for label, cfg in VARIANTS:
        print(f"\n===== variant: {label} =====")
        last = run_single_sim(cfg, show_plots=False, save_figs=False,
                              tag_suffix=f"_val_{label}")
        print(f"  {label}: T={last.get('resonance_transmission')}  "
              f"lam={last.get('resonance_wavelength_nm')}  "
              f"fwhm_um={last.get('fwhm_m', 0)*1e6:.3f}")
    return last


if __name__ == "__main__":
    run()
