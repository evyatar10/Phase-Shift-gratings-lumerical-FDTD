"""
Verification re-simulation of the current PSO global-best particle.

This run uses PSO's EXACT scan settings (10 nm window, 201 freq points,
center = baseline λ_resonance ≈ 1562.043 nm) so the source bandwidth and
frequency sampling match PSO's evaluations apples-to-apples. The standard
post-processing pipeline applies find_bragg_resonance (sim_helpers.py:83 —
sharpness × dip-depth scoring) to extract the true cavity peak T, not
max(T). Compared against PSO's reported max(T):

  - If T_resonance ≈ 0.991 → source bandwidth was not the cause; PSO is honest.
  - If T_resonance ≈ 0.972 → the 0.02 gap reproduces independent of scan
    width, so there's a hidden geometry/build difference still unresolved.

Current PSO global best (DGX 470634, gen 7 particle 1, peak_T = 0.991860):
    dw_inner = [66.00, 202.47] nm
    shift    = [125.86, 66.57] nm
    cavity   = 781.35 nm
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from simulation_config import SimulationConfig


def run(_unused_cfg=None) -> dict:
    cfg = SimulationConfig()

    cfg.grating.n_periods_each_side = 80
    cfg.grating.lengthen_cavity     = True
    cfg.grating.n_free_inner_teeth  = 2
    cfg.grating.inner_dw_nm         = [66.00, 202.47]
    cfg.grating.inner_shift_nm      = [125.86, 66.57]
    cfg.grating.cavity_width_m      = 781.35e-9
    cfg.grating.cavity_width_option = "avg"

    cfg.mesh.simulation_mode  = "optimization"

    # Match PSO exactly: 10 nm window, 201 points, center on baseline λ.
    cfg.spectral.center_wavelength_m = 1.562043e-6
    cfg.spectral.scan_width_nm       = 10.0
    cfg.spectral.n_wl_points         = 201

    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled          = False

    return run_single_sim(cfg)


if __name__ == "__main__":
    run()
