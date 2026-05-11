"""
Final-baseline reference: 10-period tanh apodization, no tooth shifts,
avg cavity width, default n_periods=80, center_mod_depth=4 nm,
simulation_mode="optimization".

For direct comparison against the PSO-converged best (modal T = 0.972 at
[69.09, 218.45, 120.45, 62.05, 774.58]) and the regular-grating baseline
(modal T = 0.9419 in optimization mesh).
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
    cfg.grating.cavity_width_option = "avg"

    cfg.apodization.enabled                    = True
    cfg.apodization.n_apod_periods_each_side   = 10
    cfg.apodization.method                     = "tanh"
    cfg.apodization.center_mod_depth_nm        = 4.0

    cfg.mesh.simulation_mode  = "optimization"

    cfg.spectral.center_wavelength_m = 1.5601e-6
    cfg.spectral.scan_width_nm       = 20.0
    cfg.spectral.n_wl_points         = 1001

    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled          = False

    return run_single_sim(cfg)


if __name__ == "__main__":
    run()
