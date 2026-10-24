"""
Study: re-simulate the two devices behind the 2026-04-02 meeting figure
"XZ Zoomed Side View — N=80 periods" vs "N=80 periods, 10 apodized" (xz.png /
xz_apod.png), whose result .mat files were lost (leaky_modes results folders
deleted). Two forwards with the 2D field monitors ON so field_xz_side is stored.

Device (user spec 2026-10-10, = the April 2026 device): TE, N=80/side, pitch
500 nm, avg width 800 nm, corrugation 300 nm, core 350 nm, cavity pitch/2 at avg
width, no tooth shift, constant index n_core 1.977 / n_clad 1.44 (the April
values — NOT today's 1.97/1.444 defaults, so the resonance lands at ~1560 nm).
Row 0: no apodization. Row 1: linear apodization over 10 teeth/side, 4 nm
centre depth. Mesh "optimization" (dx=50 nm, matches the 0.0497 um x-grid
stored in the original .fig). 2D monitors use the source limits with 51 points
over the 20 nm scan (0.4 nm bins, as the April files had).

Dispatch (Athena):
    bash athena/deploy_athena.sh --spec=runners.sweeps.xz_apod_vs_nonapod_n80
Outputs: results/xz_apod_vs_nonapod_n80/results/result_N80_avg.mat and
result_N80_A10_M4_avg.mat, each with field_xz_side (E_res + P_res).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.mesh.simulation_mode            = "optimization"
BASE.grating.n_periods_each_side     = 80
BASE.grating.cavity_width_option     = "avg"
BASE.material.n_core_const           = 1.977   # April 2026 value (figure resonance ~1560 nm)
BASE.material.n_clad_const           = 1.44
BASE.spectral.center_wavelength_m    = 1.5601e-6
BASE.spectral.scan_width_nm          = 20.0
BASE.spectral.n_wl_points            = 3001
BASE.spectral.n_2d_monitor_points    = 51
BASE.monitors.record_2d_fields       = True    # XY top, YZ cross, XZ side (the figure's plane)
BASE.monitors.record_3d_fields       = False
BASE.farfield.enabled                = False
# The April run used the 5λ transverse box: the saved xz.fig has z = ±3.6906 um,
# which is exactly (0.35 + 5.0*1.5601 - 0.5*1.5601)/2. Today's default (1.8λ)
# would give a 2.38 um-tall XZ monitor and miss the radiation lobes. NAMED §2
# numerics change vs the 1.8λ baselines: do not compare absolute T across.
BASE.span_multiplier_override        = 5.0

SPEC = SweepSpec(
    n_apod_periods_each_side = [0,      10],
    apod_method              = ["none", "linear"],
    center_mod_depth_nm      = [300.0,  4.0],
    mode  = "zipped",
    label = "xz_apod_vs_nonapod_n80",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
