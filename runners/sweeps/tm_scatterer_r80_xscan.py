"""
TM lateral-scatterer x-scan at r = 80 nm — first 5 axial points (0 .. 1.08 um).

Adds an r = 80 nm curve to the tm_scatterer_scan family (which measured r =
100 / 150 / 200 nm). Same anchored TM device and IDENTICAL numerics/domain as
tm_scatterer_scan (its control = 0.79946 and r100@x810 = 0.80151 reproduced exactly
here), so these points overlay directly on that study's peak-T-vs-x plot.

Only the first 5 axial positions are taken — x = {0, 270, 540, 810, 1080} nm at the
same 270 nm step and y = 1000 nm offset as the old r=100 half-line — covering 0 up to
just past 1 um (through the x = 810 nm optimum). The r = 0 control already exists in
tm_scatterer_scan (radius-independent), so it is not re-run here.

Device (anchored TM baseline — CLAUDE.md 4; identical to tm_scatterer_scan)
-----------------------------------------------------------------------------
  height 350 nm, pitch 516.83 nm, corrugation 400 nm, N = 80 periods/side,
  n_core 1.97 / n_clad 1.444, mesh "optimization" (dx = 50 nm), ports-only.
  Window 30 nm around 1558.46 nm (3001 pts). Domain y_span 4.8 um.

Grid (mode='zipped'): r = 80 nm SiN cylinder PAIR (mirrored ±y) at
  (x, ±1000 nm) for x in {0, 270, 540, 810, 1080} nm  => 5 tasks.

File names encode _scR80_X{x}_Y1000_pair -> unique per task, distinct from the
r=100/150/200 files. One shot (well under the QOS 100/4 cap):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_r80_xscan
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


R_NM = 80.0                              # new radius curve
Y_NM = 1000.0                            # same lateral offset as the old half-line
X_LIST_NM = [0.0, 270.0, 540.0, 810.0, 1080.0]   # first 5 points, 270 nm step


def build_base() -> SimulationConfig:
    """Anchored TM device + proven narrow window — identical to tm_scatterer_scan."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 516.83e-9
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    cfg.geometry.corrugation_depth_m = 400e-9
    cfg.geometry.core_height_m = 350e-9

    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"
    cfg.source.polarization = "TM"

    cfg.spectral.center_wavelength_m = 1.5585e-6
    cfg.spectral.scan_width_nm = 30.0
    cfg.spectral.n_wl_points = 3001

    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False

    cfg.scatterer.enabled = True
    cfg.scatterer.mirrored_y = True
    cfg.y_span_override_m = 4.8e-6
    return cfg


BASE = build_base()

_xs = list(X_LIST_NM)
_rs = [R_NM] * len(_xs)
_ys = [Y_NM] * len(_xs)

assert len(_rs) == len(_xs) == len(_ys) == 5, f"task count changed: {len(_rs)}"
assert len(set(zip(_rs, _xs, _ys))) == len(_rs), "duplicate (r, x, y) row"


SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "tm_scatterer_r80_xscan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
