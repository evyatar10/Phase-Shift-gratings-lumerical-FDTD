"""
TM in-core HOLE scan — flipped-material scatterer inside the device (paper_8 §7 variant).

The opposite-material counterpart of tm_scatterer_scan: there, SiN cylinders sit in
the oxide CLADDING beside the guide; here, a single SiO2 cylinder (n = 1.444) is
punched INTO the SiN core on the guide axis (y = 0, full 350 nm height, mesh order 1
so the hole wins the overlap with the teeth/cavity). Positions step out from the
cavity center along +x at pitch/8 = 64.60 nm, so the scan resolves the defect-mode
standing wave (8 samples per period): a hole at a field NODE should be nearly benign,
at an ANTINODE maximally scattering. Honest expectation (from the round-1 planning):
holes in the core most likely SPOIL Q / add loss — this scan measures where and how
much, and whether any position does something interesting (e.g. loss reduction by
interference, resonance retuning).

Device, window, numerics: identical to tm_scatterer_scan (anchored TM baseline —
pitch 516.83 nm, corr 400 nm, h 350 nm, N = 80, n 1.97/1.444, optimization mesh,
window 1558.5 nm / 30 nm / 3001 pts). An on-axis hole preserves BOTH symmetry
planes, so the default y/z symmetry BCs and the default domain span stay.
Expected hole-induced BLUE shift of lambda_res (removing high-index material) of
order ~1 nm near the cavity — well inside the +/-15 nm window margin.

Grid (mode='zipped', one SLURM array task per row):
  idx 0      : r = 0 control (no hole, identical numerics)
  idx 1-97   : r = 100 nm, x = k * pitch/8, k = 0..96  (0 .. 6.202 um = 12 periods)
  => 98 tasks. File tag: _scR100_X{x}_Y0_hole -> unique .fsp/.h5/.mat per task.

Run on Athena (default partition; SERIALIZE after tm_scatterer_scan — shared
data/sweep_list.txt, and the QOS allows only 100 submitted tasks per user):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_hole_scan

Output -> results/tm_hole_scan/results/ -> results_from_athena/tm_hole_scan/.
Plot with matlab_plotting/plot_hole_scan.m.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


PITCH_NM = 516.83
R_NM = 100.0               # hole radius; cavity half-length is ~129 nm so the
                           # x=0 hole sits fully inside the cavity segment
N_POS = 97                 # k = 0..96 -> 0 .. 6.202 um (12 grating periods)


def build_base() -> SimulationConfig:
    """Anchored TM device + proven narrow window — pinned explicitly (no env
    reads). Same values as tm_scatterer_scan.build_base, with the scatterer
    flipped to an on-axis SiO2 hole and the default domain span restored."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = PITCH_NM * 1e-9
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

    # On-axis SiO2 hole in the SiN core (flipped material). y = 0 keeps both
    # symmetry planes; the builder draws a single object at y = 0.
    cfg.scatterer.enabled = True
    cfg.scatterer.y_m = 0.0
    cfg.scatterer.index = 1.444
    return cfg


BASE = build_base()


# ── Position list (module literals — deterministic on both deploy sides) ──────
_x_hole = [round(k * PITCH_NM / 8.0, 1) for k in range(N_POS)]   # nm, 0 .. 6202.0

_rs = [0.0] + [R_NM] * len(_x_hole)
_xs = [0.0] + _x_hole

assert len(_rs) == len(_xs) == 98, f"task count changed: {len(_rs)}"
assert len(set(zip(_rs, _xs))) == len(_rs), \
    "duplicate (r, x) row -> file-tag collision (same .fsp/.h5/.mat)"


SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    mode  = "zipped",
    label = "tm_hole_scan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
