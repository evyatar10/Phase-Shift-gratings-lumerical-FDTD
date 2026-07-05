"""
ACCURATE-MESH CONFIRM — final validation of the in-scope cavity winner.

cavity_combo_study (job 117553, opt mesh dx=50) closed the shape question: the
optimum is purely scalar — plain RECT cavity 1050 nm (-30% loss @ +0.6% fwhm);
barrel/tri7 on top HURT (area past the optimum). The tilt rows came out -1% vs
rect 1050 (dT ~ +0.001), AT the dx=50 jitter floor (~0.002) — per CLAUDE.md §2
a floor-level candidate is not a result until it survives accurate mesh, where
the floor drops to ~1e-4 (pillar precedent, 2026-07-02).

Rows (zipped, 4 tasks, ALL accurate mesh dx~35, converged box, 1558.5/40/3001):
  0 control (rect 800)
  1 rect 1050            (the champion — headline number for the report)
  2 tilt 150 on 1050     (does the -0.001 survive?)
  3 tilt 300 on 1050     (dose partner: if real, 300 tracks 150; if noise, they scatter)
Registered predictions: row1 loss cut confirms near -30%; rows 2-3 the open
question — survive together (real, tiny) or scatter by ~1e-4 (noise, tilt adds
nothing on the widened cavity).

All four geometries are distinct -> no tag collision despite mesh not being in
the file tag (all rows same mesh anyway, own study dir).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.cavity_acc_confirm
Output -> results/cavity_acc_confirm/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_cwid   = [None,   1050.0, 1050.0, 1050.0]
_cshape = ["rect", "rect", "tilt", "tilt"]
_cdepth = [0.0,    0.0,    150.0,  300.0]

_n = 4
assert len(_cwid) == len(_cshape) == len(_cdepth) == _n
assert len(set(zip(_cwid, _cshape, _cdepth))) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm       = _cwid,
    cavity_shape          = _cshape,
    cavity_shape_depth_nm = _cdepth,
    simulation_mode       = ["accurate"] * _n,
    mode  = "zipped",
    label = "cavity_acc_confirm",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
