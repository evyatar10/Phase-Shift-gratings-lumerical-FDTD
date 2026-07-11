"""
tm_air_trench_regular — lateral AIR-TRENCH offset sweep on the REGULAR device (2026-07-08).

Companion to tm_air_trench, which ran the trench sweep on the OPTIMIZED "stack"
device (W1050 + [+20,+20] + see-saw). This repeats the same trench geometry on the
plain anchored TM pi-shift grating (build_base as-is: pitch 516.83, corrugation 400,
height 350, N=80/side, n_core 1.97) — NO cavity widening, NO inner shift, NO see-saw.
Regular-device defect resonance ~1558.5 nm.

Purpose: does the light-cone / TIR air-trench lever help the ordinary device too, or
was the gain specific to the optimized stack? Peak-T-vs-offset, 5 offsets x 2 widths.

Numerics = same converged box + accurate mesh as the optimized trench study:
  accurate mesh, y-box 8.0 um (trench edge |y|<=2.8 um keeps >=1.2 um PML clearance),
  z = 8.8 um (span-mult 5.42, program-wide converged z), window 1558.5/40/3001.

Rows (zipped, 11 tasks):
   0      control: the regular device, NO trench (spans None) — Delta reference
   1-5    air trench (n=1.0) L=84 um, width 800 nm, d {1.2,1.5,1.8,2.1,2.4} um
   6-10   air trench (n=1.0) L=84 um, width 400 nm, d {1.2,1.5,1.8,2.1,2.4} um

Dispatch (serialize — only after the queue drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_air_trench_regular
Output -> results/tm_air_trench_regular/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 8.0
BOX_Z_MULT = 5.42            # converged z = 8.8 um (job 116870), same as tm_air_trench

BASE = build_base()          # REGULAR device — no stack modifications applied
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6   # regular-device defect resonance
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.mesh.simulation_mode = "accurate"

# Trench machinery: rect scatterer, tall (covers core + evanescent), mirrored +/-y.
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.x_m = 0.0
BASE.scatterer.mirrored_y = True
BASE.scatterer.height_m = 2.0e-6

_L, _W, _D, _IDX = [], [], [], []


def row(L_um=None, w_nm=None, d_um=1.8, idx=1.0):
    _L.append(L_um); _W.append(w_nm); _D.append(d_um * 1000.0); _IDX.append(idx)


row(idx=1.0)                                     # 0 control (spans None => no trench)
for d in (1.2, 1.5, 1.8, 2.1, 2.4):              # 1-5 air trench w800
    row(L_um=84.0, w_nm=800.0, d_um=d, idx=1.0)
for d in (1.2, 1.5, 1.8, 2.1, 2.4):              # 6-10 air trench w400
    row(L_um=84.0, w_nm=400.0, d_um=d, idx=1.0)

_n = 11
assert all(len(v) == _n for v in (_L, _W, _D, _IDX))
assert len(set(zip(_L, _W, _D, _IDX))) == _n, "duplicate row -> file-tag collision"

SPEC = SweepSpec(
    scatterer_x_span_um = _L,
    scatterer_y_span_nm = _W,
    scatterer_y_nm      = _D,
    scatterer_index     = _IDX,
    mode  = "zipped",
    label = "tm_air_trench_regular",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
