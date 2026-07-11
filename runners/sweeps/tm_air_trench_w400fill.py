"""
tm_air_trench_w400fill — fill the MISSING w=400 nm air-trench offsets (2026-07-08).

The original tm_air_trench sweep scanned width 800 nm over d {1.2,1.5,1.8,2.1,2.4} µm
but width 400 nm over only d {1.5,1.8,2.1} µm. This adds the two endpoints so the
400 nm curve spans the same offset range as 800 nm:

   0   air trench (n=1.0) L=84 µm, width 400 nm, d = 1.2 µm
   1   air trench (n=1.0) L=84 µm, width 400 nm, d = 2.4 µm

Identical BASE to tm_air_trench (the stack: W1050 + [+20,+20] + see-saw 1040/980,
accurate mesh, box y=8.0 µm, z-mult 5.42, window 1556.5/40/3001) so the new points
drop straight onto the existing peak-T-vs-offset figure. Output filename tags
(Y1200 / Y2400 at W400) are distinct from every existing file.

Dispatch (serialize — only after the queue drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_air_trench_w400fill
Output -> results/tm_air_trench_w400fill/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 8.0
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5565e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.mesh.simulation_mode = "accurate"

# Base device = the stack
BASE.grating.cavity_width_m = 1050e-9
BASE.grating.inner_shift_nm = [20.0, 20.0]
BASE.grating.n_free_inner_teeth = 2
BASE.grating.width_narrow_per_tooth_m = [600e-9, 600e-9]
BASE.grating.width_wide_per_tooth_m = [1040e-9, 980e-9]

# Trench machinery: rect scatterer, tall (covers core + evanescent), mirrored ±y.
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.x_m = 0.0
BASE.scatterer.mirrored_y = True
BASE.scatterer.height_m = 2.0e-6

_L, _W, _D, _IDX = [], [], [], []


def row(L_um=84.0, w_nm=400.0, d_um=1.8, idx=1.0):
    _L.append(L_um); _W.append(w_nm); _D.append(d_um * 1000.0); _IDX.append(idx)


row(d_um=1.2)                                    # 0 w400 d=1.2 (missing endpoint)
row(d_um=2.4)                                    # 1 w400 d=2.4 (missing endpoint)

_n = 2
assert all(len(v) == _n for v in (_L, _W, _D, _IDX))
assert len(set(zip(_L, _W, _D, _IDX))) == _n, "duplicate row"

SPEC = SweepSpec(
    scatterer_x_span_um = _L,
    scatterer_y_span_nm = _W,
    scatterer_y_nm      = _D,
    scatterer_index     = _IDX,
    mode  = "zipped",
    label = "tm_air_trench_w400fill",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
