"""Rect vs circle comb: does the POST SHAPE matter at the same comb positions?

Study dir: runners/scatterers/   |   Created 2026-08-27   |   Job(s): TBD
Purpose (user): the comb is drawn as circles; fab draws rectangles more happily.
Same 57 sites (Lambda 531 / dx 401 = 270 deg / d 1.9 um / h350), circles swapped
for rectangles, on the corr-325 N=100 surrogate at the campaign box.

Rows (zipped):
  0  rect 142 x 142 nm  — EQUAL AREA to the r=80 circle (pi r^2 = 20106 nm^2)
  1  rect 160 x 160 nm  — same BOUNDING BOX (area 1.273x -> r_eff 90.3 nm)
  2  rect 100 x 200 nm  — equal area, elongated ACROSS the guide (aspect 1:2)
  3  cylinder r 90.3 nm — EQUAL AREA to row 1: separates AREA from SHAPE.
     Row 1 ~ row 3  => only the cross-section area matters, shape is free (fab wins).
     Row 1 != row 3 => the shape itself carries the scattering.

REGISTERED PREDICTION (Rayleigh regime, post ~ 0.1 lambda): the 2D polarizability
of a small dielectric post is area-dominated and only weakly shape-dependent, so
rows 0/2 ~ the stored circle control and row 1 ~ row 3 (a ~27% stronger comb).

Controls: NOT re-run (CLAUDE.md section 6). Circle r=80 at IDENTICAL numerics is
  results_from_athena/tm_comb_box_c325/results/
  result_N100_TM_avg_C325_Ybox6p8_Zbox6p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat
  MEASURED T 0.92079 / lambda 1559.011 / |FWHM| 0.8812 nm (Q 1769).
Scale for judging the deltas: the comb itself is worth dT = +0.0104 (bare N=100
  T 0.91044, results_from_igum/tm_nladder_c325, box 8.0/8.8, vs comb 0.92085 same
  box) — the dx=50 nm jitter floor in this program is ~0.0018.

Physics line (section 4): TM h350, pitch 516.83, corr 325, W800, N=100/side;
window 20 nm / 4001 pts centered 1559.5; box y 6.8 / z 6.8 (campaign box).

Dispatch (queue EMPTY of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_rect_comb
Output -> results/scat_rect_comb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common
from runners.lumopt2_design.lumopt2_design import (
    COMB_LAM_NM, COMB_DX_NM, COMB_R_NM, COMB_D_NM, COMB_N_HALF, N_COMB)

CORR_NM        = 325.0
N_PERIODS      = 100                 # settled surrogate (2 kappa L = 3.65)
SCAN_CENTER_NM = 1559.5              # q3db family numerics
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001
Y_UM, Z_MULT   = 6.8, 4.14           # campaign box (z = 0.35 + 4.14*1.5595 = 6.8)
HEIGHT_NM      = 350.0

# (shape, x_span_nm, y_span_nm, radius_nm)
ROWS = [("rect",     142.0, 142.0,  0.0),
        ("rect",     160.0, 160.0,  0.0),
        ("rect",     100.0, 200.0,  0.0),
        ("cylinder",   0.0,   0.0, 90.3)]

X_COMB = [round(k * COMB_LAM_NM + COMB_DX_NM, 1)
          for k in range(-COMB_N_HALF, COMB_N_HALF + 1)]

for shape, xs, ys, r in ROWS:
    half_y = r if shape == "cylinder" else 0.5 * ys
    assert COMB_D_NM + half_y + 1200.0 <= Y_UM * 1000.0 / 2.0, "comb too close to y PML"
    assert COMB_D_NM - half_y >= _common.TOOTH_EDGE_NM + 100.0, "comb touches the teeth"
assert COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + 240.0 <= N_PERIODS * 516.83

BASE = _common.build_ports_base()
BASE.y_span_override_m = None                       # per-row via y_span_um
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"

_N = len(ROWS)
SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * _N,
    n_periods_each_side  = [N_PERIODS] * _N,
    center_wavelength_nm = [SCAN_CENTER_NM] * _N,
    scatterer_shape      = [r[0] for r in ROWS],
    scatterer_x_span_um  = [r[1] / 1000.0 for r in ROWS],
    scatterer_y_span_nm  = [r[2] for r in ROWS],
    scatterer_radius_nm  = [r[3] for r in ROWS],
    scatterer_x_list_nm  = [X_COMB] * _N,
    scatterer_y_list_nm  = [[COMB_D_NM] * N_COMB] * _N,
    scatterer_height_nm  = [HEIGHT_NM] * _N,
    y_span_um            = [Y_UM] * _N,
    span_mult            = [Z_MULT] * _N,
    mode  = "zipped",
    label = "scat_rect_comb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"comb: {N_COMB} sites, x {X_COMB[0]} .. {X_COMB[-1]} nm, y {COMB_D_NM} nm, "
          f"circle control r {COMB_R_NM} nm (T 0.92079 stored)")
