"""Stage H — retro-Bragg comb at the MEASURED needle angle (user pick, option 1).

Study dir: runners/scatterers/   |   Created 2026-07-18   |   Job(s): TBD
Purpose: the residual leak peaks at ux = 0.980 (theta 11.6 deg from the axis,
MEASURED sub-pixel from the stage-E far fields). A cladding comb with x-period
Lambda = lambda/(2*n_clad*ux) = 551 nm Bragg-kicks that wave by 2kx -> retro
(kx -> -kx), returning it to the source where it can interfere destructively
with ongoing emission. No prior reflector was period-matched (DBRs designed for
ux 0.5/0.7; PhC a=500; all failed as near-field drains, jobs 118360/119163).

Design (from measured numbers):
  Lambda_x 551 nm, r=110 SiN cylinders (planar 350), mirrored +/-y rows;
  standoff d >= 3 um (guided-mode evanescent field ~0.5% there; in-core job
  123303 proved a ~545 nm period at FULL overlap out-couples the carrier into
  the light cone -- standoff is the load-bearing safety);
  d-scan step 0.7 um spans most of the predicted interference cycle
  lambda/(2*n_clad*sin(theta)) = 2.68 um; 2-row variant spaced 2.68 um tests
  the in-phase row buildup (the honest "photonic crystal" version).
  Box y=16 um so the side FF monitor (auto at y_span/2 - 0.8*lam = 6.75 um)
  sits OUTSIDE the comb and measures what still escapes.

REGISTERED PREDICTIONS (pre-dispatch): P1 the d-ladder is NON-monotonic with
~2.7 um period (interference signature; every failed drain family was monotonic).
P2 needle-bin FF power (|ux|>0.95) drops at the best d. Parasitic risk: comb
out-couples the guided tail ~ e^(-2*gamma*d), worst at d=3.0. Ceiling: needle
band carries 21% of W800 side power -> dT bound ~ +0.01..0.02.

Dispatch (ONE command; queue empty of other --option3 arrays):
    PRELIM_TIME=01:00:00 ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_h_retrocomb --max-concurrent=3
Output -> results/scat_h_retrocomb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAMBDA_X_NM  = 551.0          # = lambda_res/(2*n_clad*0.980), MEASURED-derived
COMB_R_NM    = 110.0
N_HALF       = 75             # sites at x = k*Lambda, k=-75..75 -> 151, +/-41.3 um
ROW_CYCLE_NM = 2680.0         # in-phase row spacing = lambda/(2*n_clad*sin(theta))
BOX_Y_UM     = 16.0           # side FF monitor auto-lands at 6.75 um (outside comb)

D_SCAN_NM    = [3000.0, 3700.0, 4400.0, 5100.0]      # single-row standoffs
D_TWO_ROW    = [3000.0, 3000.0 + ROW_CYCLE_NM]       # 2-row buildup variant

COMB_X_NM = [round(k * LAMBDA_X_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

# Own sidecar — box change (6.8 -> 16 um) shifts numerics; never reuse the
# program sidecar across numerics (CLAUDE.md section 2 / _common runbook).
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

# Port-monitor DFT memory scales with (transverse box) x (n_wl_points): at the
# 16 um box the program's 3001 points OOM-killed the first prelim at 133 GB
# (job 123347, ReqMem 128G). 1501 points + window cut 30 -> 20 nm (user,
# 2026-07-18) = 13 pm sampling (~90 samples across the 1.2 nm FWHM), ~70 GB.
# Window 1548.5-1568.5 nm around the 1558.6 W800 resonance.
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

PRELIM_BASE = _common.build_ports_base()
PRELIM_BASE.y_span_override_m = BOX_Y_UM * 1e-6
PRELIM_BASE.spectral.n_wl_points = N_WL_POINTS
PRELIM_BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
PRELIM_SPEC = SweepSpec(
    scatterer_radius_nm = [0.0],
    label = "scat_h_prelim",
)

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

# rows: (radius, x list, y list)
ROWS = [(0.0, [0.0], [3000.0])]                                   # control
ROWS += [(COMB_R_NM, COMB_X_NM, [d] * len(COMB_X_NM)) for d in D_SCAN_NM]
ROWS += [(COMB_R_NM, COMB_X_NM * 2,
          [D_TWO_ROW[0]] * len(COMB_X_NM) + [D_TWO_ROW[1]] * len(COMB_X_NM))]

_PML_CLEAR_NM = 1200.0        # > lambda/n_clad = 1080
for r, xs, ys in ROWS:
    assert max(ys) + max(r, 1.0) + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, \
        f"comb row too close to the y PML: max y {max(ys)}"
    assert min(ys) - r >= 2500.0, "standoff below the 2.5 um near-field floor"

_keys = {(r, len(xs), round(min(ys), 1), round(max(ys), 1)) for r, xs, ys in ROWS}
assert len(_keys) == len(ROWS), "stage-H rows must differ in (r, N, y endpoints)"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _, _ in ROWS],
    scatterer_x_list_nm = [xs for _, xs, _ in ROWS],
    scatterer_y_list_nm = [ys for _, _, ys in ROWS],
    mode  = "zipped",
    label = "scat_h_retrocomb",
)

if __name__ == "__main__":
    print(PRELIM_SPEC.describe())
    print(SPEC.describe())
    print(f"comb: {len(COMB_X_NM)} sites/row, Lambda {LAMBDA_X_NM} nm, span "
          f"+/-{COMB_X_NM[-1]/1000:.1f} um; d-scan {D_SCAN_NM}; 2-row {D_TWO_ROW}")
